"""多机初始化、参数分片和全局批次。"""

import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

_INITIALIZED = False


def initialize_distributed():
    global _INITIALIZED
    names = ("JAX_COORDINATOR_ADDRESS", "JAX_PROCESS_COUNT", "JAX_PROCESS_ID")
    values = [os.environ.get(name) for name in names]
    if not any(values) or _INITIALIZED:
        return
    if not all(values):
        raise ValueError("多机运行必须同时设置 JAX_COORDINATOR_ADDRESS、JAX_PROCESS_COUNT 和 JAX_PROCESS_ID。")
    jax.distributed.initialize(coordinator_address=values[0], num_processes=int(values[1]), process_id=int(values[2]))
    _INITIALIZED = True


def create_mesh(expert_parallel_size=1):
    devices = np.asarray(jax.devices())
    if devices.size % expert_parallel_size:
        raise ValueError("全局设备数必须能被 expert_parallel_size 整除。")
    return Mesh(devices.reshape(-1, expert_parallel_size), ("data", "expert"))


def tree_shardings(tree, mesh, *, shard_parameters=True):
    def specification(path, value):
        axes = [None] * len(value.shape)
        expert = "experts" in jax.tree_util.keystr(path)
        if expert and value.ndim >= 2:
            if value.shape[0] % mesh.shape["expert"]:
                raise ValueError("专家数必须能被专家并行数整除。")
            axes[0] = "expert"
            if shard_parameters and value.shape[1] % mesh.shape["data"] == 0:
                axes[1] = "data"
        elif shard_parameters and value.ndim >= 2 and value.shape[0] % mesh.shape["data"] == 0:
            axes[0] = "data"
        return NamedSharding(mesh, PartitionSpec(*axes))

    return jax.tree_util.tree_map_with_path(specification, tree)


def mesh_state_shapes(model, optimizer, key, mesh, *, shard_parameters=True):
    from .training import TrainState

    parameters = jax.eval_shape(model.init_parameters, key)
    state_shape = TrainState(
        parameters, jax.eval_shape(optimizer.init, parameters), jax.ShapeDtypeStruct((), jnp.int32)
    )
    state_shardings = tree_shardings(state_shape, mesh, shard_parameters=shard_parameters)
    return state_shape, state_shardings


def initialize_mesh_state(model, optimizer, key, mesh, *, shard_parameters=True):
    from .training import TrainState

    _, state_shardings = mesh_state_shapes(model, optimizer, key, mesh, shard_parameters=shard_parameters)
    params = jax.jit(model.init_parameters, out_shardings=state_shardings.params)(key)
    opt_state = jax.jit(optimizer.init, out_shardings=state_shardings.opt_state)(params)
    step = jax.device_put(jnp.asarray(0, dtype=jnp.int32), state_shardings.step)
    return TrainState(params, opt_state, step), state_shardings


def make_mesh_train_step(model, inner, training, optimizer, mesh, state_shardings):
    from .training import make_train_step

    replicated = NamedSharding(mesh, PartitionSpec())
    batch = NamedSharding(mesh, PartitionSpec("data", None))
    function = make_train_step(model, inner, training, optimizer, jit=False)
    return jax.jit(
        function,
        in_shardings=(state_shardings, batch, batch),
        out_shardings=(state_shardings, replicated),
        donate_argnums=(0,),
    )


def global_batch(stream, step, mesh, batch_size):
    if batch_size % mesh.shape["data"]:
        raise ValueError("全局 batch_size 必须能被数据并行数整除。")
    sharding = NamedSharding(mesh, PartitionSpec("data", None))
    cache = {}

    def rows(indices):
        current = tuple(range(*indices[0].indices(batch_size)))
        if current not in cache:
            cache[current] = stream.sample_rows(step, current)
        return cache[current]

    tokens = jax.make_array_from_callback(
        (batch_size, stream.sequence_length + 1),
        sharding,
        lambda indices: rows(indices)[0],
    )
    masks = jax.make_array_from_callback(
        (batch_size, stream.sequence_length),
        sharding,
        lambda indices: rows(indices)[1],
    )
    return tokens, masks


def redistribute(tree, shardings):
    return jax.tree.map(
        lambda value, sharding: jax.make_array_from_callback(
            value.shape, sharding, lambda indices: np.asarray(value)[indices]
        ),
        tree,
        shardings,
    )
