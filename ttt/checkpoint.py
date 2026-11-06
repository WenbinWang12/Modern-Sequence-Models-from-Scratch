"""保存参数、优化器和数据采样状态。"""

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import ExperimentConfig
from .models.ttt_e2e.modeling_ttt_e2e import TTTE2EForCausalLM
from .training import TrainState, make_optimizer


class Checkpoint(NamedTuple):
    experiment: ExperimentConfig
    state: TrainState
    data_state: dict | None


def _arrays(prefix, tree):
    return {f"{prefix}_{index:05d}": np.asarray(value) for index, value in enumerate(jax.tree.leaves(tree))}


def _restore(archive, prefix, template):
    leaves, structure = jax.tree.flatten(template)
    stored = [name for name in archive.files if name.startswith(prefix + "_")]
    if len(stored) != len(leaves):
        raise ValueError(f"{prefix} 的参数数量与配置不一致。")
    arrays = []
    for index, expected in enumerate(leaves):
        value = archive[f"{prefix}_{index:05d}"]
        if value.shape != expected.shape:
            raise ValueError(f"{prefix} 第 {index} 个数组的形状不匹配。")
        dtype = np.dtype(expected.dtype)
        # NumPy 的 npy 格式把 bfloat16 保存为等宽字节类型。
        if value.dtype.kind == "V" and value.dtype.itemsize == dtype.itemsize:
            value = value.view(dtype)
        if value.dtype != dtype:
            raise ValueError(f"{prefix} 第 {index} 个数组的精度不匹配。")
        arrays.append(jnp.asarray(value))
    return jax.tree.unflatten(structure, arrays)


def save_checkpoint(path, state, experiment, data_state=None):
    path = Path(path)
    if path.suffix != ".npz":
        return save_sharded_checkpoint(path, state, experiment, data_state)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 1,
        "experiment": experiment.to_dict(),
        "step": int(state.step),
        "data_state": data_state,
    }
    arrays = _arrays("params", state.params)
    if state.opt_state is not None:
        arrays.update(_arrays("optimizer", state.opt_state))
    arrays["metadata"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_checkpoint(path, *, load_optimizer=False, target_shardings=None):
    if Path(path).is_dir():
        return load_sharded_checkpoint(path, load_optimizer=load_optimizer, target_shardings=target_shardings)
    if target_shardings is not None:
        raise ValueError("直接分片恢复需要 Orbax 格式。")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata["format_version"] != 1:
            raise ValueError("不支持的 checkpoint 格式。")
        experiment = ExperimentConfig.from_dict(metadata["experiment"])
        model = TTTE2EForCausalLM(experiment.model)
        template = jax.eval_shape(model.init_parameters, jax.random.PRNGKey(0))
        params = _restore(archive, "params", template)
        optimizer_state = None
        if load_optimizer:
            optimizer = make_optimizer(experiment.training)
            optimizer_template = jax.eval_shape(optimizer.init, params)
            optimizer_state = _restore(archive, "optimizer", optimizer_template)
        state = TrainState(params, optimizer_state, jnp.asarray(metadata["step"], dtype=jnp.int32))
        return Checkpoint(experiment, state, metadata["data_state"])


def _orbax():
    try:
        import orbax.checkpoint as ocp
    except ImportError as error:
        raise ImportError("Orbax 检查点需要安装 checkpoint 扩展。") from error
    return ocp


def _checkpoint_identity():
    identity = uuid.uuid4().hex
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        shared = multihost_utils.broadcast_one_to_all(np.frombuffer(identity.encode("ascii"), dtype=np.uint8))
        # 广播中的整数归约可能提升 dtype，必须先恢复为字节再解码。
        identity = np.asarray(shared, dtype=np.uint8).tobytes().decode("ascii")
    return identity


def save_sharded_checkpoint(path, state, experiment, data_state=None):
    ocp = _orbax()
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"检查点目录已存在：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    from jax.experimental import multihost_utils

    identity = _checkpoint_identity()
    temporary = path.parent / f".{path.name}.{identity}.tmp"
    temporary.mkdir(exist_ok=True)
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(temporary / "params", state.params)
        checkpointer.wait_until_finished()
        if state.opt_state is not None:
            checkpointer.save(temporary / "optimizer", state.opt_state)
            checkpointer.wait_until_finished()
    if jax.process_index() == 0:
        metadata = {
            "format_version": 2,
            "experiment": experiment.to_dict(),
            "step": int(state.step),
            "data_state": data_state,
        }
        metadata_path = temporary / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    if jax.process_count() > 1:
        multihost_utils.sync_global_devices("checkpoint_committed_" + identity)


def _restore_target(tree, shardings=None):
    if shardings is not None:
        return jax.tree.map(
            lambda item, sharding: jax.ShapeDtypeStruct(item.shape, item.dtype, sharding=sharding), tree, shardings
        )
    sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
    return jax.tree.map(lambda item: jax.ShapeDtypeStruct(item.shape, item.dtype, sharding=sharding), tree)


def load_sharded_checkpoint(path, *, load_optimizer=False, target_shardings=None):
    ocp = _orbax()
    path = Path(path).resolve()
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata["format_version"] != 2:
        raise ValueError("不支持的分片检查点格式。")
    experiment = ExperimentConfig.from_dict(metadata["experiment"])
    model = TTTE2EForCausalLM(experiment.model)
    template = jax.eval_shape(model.init_parameters, jax.random.PRNGKey(0))
    with ocp.StandardCheckpointer() as checkpointer:
        params = checkpointer.restore(
            path / "params", target=_restore_target(template, target_shardings.params if target_shardings else None)
        )
        optimizer_state = None
        if load_optimizer:
            optimizer = make_optimizer(experiment.training)
            optimizer_template = jax.eval_shape(optimizer.init, template)
            optimizer_state = checkpointer.restore(
                path / "optimizer",
                target=_restore_target(optimizer_template, target_shardings.opt_state if target_shardings else None),
            )
    return Checkpoint(
        experiment,
        TrainState(
            params,
            optimizer_state,
            jax.device_put(jnp.asarray(metadata["step"], dtype=jnp.int32), target_shardings.step)
            if target_shardings
            else jnp.asarray(metadata["step"], dtype=jnp.int32),
        ),
        metadata["data_state"],
    )
