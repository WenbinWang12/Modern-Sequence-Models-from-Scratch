"""无丢 token 的分组专家计算和记忆路由。"""

import jax
import jax.numpy as jnp

from .grouped_gemm import grouped_dot
from .layers import init_linear, init_mlp, linear, swiglu


def init_router_adapter(key, config):
    keys = jax.random.split(key)
    return {
        "up": init_linear(keys[0], config.hidden_size, config.router_adapter_rank, config),
        "down": {"weight": jnp.zeros((config.num_experts, config.router_adapter_rank), dtype=config.param_dtype)},
    }


def init_moe(key, config):
    keys = jax.random.split(key, config.num_experts + 4)
    params = {
        "router": init_linear(keys[0], config.hidden_size, config.num_experts, config),
        "load_bias": jnp.zeros(config.num_experts, dtype=jnp.float32),
        "experts": jax.vmap(lambda current: init_mlp(current, config, config.moe_intermediate_size))(
            keys[1 : config.num_experts + 1]
        ),
    }
    if config.num_shared_experts:
        params["shared"] = init_mlp(keys[-3], config, config.moe_intermediate_size * config.num_shared_experts)
    if config.adaptation_target == "delta_router":
        rank = config.delta_memory_rank
        memory_keys = jax.random.split(keys[-2], 5)
        params["memory"] = {
            name: init_linear(memory_keys[index], config.hidden_size, rank, config)
            for index, name in enumerate(("key", "value", "query"))
        }
        params["memory"]["beta"] = init_linear(memory_keys[3], config.hidden_size, 1, config)
        params["memory"]["router"] = {"weight": jnp.zeros((config.num_experts, rank), dtype=config.param_dtype)}
        params["memory"]["output"] = {"weight": jnp.zeros((config.hidden_size, rank), dtype=config.param_dtype)}
    return params


def delta_read(params, hidden, state, config):
    rank = config.delta_memory_rank
    if state is None:
        state = jnp.zeros((rank, rank), dtype=jnp.float32)
    key, value, query = (
        linear(params[name], hidden, config.compute_dtype).astype(jnp.float32) for name in ("key", "value", "query")
    )
    key *= jax.lax.rsqrt(jnp.sum(key * key, axis=-1, keepdims=True) + 1e-6)
    beta = jax.nn.sigmoid(linear(params["beta"], hidden, config.compute_dtype)).astype(jnp.float32)

    def step(matrix, row):
        k, v, q, b = row
        matrix = matrix + b * jnp.outer(v - matrix @ k, k)
        return matrix, matrix @ q

    return jax.lax.scan(jax.checkpoint(step), state, (key, value, query, beta))


def _grouped_linear(params, hidden, group_sizes, dtype):
    weight = jnp.swapaxes(params["weight"].astype(dtype), -1, -2)
    output = grouped_dot(hidden.astype(dtype), weight, group_sizes)
    if "bias" in params:
        ids = jnp.repeat(jnp.arange(group_sizes.shape[0]), group_sizes, total_repeat_length=hidden.shape[0])
        output += params["bias"][ids].astype(dtype)
    return output


def moe_forward(params, hidden, config, fast_router=None, memory=None, token_mask=None):
    logits = linear(params["router"], hidden, config.compute_dtype).astype(jnp.float32)
    next_memory, memory_output = memory, None
    if fast_router is not None:
        correction = linear(fast_router["up"], hidden, config.compute_dtype)
        logits += linear(fast_router["down"], correction, config.compute_dtype).astype(jnp.float32)
    if "memory" in params:
        next_memory, reads = delta_read(params["memory"], hidden, memory, config)
        logits += linear(params["memory"]["router"], reads, config.compute_dtype).astype(jnp.float32)
        memory_output = linear(params["memory"]["output"], reads, config.compute_dtype)
    probabilities = jax.nn.softmax(logits, axis=-1)
    _, selected = jax.lax.top_k(logits + params["load_bias"], config.num_experts_per_token)
    weights = jnp.take_along_axis(probabilities, selected, axis=-1)
    weights /= jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-12)
    expert_ids = selected.reshape(-1)
    order = jnp.argsort(expert_ids, stable=True)
    sorted_ids = expert_ids[order]
    group_sizes = jnp.bincount(sorted_ids, length=config.num_experts)
    token_ids = jnp.repeat(jnp.arange(hidden.shape[0]), config.num_experts_per_token)[order]
    dispatched = hidden[token_ids]
    experts = params["experts"]
    gate = jax.nn.silu(_grouped_linear(experts["gate_proj"], dispatched, group_sizes, config.compute_dtype))
    up = _grouped_linear(experts["up_proj"], dispatched, group_sizes, config.compute_dtype)
    grouped = _grouped_linear(experts["down_proj"], gate * up, group_sizes, config.compute_dtype)
    restored = jnp.zeros_like(grouped).at[order].set(grouped)
    output = jnp.sum(
        restored.reshape(hidden.shape[0], config.num_experts_per_token, config.hidden_size) * weights[..., None], axis=1
    )
    if "shared" in params:
        output += swiglu(params["shared"], hidden, config.compute_dtype)
    if memory_output is not None:
        output += memory_output
    if token_mask is None:
        token_mask = jnp.ones(hidden.shape[0], dtype=jnp.float32)
    token_mask = token_mask.astype(jnp.float32)
    loads = jnp.bincount(
        expert_ids, weights=jnp.repeat(token_mask, config.num_experts_per_token), length=config.num_experts
    )
    frequencies = loads / jnp.maximum(jnp.sum(loads), 1)
    average = jnp.sum(probabilities * token_mask[:, None], axis=0) / jnp.maximum(jnp.sum(token_mask), 1)
    aux = config.num_experts * jnp.sum(average * jax.lax.stop_gradient(frequencies))
    z_loss = jnp.sum(jax.nn.logsumexp(logits, axis=-1) ** 2 * token_mask) / jnp.maximum(jnp.sum(token_mask), 1)
    return output.astype(config.compute_dtype), next_memory, {"loads": loads, "aux": aux, "z": z_loss}
