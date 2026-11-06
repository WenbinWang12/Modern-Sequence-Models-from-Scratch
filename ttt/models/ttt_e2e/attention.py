"""因果注意力与 KV 缓存。"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .kernels import chunked_attention, reference_attention
from .layers import init_linear, init_norm, linear, rms_norm


class KVCache(NamedTuple):
    key: jax.Array
    value: jax.Array
    positions: jax.Array


def init_attention(key, config):
    keys = jax.random.split(key, 4)
    query_size = config.num_attention_heads * config.head_dim
    kv_size = config.num_key_value_heads * config.head_dim
    params = {
        "q_proj": init_linear(keys[0], config.hidden_size, query_size, config, config.attention_bias),
        "k_proj": init_linear(keys[1], config.hidden_size, kv_size, config, config.attention_bias),
        "v_proj": init_linear(keys[2], config.hidden_size, kv_size, config, config.attention_bias),
        "o_proj": init_linear(keys[3], query_size, config.hidden_size, config, config.attention_bias),
    }
    if config.qk_norm:
        params["q_norm"] = init_norm(config.head_dim, config)
        params["k_norm"] = init_norm(config.head_dim, config)
    if config.attention_gate:
        params["gate_proj"] = init_linear(jax.random.fold_in(key, 7), config.hidden_size, query_size, config)
    return params


def empty_cache(config, capacity):
    shape = (capacity, config.num_key_value_heads, config.head_dim)
    return KVCache(
        jnp.zeros(shape, dtype=config.compute_dtype),
        jnp.zeros(shape, dtype=config.compute_dtype),
        jnp.full((capacity,), -1, dtype=jnp.int32),
    )


def grow_cache(cache, capacity, sliding):
    extra = capacity - cache.key.shape[0]
    if extra <= 0:
        return cache
    # 滑窗把空位补在左侧，确保最近的历史始终在右侧。
    before, after = (extra, 0) if sliding else (0, extra)
    pads = ((before, after), (0, 0), (0, 0))
    return KVCache(
        jnp.pad(cache.key, pads),
        jnp.pad(cache.value, pads),
        jnp.pad(cache.positions, (before, after), constant_values=-1),
    )


def apply_rope(x, positions, theta):
    frequencies = theta ** (-jnp.arange(0, x.shape[-1], 2, dtype=jnp.float32) / x.shape[-1])
    angles = positions.astype(jnp.float32)[:, None, None] * frequencies[None, None, :]
    cosine, sine = jnp.cos(angles), jnp.sin(angles)
    even, odd = x[..., 0::2].astype(jnp.float32), x[..., 1::2].astype(jnp.float32)
    rotated = jnp.stack((even * cosine - odd * sine, even * sine + odd * cosine), axis=-1)
    return rotated.reshape(x.shape).astype(x.dtype)


def attention(params, hidden, config, layer_type, *, cache=None, position=0):
    count = hidden.shape[0]
    positions = jnp.arange(count, dtype=jnp.int32) + position
    query = linear(params["q_proj"], hidden, config.compute_dtype).reshape(
        count, config.num_attention_heads, config.head_dim
    )
    key = linear(params["k_proj"], hidden, config.compute_dtype).reshape(
        count, config.num_key_value_heads, config.head_dim
    )
    value = linear(params["v_proj"], hidden, config.compute_dtype).reshape(key.shape)
    if config.qk_norm:
        query = rms_norm(query, params["q_norm"], config.rms_norm_eps)
        key = rms_norm(key, params["k_norm"], config.rms_norm_eps)
    query = apply_rope(query, positions, config.rope_theta)
    key = apply_rope(key, positions, config.rope_theta)
    sliding = layer_type == "sliding_attention"

    if cache is None:
        keys, values, key_positions = key, value, positions
        next_cache = None
    elif sliding:
        keys = jnp.concatenate((cache.key, key), axis=0)
        values = jnp.concatenate((cache.value, value), axis=0)
        key_positions = jnp.concatenate((cache.positions, positions))
        capacity = cache.key.shape[0]
        next_cache = KVCache(keys[-capacity:], values[-capacity:], key_positions[-capacity:])
    else:
        # 全注意力缓存容量由调用方保证；JIT 内不改变缓存形状。
        keys = jax.lax.dynamic_update_slice(cache.key, key, (position, 0, 0))
        values = jax.lax.dynamic_update_slice(cache.value, value, (position, 0, 0))
        key_positions = jax.lax.dynamic_update_slice(cache.positions, positions, (position,))
        next_cache = KVCache(keys, values, key_positions)

    repeats = config.num_attention_heads // config.num_key_value_heads
    keys = jnp.repeat(keys, repeats, axis=1)
    values = jnp.repeat(values, repeats, axis=1)
    window = config.sliding_window if sliding else None
    if config.attention_backend == "reference":
        output = reference_attention(query, keys, values, positions, key_positions, window)
    elif config.attention_backend == "cudnn":
        if cache is not None:
            raise ValueError("cudnn 路径仅用于无缓存前向；内循环及推理应使用 chunked。")
        output = jax.nn.dot_product_attention(
            query,
            keys,
            values,
            is_causal=True,
            local_window_size=None if window is None else (window - 1, 0),
            implementation="cudnn",
        )
    else:
        output = chunked_attention(
            query,
            keys,
            values,
            positions,
            key_positions,
            window=window,
            query_block=config.attention_query_block_size,
            key_block=config.attention_key_block_size,
        )
    output = output.reshape(count, -1)
    if config.attention_gate:
        output *= jax.nn.sigmoid(linear(params["gate_proj"], hidden, config.compute_dtype))
    return linear(params["o_proj"], output, config.compute_dtype), next_cache
