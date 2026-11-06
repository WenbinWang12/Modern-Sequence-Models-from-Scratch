"""Gated DeltaNet 的因果递归实现。"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .layers import init_linear, init_norm, linear, rms_norm


class DeltaCache(NamedTuple):
    matrix: jax.Array
    convolution: jax.Array


def init_gated_delta(key, config):
    keys = jax.random.split(key, 7)
    width = config.num_attention_heads * config.head_dim
    params = {
        name: init_linear(keys[index], config.hidden_size, width, config)
        for index, name in enumerate(("q_proj", "k_proj", "v_proj", "gate_proj"))
    }
    params["beta_proj"] = init_linear(keys[4], config.hidden_size, config.num_attention_heads, config)
    params["decay_proj"] = init_linear(keys[5], config.hidden_size, config.num_attention_heads, config, True)
    params["o_proj"] = init_linear(keys[6], width, config.hidden_size, config)
    params["norm"] = init_norm(config.head_dim, config)
    params["conv_weight"] = jnp.zeros((config.gdn_conv_kernel_size, 3 * width), dtype=config.param_dtype).at[-1].set(1)
    return params


def empty_delta_cache(config):
    return DeltaCache(
        jnp.zeros((config.num_attention_heads, config.head_dim, config.head_dim), dtype=jnp.float32),
        jnp.zeros(
            (config.gdn_conv_kernel_size - 1, 3 * config.num_attention_heads * config.head_dim),
            dtype=config.compute_dtype,
        ),
    )


def gated_delta(params, hidden, config, cache=None):
    initial = empty_delta_cache(config) if cache is None else cache
    width = config.num_attention_heads * config.head_dim
    projected = jnp.concatenate(
        tuple(linear(params[name], hidden, config.compute_dtype) for name in ("q_proj", "k_proj", "v_proj")), axis=-1
    )
    beta = jax.nn.sigmoid(linear(params["beta_proj"], hidden, config.compute_dtype)).astype(jnp.float32)
    decay = jax.nn.sigmoid(linear(params["decay_proj"], hidden, config.compute_dtype)).astype(jnp.float32)

    def step(state, batch):
        current, current_beta, current_decay = batch
        history = jnp.concatenate((state.convolution, current[None]), axis=0)
        current = jax.nn.silu(jnp.sum(history * params["conv_weight"].astype(history.dtype), axis=0))
        query, key, value = jnp.split(
            current.reshape(3, config.num_attention_heads, config.head_dim).astype(jnp.float32), 3
        )
        query, key, value = query[0], key[0], value[0]
        query *= jax.lax.rsqrt(jnp.sum(query * query, axis=-1, keepdims=True) + 1e-6)
        key *= jax.lax.rsqrt(jnp.sum(key * key, axis=-1, keepdims=True) + 1e-6)
        matrix = state.matrix * current_decay[:, None, None]
        error = value - jnp.einsum("hvd,hd->hv", matrix, key)
        matrix += current_beta[:, None, None] * jnp.einsum("hv,hd->hvd", error, key)
        output = jnp.einsum("hvd,hd->hv", matrix, query)
        next_convolution = history[1:] if config.gdn_conv_kernel_size > 1 else history[:0]
        return DeltaCache(matrix, next_convolution), output.astype(config.compute_dtype)

    final, output = jax.lax.scan(jax.checkpoint(step), initial, (projected, beta, decay))
    output = rms_norm(output, params["norm"], config.rms_norm_eps).reshape(hidden.shape[0], width)
    output *= jax.nn.silu(linear(params["gate_proj"], hidden, config.compute_dtype))
    return linear(params["o_proj"], output, config.compute_dtype), final if cache is not None else None
