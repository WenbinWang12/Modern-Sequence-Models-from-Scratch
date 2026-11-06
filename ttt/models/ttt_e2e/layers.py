"""线性层、RMSNorm 和 SwiGLU。"""

import jax
import jax.numpy as jnp


def init_linear(key, in_features, out_features, config, bias=False):
    weight = jax.random.normal(key, (out_features, in_features), dtype=jnp.float32)
    params = {"weight": (weight * config.initializer_range).astype(config.param_dtype)}
    if bias:
        params["bias"] = jnp.zeros(out_features, dtype=config.param_dtype)
    return params


def linear(params, x, dtype):
    result = x.astype(dtype) @ params["weight"].astype(dtype).T
    if "bias" in params:
        result = result + params["bias"].astype(dtype)
    return result


def init_norm(size, config):
    return jnp.ones(size, dtype=config.param_dtype)


def rms_norm(x, weight, eps):
    values = x.astype(jnp.float32)
    values = values * jax.lax.rsqrt(jnp.mean(values * values, axis=-1, keepdims=True) + eps)
    return (values * weight.astype(jnp.float32)).astype(x.dtype)


def init_mlp(key, config, intermediate_size):
    keys = jax.random.split(key, 3)
    return {
        "gate_proj": init_linear(keys[0], config.hidden_size, intermediate_size, config, config.mlp_bias),
        "up_proj": init_linear(keys[1], config.hidden_size, intermediate_size, config, config.mlp_bias),
        "down_proj": init_linear(keys[2], intermediate_size, config.hidden_size, config, config.mlp_bias),
    }


def swiglu(params, x, dtype):
    gate = jax.nn.silu(linear(params["gate_proj"], x, dtype))
    up = linear(params["up_proj"], x, dtype)
    return linear(params["down_proj"], gate * up, dtype)
