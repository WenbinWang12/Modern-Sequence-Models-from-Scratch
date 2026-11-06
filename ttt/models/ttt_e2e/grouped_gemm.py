"""分组矩阵乘法及可批处理的高阶导。"""

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap


@custom_vmap
def _native(lhs, rhs, sizes):
    return jax.lax.ragged_dot(lhs, rhs, sizes)


@_native.def_vmap
def _native_batch(axis_size, in_batched, lhs, rhs, sizes):
    values = tuple(
        value if batched else jnp.broadcast_to(value, (axis_size, *value.shape))
        for value, batched in zip((lhs, rhs, sizes), in_batched)
    )
    return jax.vmap(jax.lax.ragged_dot)(*values), True


def _differentiable(lhs, rhs, sizes):
    groups = jnp.repeat(jnp.arange(sizes.shape[0]), sizes, total_repeat_length=lhs.shape[0])

    result = jnp.zeros((lhs.shape[0], rhs.shape[-1]), dtype=lhs.dtype)
    for index in range(rhs.shape[0]):
        result = result + jnp.where((groups == index)[:, None], lhs @ rhs[index], 0)
    return result


@jax.custom_jvp
def grouped_dot(lhs, rhs, sizes):
    return _native(lhs, rhs, sizes)


@grouped_dot.defjvp
def _grouped_jvp(primals, tangents):
    lhs, rhs, sizes = primals
    lhs_tangent, rhs_tangent, _ = tangents
    # 求导时前向也走参考路径，避免旧版 ragged_dot 在嵌套扫描中的批处理错误。
    value = _differentiable(lhs, rhs, sizes)
    tangent = _differentiable(lhs_tangent, rhs, sizes) + _differentiable(lhs, rhs_tangent, sizes)
    return value, tangent
