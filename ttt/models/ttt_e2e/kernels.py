"""可进行二阶求导的分块注意力。"""

import jax
import jax.numpy as jnp


def reference_attention(query, key, value, query_positions, key_positions, window=None):
    mask = (key_positions[None, :] >= 0) & (key_positions[None, :] <= query_positions[:, None])
    if window is not None:
        mask &= query_positions[:, None] - key_positions[None, :] < window
    scores = jnp.einsum("thd,shd->hts", query, key).astype(jnp.float32) / query.shape[-1] ** 0.5
    weights = jax.nn.softmax(jnp.where(mask[None], scores, -jnp.inf), axis=-1).astype(value.dtype)
    return jnp.einsum("hts,shd->thd", weights, value)


def blocked_sliding_attention(query, key, value, positions, window):
    """无缓存的滑窗注意力：按 window 切块，每块只与上一块和本块的键做稠密注意力。

    位置 p 的查询只能看到 (p - window, p] 内的键，这些键必然落在查询所在块及其前一块中，
    因此结果与 reference_attention 完全一致，但计算量从 O(T^2) 降为 O(T * window)，
    且各块之间没有串行依赖。
    """
    length, heads, dimension = query.shape
    pad = (-length) % window
    blocks = (length + pad) // window

    def split(values):
        return jnp.pad(values, ((0, pad), (0, 0), (0, 0))).reshape(blocks, window, heads, dimension)

    def with_previous(values, fill):
        return jnp.concatenate((jnp.concatenate((jnp.full_like(values[:1], fill), values[:-1])), values), axis=1)

    # 补齐的查询沿用连续位置，保证每行至少有一个可见键；补齐的键标记为 -1，永远不可见。
    query_positions = (positions[0] + jnp.arange(length + pad, dtype=positions.dtype)).reshape(blocks, window)
    key_positions = jnp.pad(positions, (0, pad), constant_values=-1).reshape(blocks, window)
    output = jax.vmap(reference_attention, in_axes=(0, 0, 0, 0, 0, None))(
        split(query),
        with_previous(split(key), 0),
        with_previous(split(value), 0),
        query_positions,
        with_previous(key_positions, -1),
        window,
    )
    return output.reshape(-1, heads, dimension)[:length]


def chunked_attention(
    query, key, value, query_positions, key_positions, *, window=None, query_block=128, key_block=256
):
    q_length, heads, dimension = query.shape
    k_length = key.shape[0]
    query_block = min(query_block, q_length)
    key_block = min(key_block, k_length)
    q_pad = (-q_length) % query_block
    k_pad = (-k_length) % key_block
    queries = jnp.pad(query, ((0, q_pad), (0, 0), (0, 0))).reshape(-1, query_block, heads, dimension)
    keys = jnp.pad(key, ((0, k_pad), (0, 0), (0, 0))).reshape(-1, key_block, heads, dimension)
    values = jnp.pad(value, ((0, k_pad), (0, 0), (0, 0))).reshape(keys.shape)
    q_positions = jnp.pad(query_positions, (0, q_pad), constant_values=-1).reshape(-1, query_block)
    k_positions = jnp.pad(key_positions, (0, k_pad), constant_values=-1).reshape(-1, key_block)
    minimum = jnp.finfo(jnp.float32).min

    def query_step(unused, query_batch):
        current_query, positions = query_batch
        initial = (
            jnp.full((heads, query_block), minimum),
            jnp.zeros((heads, query_block)),
            jnp.zeros((heads, query_block, dimension)),
        )

        def key_step(carry, key_batch):
            maximum, denominator, numerator = carry
            current_key, current_value, current_positions = key_batch
            allowed = (current_positions[None] >= 0) & (current_positions[None] <= positions[:, None])
            allowed &= positions[:, None] >= 0
            if window is not None:
                allowed &= positions[:, None] - current_positions[None] < window
            scores = jnp.einsum("thd,shd->hts", current_query, current_key).astype(jnp.float32)
            scores = jnp.where(allowed[None], scores / dimension**0.5, minimum)
            next_maximum = jnp.maximum(maximum, jnp.max(scores, axis=-1))
            previous_scale = jnp.exp(maximum - next_maximum)
            weights = jnp.where(allowed[None], jnp.exp(scores - next_maximum[..., None]), 0.0)
            denominator = previous_scale * denominator + jnp.sum(weights, axis=-1)
            numerator = previous_scale[..., None] * numerator + jnp.einsum(
                "hts,shd->htd",
                weights,
                current_value.astype(jnp.float32),
            )
            return (next_maximum, denominator, numerator), None

        (_, denominator, numerator), _ = jax.lax.scan(
            jax.checkpoint(key_step),
            initial,
            (keys, values, k_positions),
        )
        output = numerator / jnp.maximum(denominator[..., None], 1e-20)
        return unused, jnp.transpose(output, (1, 0, 2)).astype(value.dtype)

    _, output = jax.lax.scan(jax.checkpoint(query_step), None, (queries, q_positions))
    return output.reshape(-1, heads, dimension)[:q_length]
