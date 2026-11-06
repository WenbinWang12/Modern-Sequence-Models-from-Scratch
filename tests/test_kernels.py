"""分块算子的值、梯度和二阶导。"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ttt.inner_loop import token_cross_entropy
from ttt.loss import chunked_cross_entropy
from ttt.models.ttt_e2e.kernels import chunked_attention, reference_attention


@pytest.mark.parametrize("window", [None, 3])
def test_chunked_attention_derivatives(window):
    keys = jax.random.split(jax.random.PRNGKey(3), 3)
    query = jax.random.normal(keys[0], (5, 2, 4))
    key = jax.random.normal(keys[1], (7, 2, 4))
    value = jax.random.normal(keys[2], (7, 2, 4))
    positions = jnp.arange(5) + 2
    key_positions = jnp.arange(7)

    def tiled(q):
        return chunked_attention(q, key, value, positions, key_positions, window=window, query_block=2, key_block=3)

    def reference(q):
        return reference_attention(q, key, value, positions, key_positions, window)

    np.testing.assert_allclose(tiled(query), reference(query), rtol=1e-5, atol=1e-6)
    first = jax.grad(lambda q: jnp.sum(tiled(q) ** 2))
    second = jax.grad(lambda q: jnp.sum(reference(q) ** 2))
    np.testing.assert_allclose(first(query), second(query), rtol=1e-4, atol=2e-6)
    direction = jnp.ones_like(query)
    np.testing.assert_allclose(
        jax.jvp(first, (query,), (direction,))[1], jax.jvp(second, (query,), (direction,))[1], rtol=1e-4, atol=3e-6
    )


def test_chunked_cross_entropy_derivatives():
    hidden = jax.random.normal(jax.random.PRNGKey(5), (5, 4))
    weight = jax.random.normal(jax.random.PRNGKey(6), (11, 4))
    targets = jnp.asarray([1, 3, 5, 7, 9])
    mask = jnp.asarray([1, 1, 1, 1, 0], dtype=jnp.float32)

    def tiled(x):
        return chunked_cross_entropy(x, weight, targets, mask, token_block=2, vocab_block=3)

    def reference(x):
        return token_cross_entropy(x @ weight.T, targets, mask)

    np.testing.assert_allclose(tiled(hidden), reference(hidden), rtol=1e-5, atol=1e-6)
    gradient = jax.grad(lambda x: jnp.sum(tiled(x)))
    expected = jax.grad(lambda x: jnp.sum(reference(x)))
    np.testing.assert_allclose(gradient(hidden), expected(hidden), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(
        jax.jvp(gradient, (hidden,), (jnp.ones_like(hidden),))[1],
        jax.jvp(expected, (hidden,), (jnp.ones_like(hidden),))[1],
        rtol=1e-4,
        atol=2e-6,
    )
