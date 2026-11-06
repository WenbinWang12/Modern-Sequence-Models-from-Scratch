"""共享专家权重经 vmap 后仍能求完整二阶导。"""

import jax
import jax.numpy as jnp
import numpy as np

from ttt.models.ttt_e2e.grouped_gemm import grouped_dot


def test_batched_shared_experts_value_grad_hvp():
    lhs = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 3))
    rhs = jax.random.normal(jax.random.PRNGKey(2), (3, 3, 4))
    sizes = jnp.asarray([[2, 0, 3], [1, 3, 1]], dtype=jnp.int32)

    def reference(x, weights, counts):
        groups = jnp.repeat(jnp.arange(3), counts, total_repeat_length=5)
        return jax.vmap(lambda row, index: row @ weights[index])(x, groups)

    def objective(function, weights):
        values = jax.vmap(function, in_axes=(0, None, 0))(lhs, weights, sizes)
        return jnp.sum(jnp.sin(values))

    def actual(weights):
        return objective(grouped_dot, weights)

    def expected(weights):
        return objective(reference, weights)

    np.testing.assert_allclose(actual(rhs), expected(rhs), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(jax.grad(actual)(rhs), jax.grad(expected)(rhs), rtol=1e-5, atol=1e-5)
    direction = jnp.ones_like(rhs) * 0.01
    observed = jax.jvp(jax.grad(actual), (rhs,), (direction,))[1]
    wanted = jax.jvp(jax.grad(expected), (rhs,), (direction,))[1]
    np.testing.assert_allclose(observed, wanted, rtol=1e-4, atol=1e-5)
