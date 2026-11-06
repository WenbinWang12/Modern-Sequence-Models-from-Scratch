"""内循环和外循环梯度。"""

import jax
import jax.numpy as jnp
import numpy as np

from ttt.inner_loop import InnerLoopConfig, sequence_loss, sgd_update


def test_sgd_retains_second_order_derivative():
    def outer_loss(initial):
        weights = ({"weight": initial},)
        gradient = jax.grad(lambda value: 0.5 * (value - 1.0) ** 2)(initial)
        updated, _ = sgd_update(weights, ({"weight": gradient},), 0.1, 100.0)
        return 0.5 * (updated[0]["weight"] - 3.0) ** 2

    np.testing.assert_allclose(jax.grad(outer_loss)(jnp.asarray(2.0)), -0.99, atol=1e-6)


def test_zero_gradient_second_derivative_is_finite():
    def objective(value):
        updated, _ = sgd_update((value,), (value * 0,), 1.0, 1.0)
        return updated[0] ** 2

    assert np.isfinite(float(jax.grad(jax.grad(objective))(jnp.asarray(0.0))))


def test_model_gradient_matches_finite_difference(model_and_params):
    model, params = model_and_params
    tokens = jnp.asarray([1, 2, 3, 4, 5])
    inner = InnerLoopConfig(learning_rate=0.2, clip_norm=100.0)
    base = params["layers"][-1]["fast_mlp"]["down_proj"]["weight"]

    def objective(value):
        layers = list(params["layers"])
        last = dict(layers[-1])
        fast = dict(last["fast_mlp"])
        down = dict(fast["down_proj"])
        down["weight"] = base.at[0, 0].set(value)
        fast["down_proj"] = down
        last["fast_mlp"] = fast
        layers[-1] = last
        modified = {**params, "layers": tuple(layers)}
        return sequence_loss(model, modified, tokens, inner).loss

    value = base[0, 0]
    analytic = jax.grad(objective)(value)
    epsilon = 0.005
    numeric = (objective(value + epsilon) - objective(value - epsilon)) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numeric, rtol=0.03, atol=0.0001)
