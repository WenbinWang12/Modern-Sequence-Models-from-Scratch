"""GDN、MoE、路由与 Loop 的核心性质。"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ttt.inner_loop import InnerLoopConfig, sequence_loss
from ttt.models.ttt_e2e.layers import swiglu
from ttt.models.ttt_e2e.modeling_ttt_e2e import TTTE2EForCausalLM
from ttt.models.ttt_e2e.moe import moe_forward
from ttt.runtime import TTTSession


@pytest.mark.parametrize(
    "mode", ["gdn", "moe", "router", "delta_router", "early_ffn", "middle_block", "middle_attention"]
)
def test_extended_streaming_and_gradients(model_and_params, mode):
    original, _ = model_and_params
    changes = {"attention_backend": "reference"}
    if mode == "gdn":
        changes.update(layer_types=("gated_delta", "full_attention"), use_fast_mlp=False, ttt_suffix_length=0)
    elif mode in ("moe", "router", "delta_router"):
        changes.update(num_experts=2, num_experts_per_token=2, first_k_dense_layers=0)
        if mode != "moe":
            changes.update(adaptation_target=mode, use_fast_mlp=False, ttt_suffix_length=1 if mode == "router" else 0)
    else:
        changes.update(loop_mode=mode, loop_start=0, loop_end=1)
    config = replace(original.config, **changes)
    model = TTTE2EForCausalLM(config)
    params = model.init_parameters(jax.random.PRNGKey(2))
    tokens = jnp.asarray([1, 2, 3, 4, 5])
    inner = InnerLoopConfig(learning_rate=0.1)
    result = sequence_loss(model, params, tokens, inner, return_logits=True)
    session = TTTSession(model, params, inner, jit=False)
    actual = jnp.stack([session.decode(int(token)) for token in tokens])
    np.testing.assert_allclose(actual[:-1], result.logits, atol=2e-5, rtol=1e-4)
    gradients = jax.grad(lambda weights: sequence_loss(model, weights, tokens, inner).loss)(params)
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in jax.tree.leaves(gradients))


def test_grouped_experts_equal_explicit_mixture(model_and_params):
    original, _ = model_and_params
    config = replace(
        original.config, num_experts=3, num_experts_per_token=2, first_k_dense_layers=0, num_shared_experts=0
    )
    model = TTTE2EForCausalLM(config)
    params = model.init_parameters(jax.random.PRNGKey(3))["layers"][0]["moe"]
    hidden = jax.random.normal(jax.random.PRNGKey(4), (5, config.hidden_size))
    actual, _, _ = moe_forward(params, hidden, config)
    logits = hidden @ params["router"]["weight"].T
    values, selected = jax.lax.top_k(logits, 2)
    weights = jax.nn.softmax(values, axis=-1)
    expected = []
    for row in range(hidden.shape[0]):
        mixture = jnp.zeros(config.hidden_size)
        for visit in range(2):
            expert = jax.tree.map(lambda value: value[selected[row, visit]], params["experts"])
            mixture += weights[row, visit] * swiglu(expert, hidden[row : row + 1], config.compute_dtype)[0]
        expected.append(mixture)
    np.testing.assert_allclose(actual, jnp.stack(expected), rtol=1e-4, atol=1e-6)
