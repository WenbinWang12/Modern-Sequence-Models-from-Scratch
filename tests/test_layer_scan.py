"""层扫描应与逐层执行保持前向、元梯度和流式状态一致。"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ttt.inner_loop import InnerLoopConfig, sequence_loss
from ttt.models.ttt_e2e import TTTE2EForCausalLM
from ttt.runtime import TTTSession


@pytest.mark.parametrize(
    "kind", ["dense", "moe", "gdn", "router", "delta_router", "early_ffn", "middle_block", "middle_attention"]
)
def test_layer_scan_matches_unrolled(model_and_params, kind):
    base, _ = model_and_params
    fields = dict(num_hidden_layers=4, ttt_suffix_length=2, layer_types=("sliding_attention",) * 4)
    if kind == "gdn":
        fields.update(layer_types=("gated_delta",) * 4, use_fast_mlp=False, ttt_suffix_length=0)
    if kind in ("moe", "router", "delta_router"):
        fields.update(num_experts=2, first_k_dense_layers=0)
    if kind in ("router", "delta_router"):
        fields.update(use_fast_mlp=False, adaptation_target=kind, ttt_suffix_length=0 if kind == "delta_router" else 2)
    if kind in ("early_ffn", "middle_block", "middle_attention"):
        fields.update(loop_mode=kind, loop_start=0, loop_end=2)
    config = replace(base.config, **fields)
    scanned, reference = TTTE2EForCausalLM(config, True), TTTE2EForCausalLM(config, False)
    params = reference.init_parameters(jax.random.PRNGKey(31))
    tokens = jnp.asarray([1, 2, 3, 4, 5], dtype=jnp.int32)
    inner = InnerLoopConfig(learning_rate=0.1)
    actual = sequence_loss(scanned, params, tokens, inner, return_logits=True)
    expected = sequence_loss(reference, params, tokens, inner, return_logits=True)
    np.testing.assert_allclose(actual.logits, expected.logits, rtol=1e-4, atol=2e-5)
    np.testing.assert_allclose(actual.router_loads, expected.router_loads, atol=0)
    functions = [
        lambda weights, model=model: sequence_loss(model, weights, tokens, inner).loss for model in (scanned, reference)
    ]
    observed, wanted = [jax.jit(jax.grad(function))(params) for function in functions]
    for a, b in zip(jax.tree.leaves(observed), jax.tree.leaves(wanted)):
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=2e-5)
    session = TTTSession(scanned, params, inner, jit=True)
    logits = jnp.stack([session.decode(int(token)) for token in tokens])
    np.testing.assert_allclose(logits[:-1], expected.logits, rtol=1e-4, atol=2e-5)
