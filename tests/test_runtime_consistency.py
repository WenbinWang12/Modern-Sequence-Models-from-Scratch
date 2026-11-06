"""分块输入、逐 token 输入与训练路径一致。"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ttt.inner_loop import InnerLoopConfig, sequence_loss
from ttt.models.ttt_e2e.modeling_ttt_e2e import TTTE2EForCausalLM
from ttt.runtime import TTTSession


@pytest.mark.parametrize("length", [5, 6, 9])
def test_streaming_matches_training(model_and_params, length):
    model, params = model_and_params
    inner = InnerLoopConfig(learning_rate=0.1)
    tokens = np.arange(1, length + 1, dtype=np.int32)
    expected = sequence_loss(model, params, jnp.asarray(tokens), inner, return_logits=True)
    session = TTTSession(model, params, inner, jit=False)
    actual = jnp.concatenate((session.observe(tokens[:1]), session.observe(tokens[1:4]), session.observe(tokens[4:])))
    np.testing.assert_allclose(actual[:-1], expected.logits, rtol=1e-4, atol=1e-5)
    for observed, reference in zip(jax.tree.leaves(session.state.fast_weights), jax.tree.leaves(expected.fast_weights)):
        np.testing.assert_allclose(observed, reference, rtol=1e-4, atol=1e-5)
    assert session.state.updates == (length - 1) // model.config.ttt_chunk_size


def test_full_attention_in_suffix_and_old_kv(model_and_params):
    original, _ = model_and_params
    config = replace(original.config, layer_types=("sliding_attention", "full_attention"), ttt_chunk_size=3)
    model = TTTE2EForCausalLM(config)
    params = model.init_parameters(jax.random.PRNGKey(2))
    tokens = np.arange(1, 11, dtype=np.int32)
    inner = InnerLoopConfig(learning_rate=0.1)
    expected = sequence_loss(model, params, jnp.asarray(tokens), inner, return_logits=True)
    session = TTTSession(model, params, inner, jit=False)
    actual = jnp.stack([session.decode(int(token)) for token in tokens])
    np.testing.assert_allclose(actual[:-1], expected.logits, rtol=1e-4, atol=1e-5)


def test_no_adaptation_matches_plain_forward(model_and_params):
    model, params = model_and_params
    tokens = np.arange(1, 10, dtype=np.int32)
    session = TTTSession(model, params, InnerLoopConfig(enabled=False), jit=False)
    expected = model.forward(params, jnp.asarray(tokens)).logits
    actual = session.observe(tokens)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    assert session.state.updates == 0


def test_session_isolation_and_reset(model_and_params):
    model, params = model_and_params
    first = TTTSession(model, params, InnerLoopConfig(learning_rate=0.1), jit=False)
    second = TTTSession(model, params, InnerLoopConfig(learning_rate=0.1), jit=False)
    tokens = np.asarray([1, 2, 3, 4, 5], dtype=np.int32)
    expected = first.observe(tokens)
    assert second.state.position == 0 and second.state.updates == 0
    np.testing.assert_allclose(second.observe(tokens), expected, atol=1e-6)
    first.reset()
    np.testing.assert_allclose(first.observe(tokens), expected, atol=1e-6)


def test_update_waits_for_observed_target(model_and_params):
    model, params = model_and_params
    session = TTTSession(model, params, jit=False)
    session.observe(np.asarray([1, 2], dtype=np.int32))
    assert session.state.updates == 0
    session.decode(3)
    assert session.state.updates == 1
