"""模型变体和批内独立状态。"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ttt.checkpoint import load_checkpoint, save_checkpoint
from ttt.config import ExperimentConfig
from ttt.inner_loop import InnerLoopConfig, sequence_loss
from ttt.models.ttt_e2e.modeling_ttt_e2e import TTTE2EForCausalLM
from ttt.training import TrainConfig, TrainState, make_optimizer, make_train_step


@pytest.mark.parametrize("use_fast_mlp,tie_embeddings", [(False, True), (False, False), (True, False)])
def test_backbone_variants(model_and_params, use_fast_mlp, tie_embeddings):
    original, _ = model_and_params
    config = replace(
        original.config,
        use_fast_mlp=use_fast_mlp,
        ttt_suffix_length=1 if use_fast_mlp else 0,
        tie_word_embeddings=tie_embeddings,
    )
    model = TTTE2EForCausalLM(config)
    params = model.init_parameters(jax.random.PRNGKey(3))
    tokens = jnp.asarray([1, 2, 3, 4, 5])
    actual = sequence_loss(model, params, tokens, InnerLoopConfig(enabled=False), return_logits=True)
    expected = model.forward(params, tokens[:-1])
    np.testing.assert_allclose(actual.logits, expected.logits, rtol=1e-4, atol=1e-5)


def test_batch_sequences_have_independent_fast_weights(model_and_params):
    model, params = model_and_params
    inner = InnerLoopConfig(learning_rate=0.1)
    tokens = jnp.asarray([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    batched = jax.vmap(lambda row: sequence_loss(model, params, row, inner).loss)(tokens)
    individual = jnp.stack([sequence_loss(model, params, row, inner).loss for row in tokens])
    np.testing.assert_allclose(batched, individual, atol=1e-6)


def test_rematerialization_preserves_outer_gradient(model_and_params):
    original, params = model_and_params
    remat = TTTE2EForCausalLM(original.config, gradient_checkpointing=True)
    tokens = jnp.asarray([1, 2, 3, 4, 5])
    inner = InnerLoopConfig(learning_rate=0.1)
    before = jax.grad(lambda weights: sequence_loss(original, weights, tokens, inner).loss)(params)
    after = jax.grad(lambda weights: sequence_loss(remat, weights, tokens, inner).loss)(params)
    for left, right in zip(jax.tree.leaves(before), jax.tree.leaves(after)):
        np.testing.assert_allclose(left, right, rtol=1e-4, atol=1e-5)


def test_bfloat16_checkpoint_roundtrip(model_and_params, tmp_path):
    original, _ = model_and_params
    config = replace(original.config, param_dtype="bfloat16", compute_dtype="bfloat16")
    model = TTTE2EForCausalLM(config)
    params = model.init_parameters(jax.random.PRNGKey(4))
    experiment = ExperimentConfig(config, InnerLoopConfig(), TrainConfig(sequence_length=4), tokenizer="custom")
    path = tmp_path / "bf16.npz"
    save_checkpoint(path, TrainState(params, None, jnp.asarray(0)), experiment)
    restored = load_checkpoint(path)
    for left, right in zip(jax.tree.leaves(params), jax.tree.leaves(restored.state.params)):
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)


def test_data_parallel_step_matches_single_device(model_and_params):
    model, params = model_and_params
    config = TrainConfig(steps=2, sequence_length=4, warmup_steps=0, inner_warmup_steps=0)
    inner = InnerLoopConfig(learning_rate=0.1)
    optimizer = make_optimizer(config)
    state = TrainState(params, optimizer.init(params), jnp.asarray(0))
    tokens = jnp.asarray([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    devices = jax.local_devices()
    tokens = tokens[jnp.arange(max(2, len(devices))) % 2]
    masks = jnp.ones((tokens.shape[0], 4))
    single, _ = make_train_step(model, inner, config, optimizer)(state, tokens, masks)
    replicated = jax.device_put_replicated(state, devices)
    sharded_tokens = tokens.reshape(len(devices), -1, 5)
    sharded_masks = masks.reshape(len(devices), -1, 4)
    parallel, _ = make_train_step(model, inner, config, optimizer, distributed=True)(
        replicated,
        sharded_tokens,
        sharded_masks,
    )
    for left, right in zip(jax.tree.leaves(single.params), jax.tree.leaves(parallel.params)):
        np.testing.assert_allclose(left, right[0], rtol=1e-4, atol=1e-6)
