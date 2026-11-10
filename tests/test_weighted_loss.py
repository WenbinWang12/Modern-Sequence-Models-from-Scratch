"""用小于 1 的权重和稀疏设备计数检查真实损失及梯度。"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ttt.inner_loop import InnerLoopConfig, sequence_loss
from ttt.training import TrainConfig, TrainState, make_train_step


def _reference_nll(model, params, tokens, weights):
    # 独立从未加权 logits 构造全局分子；只测试未做适应的预测目标。
    logits = sequence_loss(
        model, params, tokens, InnerLoopConfig(enabled=False), return_logits=True,
    ).logits
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    losses = -jnp.take_along_axis(log_probs, tokens[1:, None], axis=-1)[:, 0]
    return jnp.sum(losses * weights)


def _assert_tree_close(left, right, *, atol=2e-6):
    for actual, expected in zip(jax.tree.leaves(left), jax.tree.leaves(right)):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=atol)


def test_fractional_sequence_loss_and_gradient_use_actual_weight_sum(model_and_params):
    model, params = model_and_params
    tokens = jnp.asarray([1, 2, 3, 4])
    weights = jnp.asarray([0.25, 0.5, 0.0])
    inner = InnerLoopConfig(enabled=False)

    def actual(current):
        return sequence_loss(model, current, tokens, inner, weights).loss

    def reference(current):
        return _reference_nll(model, current, tokens, weights) / jnp.sum(weights)

    np.testing.assert_allclose(actual(params), reference(params), rtol=2e-5)
    _assert_tree_close(jax.grad(actual)(params), jax.grad(reference)(params))


def test_scaling_all_positive_weights_preserves_inner_updates(model_and_params):
    model, params = model_and_params
    tokens = jnp.asarray([1, 2, 3, 4, 5])
    inner = InnerLoopConfig(learning_rate=0.1)
    full = sequence_loss(model, params, tokens, inner, jnp.ones(4), return_logits=True)
    scaled = sequence_loss(model, params, tokens, inner, jnp.full(4, 0.125), return_logits=True)
    np.testing.assert_allclose(scaled.loss, full.loss, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(scaled.logits, full.logits, rtol=2e-5, atol=2e-6)
    _assert_tree_close(scaled.fast_weights, full.fast_weights)


def test_outer_fractional_weighting_matches_global_loss_and_gradient(model_and_params):
    model, params = model_and_params
    tokens = jnp.asarray([[1, 2, 3, 4], [5, 6, 7, 8]])
    weights = jnp.asarray([[0.1, 0.2, 0.0], [0.05, 0.0, 0.0]])
    learning_rate = 0.01
    optimizer = optax.sgd(learning_rate)
    state = TrainState(params, optimizer.init(params), jnp.asarray(0))
    config = TrainConfig(batch_size=2, sequence_length=3, inner_warmup_steps=0)
    step = make_train_step(model, InnerLoopConfig(enabled=False), config, optimizer, jit=False)

    def reference(current):
        numerator = sum(_reference_nll(model, current, row, mask) for row, mask in zip(tokens, weights))
        return numerator / jnp.sum(weights)

    updated, metrics = step(state, tokens, weights)
    expected_gradient = jax.grad(reference)(params)
    expected_params = jax.tree.map(lambda value, grad: value - learning_rate * grad, params, expected_gradient)
    np.testing.assert_allclose(metrics["nll"], reference(params), rtol=2e-5)
    np.testing.assert_allclose(metrics["loss"], reference(params), rtol=2e-5)
    np.testing.assert_allclose(metrics["grad_norm"], optax.global_norm(expected_gradient), rtol=2e-5)
    _assert_tree_close(updated.params, expected_params)


def test_all_zero_weights_have_zero_nll_and_zero_sgd_update(model_and_params):
    model, params = model_and_params
    tokens = jnp.asarray([1, 2, 3, 4])
    weights = jnp.zeros(3)
    inner = InnerLoopConfig(enabled=False)
    output = sequence_loss(model, params, tokens, inner, weights)
    assert float(output.loss) == 0.0
    np.testing.assert_array_equal(output.token_losses, np.zeros(3))
    gradient = jax.grad(lambda current: sequence_loss(model, current, tokens, inner, weights).loss)(params)
    _assert_tree_close(gradient, jax.tree.map(jnp.zeros_like, params), atol=0.0)
    # 无权重衰减、无动量的 SGD 才保证零梯度时参数不变。
    optimizer = optax.sgd(0.01)
    state = TrainState(params, optimizer.init(params), jnp.asarray(0))
    config = TrainConfig(sequence_length=3, inner_warmup_steps=0)
    step = make_train_step(model, inner, config, optimizer, jit=False)
    updated, metrics = step(state, tokens[None], weights[None])
    assert float(metrics["nll"]) == 0.0
    _assert_tree_close(updated.params, params, atol=0.0)


def test_pmap_sparse_binary_targets_preserve_global_mean():
    # 在远端 CPU 测试进程内启用两个逻辑 CPU 设备，不需要 GPU 或预先配置多卡。
    program = textwrap.dedent("""
        import jax
        import jax.numpy as jnp
        import numpy as np
        import optax
        from ttt.models.ttt_e2e import TTTE2EConfig, TTTE2EForCausalLM
        from ttt.inner_loop import InnerLoopConfig
        from ttt.training import TrainConfig, TrainState, make_train_step

        assert jax.local_device_count() == 2
        config = TTTE2EConfig(
            vocab_size=19, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, sliding_window=4,
            max_position_embeddings=32, ttt_chunk_size=2, compute_dtype="float32",
            post_norm=False, layer_types=("full_attention", "sliding_attention"),
        )
        model = TTTE2EForCausalLM(config)
        params = model.init_parameters(jax.random.PRNGKey(7))
        optimizer = optax.sgd(0.01)
        initial = TrainState(params, optimizer.init(params), jnp.asarray(0))
        inner = InnerLoopConfig(enabled=False)
        training = TrainConfig(sequence_length=2, inner_warmup_steps=0)
        single_step = make_train_step(model, inner, training, optimizer, jit=False)
        parallel_step = make_train_step(model, inner, training, optimizer, distributed=True)
        tokens = jnp.asarray([[1, 2, 3], [4, 5, 6]])
        masks = jnp.asarray([[1.0, 0.0], [0.0, 0.0]])
        expected, expected_metrics = single_step(initial, tokens, masks)
        replicated = jax.device_put_replicated(initial, jax.local_devices())
        actual, metrics = parallel_step(replicated, tokens[:, None, :], masks[:, None, :])
        np.testing.assert_allclose(metrics["nll"], expected_metrics["nll"], rtol=2e-5)
        np.testing.assert_allclose(metrics["loss"], expected_metrics["loss"], rtol=2e-5)
        for result, reference in zip(jax.tree.leaves(actual.params), jax.tree.leaves(expected.params)):
            for replica in result:
                np.testing.assert_allclose(replica, reference, rtol=2e-5, atol=2e-6)
    """)
    environment = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=2")
    environment.pop("JAX_NUM_CPU_DEVICES", None)
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=Path(__file__).resolve().parents[1],
        env=environment, capture_output=True, text=True, timeout=180, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
