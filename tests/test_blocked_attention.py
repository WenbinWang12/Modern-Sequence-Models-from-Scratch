"""按窗口分块的滑窗注意力：与逐位置掩码的参考实现等价。"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ttt.inner_loop import InnerLoopConfig, sequence_loss
from ttt.models.ttt_e2e import TTTE2EConfig, TTTE2EForCausalLM
from ttt.models.ttt_e2e.kernels import blocked_sliding_attention, reference_attention


@pytest.mark.parametrize(("length", "window"), [(12, 4), (13, 4), (5, 4), (9, 8), (17, 3)])
def test_blocked_attention_matches_reference(length, window):
    keys = jax.random.split(jax.random.PRNGKey(length * 31 + window), 3)
    query, key, value = (jax.random.normal(item, (length, 2, 4)) for item in keys)
    positions = jnp.arange(length) + 5

    def blocked(q, k):
        return blocked_sliding_attention(q, k, value, positions, window)

    def reference(q, k):
        return reference_attention(q, k, value, positions, positions, window)

    np.testing.assert_allclose(blocked(query, key), reference(query, key), rtol=1e-5, atol=1e-6)
    first = jax.grad(lambda q, k: jnp.sum(blocked(q, k) ** 2), argnums=(0, 1))
    second = jax.grad(lambda q, k: jnp.sum(reference(q, k) ** 2), argnums=(0, 1))
    for left, right in zip(first(query, key), second(query, key), strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-4, atol=3e-6)
    # 外循环需要二阶导：沿同一方向的 Hessian-向量积也必须一致。
    direction = jnp.ones_like(query)
    left = jax.jvp(lambda q: first(q, key)[0], (query,), (direction,))[1]
    right = jax.jvp(lambda q: second(q, key)[0], (query,), (direction,))[1]
    np.testing.assert_allclose(left, right, rtol=1e-4, atol=5e-6)


def small_model(backend):
    config = TTTE2EConfig(
        vocab_size=23,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        sliding_window=4,
        max_position_embeddings=64,
        ttt_suffix_length=1,
        ttt_chunk_size=4,
        attention_backend=backend,
        compute_dtype="float32",
        post_norm=False,
    )
    model = TTTE2EForCausalLM(config)
    return model, model.init_parameters(jax.random.PRNGKey(11))


def test_blocked_backend_matches_reference_through_the_meta_objective():
    tokens = jnp.asarray([1, 5, 2, 7, 3, 9, 4, 11, 6, 13, 8, 15, 10, 17, 12, 19, 14, 21])
    inner = InnerLoopConfig(learning_rate=0.3, clip_norm=100.0)
    reference_model, params = small_model("reference")
    blocked_model = TTTE2EForCausalLM(replace(reference_model.config, attention_backend="blocked"))

    def loss(model, weights):
        return sequence_loss(model, weights, tokens, inner).loss

    np.testing.assert_allclose(loss(blocked_model, params), loss(reference_model, params), rtol=1e-5)
    blocked_grads = jax.grad(lambda weights: loss(blocked_model, weights))(params)
    reference_grads = jax.grad(lambda weights: loss(reference_model, weights))(params)
    for left, right in zip(jax.tree.leaves(blocked_grads), jax.tree.leaves(reference_grads), strict=True):
        np.testing.assert_allclose(left, right, rtol=2e-3, atol=2e-6)


def test_blocked_backend_has_no_future_leak():
    model, params = small_model("blocked")
    tokens = jnp.asarray([1, 5, 2, 7, 3, 9, 4, 11, 6, 13, 8, 15, 10, 17, 12, 19, 14, 21])
    changed = tokens.at[11:].set(jnp.asarray([20, 18, 16, 22, 1, 2, 3]))
    inner = InnerLoopConfig(learning_rate=0.3)
    before = sequence_loss(model, params, tokens, inner, return_logits=True)
    after = sequence_loss(model, params, changed, inner, return_logits=True)
    # 位置 0..9 的预测只依赖 token 0..10。
    np.testing.assert_allclose(before.logits[:10], after.logits[:10], atol=1e-6)
    assert float(jnp.max(jnp.abs(before.logits[11:] - after.logits[11:]))) > 1e-4


def test_blocked_backend_ignores_context_outside_the_window_without_updates():
    model, params = small_model("blocked")
    tokens = jnp.asarray([1, 5, 2, 7, 3, 9, 4, 11, 6, 13, 8, 15, 10, 17, 12, 19, 14, 21])
    changed = tokens.at[:3].set(jnp.asarray([22, 20, 18]))
    off = InnerLoopConfig(enabled=False)
    before = sequence_loss(model, params, tokens, off, return_logits=True)
    after = sequence_loss(model, params, changed, off, return_logits=True)
    # 三层、窗口 4：位置 t 的感受野最远到 t-9，所以位置 12 之后看不到 token 0..2。
    np.testing.assert_allclose(before.logits[12:], after.logits[12:], atol=1e-6)
    # 打开测试时更新后，窗口之外的内容通过快权重影响后面的预测。
    on = InnerLoopConfig(learning_rate=0.3)
    before = sequence_loss(model, params, tokens, on, return_logits=True)
    after = sequence_loss(model, params, changed, on, return_logits=True)
    assert float(jnp.max(jnp.abs(before.logits[12:] - after.logits[12:]))) > 1e-6
