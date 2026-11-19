"""多 token 预测头：单头退化、因果性与掩码。"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ttt.inner_loop import InnerLoopConfig, sequence_loss
from ttt.models.ttt_e2e import TTTE2EConfig, TTTE2EForCausalLM
from ttt.runtime import TTTSession

TOKENS = jnp.asarray([1, 5, 2, 7, 3, 9, 4, 11, 6, 13, 8, 15, 10, 17, 12, 19, 14, 21])


def build(heads=1, **changes):
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
        multi_token_heads=heads,
        compute_dtype="float32",
        post_norm=False,
        **changes,
    )
    model = TTTE2EForCausalLM(config)
    return model, model.init_parameters(jax.random.PRNGKey(11))


def test_single_head_matches_the_original_objective():
    """k=1 时参数结构和损失都必须与加入多头之前完全一致。"""
    model, params = build(heads=1)
    assert "multi_token" not in params
    inner = InnerLoopConfig(learning_rate=0.3, clip_norm=100.0)
    output = sequence_loss(model, params, TOKENS, inner)
    assert float(output.multi_token_loss) == 0.0
    # 逐 token 损失与手工计算的下一 token 交叉熵一致。
    logits = sequence_loss(model, params, TOKENS, inner, return_logits=True).logits
    expected = -jax.nn.log_softmax(logits.astype(jnp.float32))[jnp.arange(len(TOKENS) - 1), TOKENS[1:]]
    np.testing.assert_allclose(output.token_losses, expected, rtol=1e-5, atol=1e-6)


def test_extra_heads_do_not_change_the_reported_next_token_loss():
    """附加头只改变被优化的目标，不改变上报的 nll 与逐 token 损失。"""
    model, params = build(heads=3)
    assert len(params["multi_token"]) == 2
    inner = InnerLoopConfig(learning_rate=0.3, clip_norm=100.0)
    with_heads = sequence_loss(model, params, TOKENS, inner)
    # 把附加头的权重清零不会影响下一 token 损失，因为内循环默认只用主头。
    narrow = TTTE2EForCausalLM(replace(model.config, multi_token_heads=1))
    single = sequence_loss(narrow, {k: v for k, v in params.items() if k != "multi_token"}, TOKENS, inner)
    np.testing.assert_allclose(with_heads.token_losses, single.token_losses, rtol=1e-5, atol=1e-6)
    assert float(with_heads.multi_token_loss) > 0.0


@pytest.mark.parametrize("multi_token", [False, True])
def test_updates_never_use_an_unobserved_target(multi_token):
    """改变某个块之后的 token，不能影响该块及更早位置的预测。

    这是多 token 头最容易出错的地方：第 j 个头会预测 t+j，若不屏蔽块尾，
    第 c 块的更新就会用到第 c+1 块的答案。
    """
    model, params = build(heads=3)
    inner = InnerLoopConfig(learning_rate=0.5, clip_norm=100.0, multi_token=multi_token)
    chunk = model.config.ttt_chunk_size
    before = sequence_loss(model, params, TOKENS, inner, return_logits=True)
    # 第 2 块结束于输入位置 8；改动其后的全部 token。
    changed = sequence_loss(model, params, TOKENS.at[9:].set(jnp.asarray([20, 18, 16, 22, 1, 2, 3, 5, 4])), inner, return_logits=True)
    np.testing.assert_allclose(before.logits[:8], changed.logits[:8], atol=1e-6)
    assert float(jnp.max(jnp.abs(before.logits[9:] - changed.logits[9:]))) > 1e-5
    assert np.isfinite(np.asarray(before.multi_token_loss)).all()
    # 块尾被屏蔽的位置确实存在：三个头在长度 4 的块上分别有 4、3、2 个有效位置。
    assert chunk == 4


def test_inner_objective_flag_changes_the_fast_weights():
    """开启 multi_token 后内循环目标改变，快权重轨迹必须随之改变。"""
    model, params = build(heads=3)
    base = InnerLoopConfig(learning_rate=0.5, clip_norm=100.0, multi_token=False)
    with_aux = replace(base, multi_token=True)
    first = sequence_loss(model, params, TOKENS, base)
    second = sequence_loss(model, params, TOKENS, with_aux)
    difference = sum(
        float(jnp.sum(jnp.abs(a - b)))
        for a, b in zip(jax.tree.leaves(first.fast_weights), jax.tree.leaves(second.fast_weights), strict=True)
    )
    assert difference > 1e-6
    # 第一个块的预测发生在任何更新之前，两种设置必须完全相同。
    logits_a = sequence_loss(model, params, TOKENS, base, return_logits=True).logits
    logits_b = sequence_loss(model, params, TOKENS, with_aux, return_logits=True).logits
    np.testing.assert_allclose(logits_a[: model.config.ttt_chunk_size], logits_b[: model.config.ttt_chunk_size], atol=1e-6)


def test_outer_gradient_is_finite_with_extra_heads():
    model, params = build(heads=4)
    inner = InnerLoopConfig(learning_rate=0.3, clip_norm=1.0, multi_token=True)

    def objective(weights):
        output = sequence_loss(model, weights, TOKENS, inner)
        return output.loss + output.multi_token_loss

    grads = jax.grad(objective)(params)
    assert all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(grads))
    assert any(np.abs(np.asarray(value)).sum() > 0 for value in jax.tree.leaves(grads["multi_token"]))


def test_streaming_multi_token_updates_match_training():
    """在线更新只能使用已观测目标，但应与训练时的因果掩码完全一致。"""
    model, params = build(heads=3)
    inner = InnerLoopConfig(learning_rate=0.3, clip_norm=100.0, multi_token=True)
    expected = sequence_loss(model, params, TOKENS, inner, return_logits=True)
    session = TTTSession(model, params, inner, jit=False)
    actual = session.observe(np.asarray(TOKENS), return_logits=True)
    np.testing.assert_allclose(actual[:-1], expected.logits, rtol=1e-4, atol=1e-5)
    for observed, reference in zip(
        jax.tree.leaves(session.state.fast_weights), jax.tree.leaves(expected.fast_weights), strict=True
    ):
        np.testing.assert_allclose(observed, reference, rtol=1e-4, atol=1e-5)


def test_outer_multi_token_loss_keeps_cross_chunk_targets():
    """外循环应保留跨块目标，而当前块的因果预测不能看到未来输入。"""
    model, params = build(heads=2)
    inner = InnerLoopConfig(learning_rate=0.3, clip_norm=100.0)
    original = sequence_loss(model, params, TOKENS, inner, return_logits=True)
    changed_tokens = TOKENS.at[4].set(22)
    changed = sequence_loss(model, params, changed_tokens, inner, return_logits=True)
    np.testing.assert_allclose(original.logits[:4], changed.logits[:4], atol=1e-6)
    assert abs(float(original.multi_token_loss - changed.multi_token_loss)) > 1e-6
