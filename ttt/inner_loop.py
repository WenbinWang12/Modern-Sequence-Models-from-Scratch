"""块级 SGD 更新与完整元学习序列。"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .loss import chunked_cross_entropy
from .models.ttt_e2e.modeling_ttt_e2e import ModelOutput


@dataclass(frozen=True)
class InnerLoopConfig:
    enabled: bool = True
    learning_rate: float = 1.0
    clip_norm: float = 1.0
    track_post_loss: bool = False
    first_order: bool = False  # 外循环不对内循环梯度求导（一阶近似），仅用于消融。
    multi_token: bool = False  # 内循环目标是否也包含附加预测头（默认只用下一 token）。

    def __post_init__(self):
        flags = (self.enabled, self.track_post_loss, self.first_order, self.multi_token)
        if not all(isinstance(flag, bool) for flag in flags):
            raise ValueError("enabled、track_post_loss、first_order 和 multi_token 必须是布尔值。")
        for name in ("learning_rate", "clip_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是有限的正数。")


class SequenceOutput(NamedTuple):
    loss: jax.Array
    token_losses: jax.Array
    logits: jax.Array | None
    fast_weights: tuple
    cache: tuple
    grad_norms: jax.Array
    router_loads: jax.Array
    regularization_loss: jax.Array
    fast_weight_norms: jax.Array
    update_norms: jax.Array
    post_losses: jax.Array
    multi_token_loss: jax.Array


def token_cross_entropy(logits, targets, mask=None):
    if mask is None:
        mask = jnp.ones(targets.shape, dtype=jnp.float32)
    safe_targets = jnp.where(mask > 0, targets, 0)
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    losses = -jnp.take_along_axis(log_probs, safe_targets[..., None], axis=-1)[..., 0]
    return jnp.where(mask > 0, losses * mask, 0.0)


def sgd_update(fast_weights, grads, learning_rate, clip_norm):
    # 加小量避免零梯度处的二阶导出现 NaN。
    squared = sum(jnp.sum(g.astype(jnp.float32) ** 2) for g in jax.tree.leaves(grads))
    norm = jnp.sqrt(squared + 1e-12)
    scale = jnp.minimum(1.0, clip_norm / norm)
    updated = jax.tree.map(lambda weight, grad: weight - learning_rate * scale * grad, fast_weights, grads)
    return updated, norm


def inner_step(
    model,
    params,
    fast_weights,
    hidden,
    targets,
    mask,
    cache,
    position,
    inner,
    learning_rate=None,
    *,
    return_logits=True,
    outer_mask=None,
):
    config = model.config
    heads = config.multi_token_heads
    # 保留单头调用方的兼容性。旧的推理入口传入一维目标和掩码，
    # 多头实现内部统一使用 [时间步, 预测头] 两维表示。
    if targets.ndim == 1:
        targets = targets[:, None]
    if mask.ndim == 1:
        mask = mask[:, None]
    if targets.ndim != 2 or mask.ndim != 2:
        raise ValueError("targets 和 mask 必须是一维或二维数组。")
    if targets.shape != mask.shape or targets.shape[1] != heads:
        raise ValueError("targets、mask 的形状必须为 [时间步, multi_token_heads]。")
    if outer_mask is None:
        outer_mask = mask
    elif outer_mask.ndim == 1:
        outer_mask = outer_mask[:, None]
    if outer_mask.ndim != 2 or outer_mask.shape != targets.shape:
        raise ValueError("outer_mask 的形状必须为 [时间步, multi_token_heads]。")

    def normalise(values, weights):
        total = jnp.sum(weights)
        # 只有零权重才用 1 防止除零；小于 1 的正权重仍须正常归一化。
        return jnp.sum(values) / jnp.where(total > 0, total, 1.0)

    def objective(current_fast):
        features, next_cache, auxiliary = model.suffix_hidden(
            params,
            hidden,
            current_fast,
            cache,
            position,
            token_mask=mask[:, 0] > 0,
            return_aux=True,
        )
        weight, bias = model.output_projection(params)

        def head_losses(head, active_mask):
            states = model.head_hidden(params, features, head)
            column = active_mask[:, head]
            labels = jnp.where(column > 0, targets[:, head], 0)
            if return_logits:
                return token_cross_entropy(model.project_logits(params, states), labels, column), states
            return chunked_cross_entropy(
                states,
                weight.astype(config.compute_dtype),
                labels,
                column,
                bias=bias,
                token_block=config.loss_token_block_size,
                vocab_block=config.loss_vocab_block_size,
            ), states

        losses, main_states = head_losses(0, mask)
        logits = model.project_logits(params, main_states) if return_logits else None
        output = ModelOutput(logits, next_cache, auxiliary)
        loss = normalise(losses, mask[:, 0])
        extra = jnp.array(0.0)
        if heads > 1:
            # 内循环只使用已经观测到的目标。各附加头先独立归一化，再取平均。
            extra = sum(normalise(head_losses(head, mask)[0], mask[:, head]) for head in range(1, heads))
            extra = config.multi_token_weight * extra / (heads - 1)
        value = loss + extra if inner.multi_token else loss
        # 外循环可以使用序列中已经给出的跨块目标。这里返回加权损失和有效权重，
        # 由 sequence_loss 在整条序列上统一归一化，避免尾块被过度加权。
        outer_sum = jnp.zeros((heads - 1,), dtype=jnp.float32)
        outer_weight = jnp.zeros((heads - 1,), dtype=jnp.float32)
        for head in range(1, heads):
            outer_sum = outer_sum.at[head - 1].set(jnp.sum(head_losses(head, outer_mask)[0]))
            outer_weight = outer_weight.at[head - 1].set(jnp.sum(outer_mask[:, head]))
        return value, (output, losses, outer_sum, outer_weight)

    if inner.enabled and fast_weights:
        (_, (output, losses, outer_sum, outer_weight)), grads = jax.value_and_grad(objective, has_aux=True)(fast_weights)
        if inner.first_order:
            # 把内循环梯度当作常数：前向数值不变，外循环少了穿过更新量的二阶项。
            grads = jax.lax.stop_gradient(grads)
        lr = inner.learning_rate if learning_rate is None else learning_rate
        updated, norm = sgd_update(fast_weights, grads, lr, inner.clip_norm)
    else:
        _, (output, losses, outer_sum, outer_weight) = objective(fast_weights)
        updated, norm = fast_weights, jnp.array(0.0)
    post_loss = objective(updated)[0] if inner.track_post_loss else jnp.array(0.0)
    # 返回的 loss 始终是下一 token 损失，使所有指标和评测与单头设置可比。
    loss = jnp.sum(losses) / jnp.where(jnp.sum(mask[:, 0]) > 0, jnp.sum(mask[:, 0]), 1.0)
    return loss, output, losses, updated, norm, post_loss, outer_sum, outer_weight


def sequence_loss(model, params, tokens, inner, loss_mask=None, *, return_logits=False, learning_rate=None):
    if tokens.ndim != 1 or tokens.shape[0] < 2:
        raise ValueError("tokens 必须是一维序列，至少包含两个 token。")
    length = tokens.shape[0] - 1
    if length > model.config.max_position_embeddings:
        raise ValueError("序列超过 max_position_embeddings。")
    if loss_mask is None:
        loss_mask = jnp.ones((length,), dtype=jnp.float32)
    if loss_mask.shape != (length,):
        raise ValueError("loss_mask 必须与下一 token 训练对数量一致。")
    chunk_size = model.config.ttt_chunk_size
    heads = model.config.multi_token_heads
    chunks = (length + chunk_size - 1) // chunk_size
    padded_length = chunks * chunk_size
    inputs = jnp.pad(tokens[:-1], (0, padded_length - length))
    # 第 j 个头在输入位置 t 预测 tokens[t+1+j]。外循环保留所有位于序列内的目标，
    # 内循环另外屏蔽尚未在当前块中观测到的目标，防止答案提前进入快权重更新。
    extended = jnp.pad(tokens, (0, padded_length + heads - tokens.shape[0]))
    weights = jnp.pad(loss_mask.astype(jnp.float32), (0, padded_length + heads - length))
    offsets = jnp.arange(padded_length)
    within = offsets % chunk_size
    columns, inner_masks, outer_masks = [], [], []
    for head in range(heads):
        columns.append(extended[offsets + 1 + head])
        valid = weights[offsets + head] * ((offsets + 1 + head) <= length)
        observed = (within + head) < chunk_size
        outer_masks.append(valid)
        inner_masks.append(valid * observed)
    targets = jnp.stack(columns, axis=-1).reshape(chunks, chunk_size, heads)
    inner_mask = jnp.stack(inner_masks, axis=-1).astype(jnp.float32).reshape(chunks, chunk_size, heads)
    outer_mask = jnp.stack(outer_masks, axis=-1).astype(jnp.float32).reshape(chunks, chunk_size, heads)
    # 前段参数不随块更新，可一次计算整个因果序列。
    hidden, _, prefix_aux = model.prefix(params, inputs, token_mask=jnp.arange(padded_length) < length, return_aux=True)
    hidden = hidden.reshape(chunks, chunk_size, model.config.hidden_size)
    fast = model.initial_fast_weights(params)
    cache = model.init_cache(range(model.config.prefix_length, model.config.num_hidden_layers), padded_length)

    def step(carry, batch):
        current_fast, current_cache = carry
        features, labels, weights, outer_weights, index = batch
        position = index * chunk_size
        _, output, losses, next_fast, norm, post_loss, extra_sum, extra_weight = inner_step(
            model,
            params,
            current_fast,
            features,
            labels,
            weights,
            current_cache,
            position,
            inner,
            learning_rate,
            return_logits=return_logits,
            outer_mask=outer_weights,
        )
        # 尾部未满一个块时只评分，等待更多观测后再更新。
        next_fast = jax.tree.map(
            lambda new, old: jnp.where(position + chunk_size <= length, new, old),
            next_fast,
            current_fast,
        )
        fast_norm = jnp.sqrt(sum(jnp.sum(value.astype(jnp.float32) ** 2) for value in jax.tree.leaves(next_fast)))
        delta = jax.tree.map(jnp.subtract, next_fast, current_fast)
        update_norm = jnp.sqrt(sum(jnp.sum(value.astype(jnp.float32) ** 2) for value in jax.tree.leaves(delta)))
        metrics = (losses, norm, output.aux, fast_norm, update_norm, post_loss, extra_sum, extra_weight)
        if return_logits:
            metrics = (*metrics, output.logits)
        return (next_fast, output.cache), metrics

    # 两层时间扫描减少反传保存的快权重快照，组内前向在反传时重算。
    group_size = max(index for index in range(1, math.isqrt(chunks) + 1) if chunks % index == 0)
    batches = (hidden, targets, inner_mask, outer_mask, jnp.arange(chunks, dtype=jnp.int32))
    grouped = jax.tree.map(lambda value: value.reshape(chunks // group_size, group_size, *value.shape[1:]), batches)

    def group_step(carry, batch):
        return jax.lax.scan(jax.checkpoint(step), carry, batch)

    (fast, cache), grouped_outputs = jax.lax.scan(jax.checkpoint(group_step), (fast, cache), grouped)
    outputs = jax.tree.map(lambda value: value.reshape(chunks, *value.shape[2:]), grouped_outputs)
    token_losses = outputs[0].reshape(-1)[:length]
    logits = outputs[8].reshape(padded_length, -1)[:length] if return_logits else None
    weight_sum = jnp.sum(loss_mask)
    loss = jnp.sum(token_losses) / jnp.where(weight_sum > 0, weight_sum, 1.0)
    main_weight = jnp.sum(inner_mask[:, :, 0], axis=1)
    fractions = main_weight / jnp.where(jnp.sum(main_weight) > 0, jnp.sum(main_weight), 1.0)
    if heads > 1:
        head_sums = jnp.sum(outputs[6], axis=0)
        head_weights = jnp.sum(outputs[7], axis=0)
        active_heads = head_weights > 0
        head_losses = head_sums / jnp.where(active_heads, head_weights, 1.0)
        multi_token_loss = model.config.multi_token_weight * jnp.sum(head_losses * active_heads) / jnp.where(
            jnp.sum(active_heads) > 0, jnp.sum(active_heads), 1.0
        )
    else:
        multi_token_loss = jnp.array(0.0, dtype=jnp.float32)
    suffix_aux = outputs[2]
    routing = prefix_aux["loads"] + jnp.sum(suffix_aux["loads"], axis=0)
    regularization = (
        model.config.router_aux_loss_coef * (prefix_aux["aux"] + jnp.sum(fractions * suffix_aux["aux"]))
        + model.config.router_z_loss_coef * (prefix_aux["z"] + jnp.sum(fractions * suffix_aux["z"]))
    ) / jnp.maximum(prefix_aux["blocks"] + jnp.sum(fractions * suffix_aux["blocks"]), 1)
    return SequenceOutput(
        loss,
        token_losses,
        logits,
        fast,
        cache,
        outputs[1],
        routing,
        regularization,
        outputs[3],
        outputs[4],
        outputs[5],
        multi_token_loss,
    )
