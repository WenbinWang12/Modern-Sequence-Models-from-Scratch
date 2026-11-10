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

    def __post_init__(self):
        if not isinstance(self.enabled, bool) or not isinstance(self.track_post_loss, bool):
            raise ValueError("enabled 必须是布尔值。")
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
):
    def objective(current_fast):
        features, next_cache, auxiliary = model.suffix_hidden(
            params,
            hidden,
            current_fast,
            cache,
            position,
            token_mask=mask > 0,
            return_aux=True,
        )
        if return_logits:
            output = ModelOutput(model.project_logits(params, features), next_cache, auxiliary)
            losses = token_cross_entropy(output.logits, targets, mask)
        else:
            weight, bias = model.output_projection(params)
            losses = chunked_cross_entropy(
                features,
                weight.astype(model.config.compute_dtype),
                jnp.where(mask > 0, targets, 0),
                mask,
                bias=bias,
                token_block=model.config.loss_token_block_size,
                vocab_block=model.config.loss_vocab_block_size,
            )
            output = ModelOutput(None, next_cache, auxiliary)
        weight_sum = jnp.sum(mask)
        # 只有零权重才用 1 防止除零；小于 1 的正权重仍须正常归一化。
        loss = jnp.sum(losses) / jnp.where(weight_sum > 0, weight_sum, 1.0)
        return loss, (output, losses)

    if inner.enabled and fast_weights:
        (loss, (output, losses)), grads = jax.value_and_grad(objective, has_aux=True)(fast_weights)
        lr = inner.learning_rate if learning_rate is None else learning_rate
        updated, norm = sgd_update(fast_weights, grads, lr, inner.clip_norm)
    else:
        loss, (output, losses) = objective(fast_weights)
        updated, norm = fast_weights, jnp.array(0.0)
    post_loss = objective(updated)[0] if inner.track_post_loss else jnp.array(0.0)
    return loss, output, losses, updated, norm, post_loss


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
    chunks = (length + chunk_size - 1) // chunk_size
    padded_length = chunks * chunk_size
    inputs = jnp.pad(tokens[:-1], (0, padded_length - length))
    targets = jnp.pad(tokens[1:], (0, padded_length - length)).reshape(chunks, chunk_size)
    mask = jnp.pad(loss_mask.astype(jnp.float32), (0, padded_length - length)).reshape(chunks, chunk_size)
    # 前段参数不随块更新，可一次计算整个因果序列。
    hidden, _, prefix_aux = model.prefix(params, inputs, token_mask=jnp.arange(padded_length) < length, return_aux=True)
    hidden = hidden.reshape(chunks, chunk_size, model.config.hidden_size)
    fast = model.initial_fast_weights(params)
    cache = model.init_cache(range(model.config.prefix_length, model.config.num_hidden_layers), padded_length)

    def step(carry, batch):
        current_fast, current_cache = carry
        features, labels, weights, index = batch
        position = index * chunk_size
        _, output, losses, next_fast, norm, post_loss = inner_step(
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
        metrics = (losses, norm, output.aux, fast_norm, update_norm, post_loss)
        if return_logits:
            metrics = (*metrics, output.logits)
        return (next_fast, output.cache), metrics

    # 两层时间扫描减少反传保存的快权重快照，组内前向在反传时重算。
    group_size = max(index for index in range(1, math.isqrt(chunks) + 1) if chunks % index == 0)
    batches = (hidden, targets, mask, jnp.arange(chunks, dtype=jnp.int32))
    grouped = jax.tree.map(lambda value: value.reshape(chunks // group_size, group_size, *value.shape[1:]), batches)

    def group_step(carry, batch):
        return jax.lax.scan(jax.checkpoint(step), carry, batch)

    (fast, cache), grouped_outputs = jax.lax.scan(jax.checkpoint(group_step), (fast, cache), grouped)
    outputs = jax.tree.map(lambda value: value.reshape(chunks, *value.shape[2:]), grouped_outputs)
    token_losses = outputs[0].reshape(-1)[:length]
    logits = outputs[6].reshape(padded_length, -1)[:length] if return_logits else None
    weight_sum = jnp.sum(loss_mask)
    loss = jnp.sum(token_losses) / jnp.where(weight_sum > 0, weight_sum, 1.0)
    fractions = jnp.sum(mask > 0, axis=1) / jnp.maximum(jnp.sum(mask > 0), 1)
    suffix_aux = outputs[2]
    routing = prefix_aux["loads"] + jnp.sum(suffix_aux["loads"], axis=0)
    regularization = (
        model.config.router_aux_loss_coef * (prefix_aux["aux"] + jnp.sum(fractions * suffix_aux["aux"]))
        + model.config.router_z_loss_coef * (prefix_aux["z"] + jnp.sum(fractions * suffix_aux["z"]))
    ) / jnp.maximum(prefix_aux["blocks"] + jnp.sum(fractions * suffix_aux["blocks"]), 1)
    return SequenceOutput(
        loss, token_losses, logits, fast, cache, outputs[1], routing, regularization, outputs[3], outputs[4], outputs[5]
    )
