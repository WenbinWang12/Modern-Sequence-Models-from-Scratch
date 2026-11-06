"""外循环优化与训练状态。"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from .inner_loop import sequence_loss


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 100
    batch_size: int = 1
    sequence_length: int = 32
    learning_rate: float = 0.0003
    weight_decay: float = 0.1
    clip_norm: float = 1.0
    warmup_steps: int = 10
    inner_warmup_steps: int = 10
    inner_lr_start: float = 0.1
    seed: int = 0
    checkpoint_every: int = 10
    log_every: int = 1
    gradient_checkpointing: bool = False
    data_parallel: bool = False
    checkpoint_format: str = "orbax"
    parallelism: str = "auto"
    expert_parallel_size: int = 1
    shard_parameters: bool = True

    def __post_init__(self):
        for name in ("steps", "batch_size", "sequence_length", "checkpoint_every", "log_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数。")
        for name in ("warmup_steps", "inner_warmup_steps", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} 必须是非负整数。")
        for name in ("learning_rate", "clip_norm", "inner_lr_start"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是有限的正数。")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay 必须是有限的非负数。")
        if not isinstance(self.gradient_checkpointing, bool) or not isinstance(self.data_parallel, bool):
            raise ValueError("gradient_checkpointing 和 data_parallel 必须是布尔值。")
        if self.checkpoint_format not in ("npz", "orbax"):
            raise ValueError("checkpoint_format 必须为 npz 或 orbax。")
        if self.parallelism not in ("auto", "pmap", "mesh"):
            raise ValueError("parallelism 必须为 auto、pmap 或 mesh。")
        if (
            isinstance(self.expert_parallel_size, bool)
            or not isinstance(self.expert_parallel_size, int)
            or self.expert_parallel_size < 1
        ):
            raise ValueError("expert_parallel_size 必须为正整数。")
        if not isinstance(self.shard_parameters, bool):
            raise ValueError("shard_parameters 必须为布尔值。")


class TrainState(NamedTuple):
    params: dict
    opt_state: tuple
    step: jax.Array


def make_optimizer(config):
    def schedule(step):
        warmup = jnp.minimum((step + 1) / max(config.warmup_steps, 1), 1.0)
        progress = jnp.clip((step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1), 0, 1)
        return config.learning_rate * warmup * 0.5 * (1 + jnp.cos(jnp.pi * progress))

    return optax.chain(
        optax.clip_by_global_norm(config.clip_norm),
        optax.adamw(
            schedule,
            b1=0.9,
            b2=0.95,
            weight_decay=config.weight_decay,
            mask=lambda params: jax.tree.map(lambda value: value.ndim >= 2, params),
        ),
    )


def make_train_step(model, inner, training, optimizer, *, jit=True, distributed=False):
    def step(state, tokens, loss_masks):
        fraction = jnp.minimum(state.step / max(training.inner_warmup_steps, 1), 1.0)
        inner_lr = jnp.where(
            training.inner_warmup_steps > 0,
            training.inner_lr_start + (inner.learning_rate - training.inner_lr_start) * fraction,
            inner.learning_rate,
        )

        def objective(params):
            def per_sequence(weights, ids, mask):
                output = sequence_loss(model, weights, ids, inner, mask, learning_rate=inner_lr)
                statistics = {
                    "inner_grad_norm": jnp.mean(output.grad_norms),
                    "fast_weight_norm": jnp.mean(output.fast_weight_norms),
                    "inner_update_norm": jnp.mean(output.update_norms),
                    "router_loads": output.router_loads,
                    "regularization_loss": output.regularization_loss,
                    "nll": output.loss,
                }
                if inner.track_post_loss:
                    statistics["inner_post_loss"] = jnp.mean(output.post_losses)
                return output.loss + output.regularization_loss, statistics

            losses, statistics = jax.vmap(per_sequence, in_axes=(None, 0, 0))(params, tokens, loss_masks)
            counts = jnp.sum(loss_masks, axis=1)
            denominator = jnp.sum(counts)
            if distributed:
                denominator = jax.lax.pmean(denominator, "data")
            average = jax.tree.map(lambda value: jnp.mean(value, axis=0), statistics)
            average["nll"] = jnp.sum(statistics["nll"] * counts) / jnp.maximum(denominator, 1)
            return jnp.sum(losses * counts) / jnp.maximum(denominator, 1), average

        (loss, statistics), grads = jax.value_and_grad(objective, has_aux=True)(state.params)
        if distributed:
            grads = jax.lax.pmean(grads, "data")
            loss = jax.lax.pmean(loss, "data")
            statistics = jax.lax.pmean(statistics, "data")
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        if model.config.num_experts and model.config.router_bias_update_rate:
            layers = list(params["layers"])
            for index, layer in enumerate(layers):
                if "moe" in layer:
                    loads = statistics["router_loads"][index]
                    bias = layer["moe"]["load_bias"]
                    bias = bias + model.config.router_bias_update_rate * jnp.sign(jnp.mean(loads) - loads)
                    layers[index] = {**layer, "moe": {**layer["moe"], "load_bias": bias - jnp.mean(bias)}}
            params = {**params, "layers": tuple(layers)}
        metrics = {
            "loss": loss,
            "grad_norm": optax.global_norm(grads),
            "inner_lr": inner_lr,
            **{key: value for key, value in statistics.items() if key != "router_loads"},
        }
        return TrainState(params, opt_state, state.step + 1), metrics

    if distributed:
        return jax.pmap(step, axis_name="data")
    return jax.jit(step) if jit else step
