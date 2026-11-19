"""跨会话复用已编译的推理函数。"""

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .inner_loop import inner_step
from .models.ttt_e2e.modeling_ttt_e2e import TTTE2EForCausalLM


class RuntimeKernels(NamedTuple):
    prefix: object
    suffix: object
    suffix_hidden: object
    project_logits: object
    update: object


@lru_cache(maxsize=16)
def runtime_kernels(config, inner, jit, gradient_checkpointing):
    model = TTTE2EForCausalLM(config, gradient_checkpointing)

    def update(params, fast, features, targets, cache, position):
        # 当前块结束时，targets 包含主头的目标 x[t+1]，附加头可以复用其中已经
        # 观测到的前缀。对第 h 个附加头右移 h 位，块尾尚未观测的位置置零。
        heads = config.multi_token_heads
        offsets = jnp.arange(targets.shape[0])
        columns = jnp.stack(
            [
                jnp.where(
                    offsets + head < targets.shape[0],
                    targets[jnp.minimum(offsets + head, targets.shape[0] - 1)],
                    0,
                )
                for head in range(heads)
            ],
            axis=-1,
        )
        weights = jnp.stack(
            [((offsets + head) < targets.shape[0]).astype(jnp.float32) for head in range(heads)],
            axis=-1,
        )
        result = inner_step(
            model,
            params,
            fast,
            features,
            columns,
            weights,
            cache,
            position,
            inner,
            return_logits=False,
        )
        return result[3]

    functions = (model.prefix, model.suffix, model.suffix_hidden, model.project_logits, update)
    return RuntimeKernels(*(jax.jit(function) if jit else function for function in functions))
