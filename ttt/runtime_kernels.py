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
        result = inner_step(
            model,
            params,
            fast,
            features,
            targets,
            jnp.ones(targets.shape, dtype=jnp.float32),
            cache,
            position,
            inner,
            return_logits=False,
        )
        return result[3]

    functions = (model.prefix, model.suffix, model.suffix_hidden, model.project_logits, update)
    return RuntimeKernels(*(jax.jit(function) if jit else function for function in functions))
