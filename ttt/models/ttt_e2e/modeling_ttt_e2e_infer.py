"""TTT-E2E 推理模式模型入口。"""

from .modeling_ttt_e2e import TTTE2EForCausalLM

__all__ = ["InferenceTTTE2EForCausalLM"]


class InferenceTTTE2EForCausalLM(TTTE2EForCausalLM):
    """关闭梯度检查点的推理模型。"""

    def __init__(self, config, gradient_checkpointing=False):
        super().__init__(config, gradient_checkpointing=gradient_checkpointing)
