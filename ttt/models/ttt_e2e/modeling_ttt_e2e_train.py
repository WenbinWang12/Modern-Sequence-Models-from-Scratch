"""TTT-E2E 训练模式模型入口。"""

from .modeling_ttt_e2e import TTTE2EForCausalLM

__all__ = ["TrainingTTTE2EForCausalLM"]


class TrainingTTTE2EForCausalLM(TTTE2EForCausalLM):
    """启用梯度检查点的训练模型。"""

    def __init__(self, config, gradient_checkpointing=True):
        super().__init__(config, gradient_checkpointing=gradient_checkpointing)
