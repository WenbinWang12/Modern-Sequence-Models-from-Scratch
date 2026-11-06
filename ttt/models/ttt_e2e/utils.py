"""TTT-E2E 模型公共基类。"""

from ..base_model import PreTrainedModel
from .configuration_ttt_e2e import TTTE2EConfig

__all__ = ["TTTE2EPreTrainedModel"]


class TTTE2EPreTrainedModel(PreTrainedModel):
    """TTT-E2E 各运行模式共用的模型基类。"""

    config_class = TTTE2EConfig
    base_model_prefix = "ttt_e2e"
