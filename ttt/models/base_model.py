"""模型基类。"""

from __future__ import annotations

from typing import ClassVar

__all__ = ["PreTrainedModel"]


class PreTrainedModel:
    """保存模型配置并统一检查模型初始化参数。"""

    config_class: ClassVar[type] = object
    base_model_prefix: ClassVar[str] = "model"

    def __init__(self, config):
        if not isinstance(config, self.config_class):
            raise TypeError(f"config 必须是 {self.config_class.__name__}，实际为 {type(config).__name__}。")
        self.config = config
        self.params = None

    def load_parameters(self, params):
        """把已经校验过的参数绑定到模型实例。"""
        self.params = params
        return self
