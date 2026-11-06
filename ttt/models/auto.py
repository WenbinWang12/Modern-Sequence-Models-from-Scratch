"""自动构建模型配置和因果语言模型。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from .registry import get_config_class, get_model_class

__all__ = ["AutoConfig", "AutoModelForCausalLM"]


def _load_builtin_models() -> None:
    # 延迟导入，避免导入自动接口时加载所有模型实现。
    from . import ttt_e2e  # noqa: F401


class AutoConfig:
    """根据 model_type 或预训练目录构建配置。"""

    @classmethod
    def from_dict(cls, values):
        _load_builtin_models()
        if not isinstance(values, Mapping):
            raise TypeError("配置必须是字典。")
        values = dict(values)
        model_type = values.get("model_type")
        if not isinstance(model_type, str):
            raise ValueError("配置必须包含字符串字段 model_type。")
        return get_config_class(model_type).from_dict(values)

    @classmethod
    def from_pretrained(cls, path: str | Path):
        _load_builtin_models()
        source = Path(path).expanduser().resolve()
        if source.is_dir():
            source = source / "config.json"
        try:
            values = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"无法读取配置文件：{source}") from error
        if isinstance(values, dict) and "model_type" not in values and isinstance(values.get("model"), dict):
            values = values["model"]
        if isinstance(values, dict) and "model_type" not in values:
            # 早期实验配置没有写入 model_type；该目录目前只存放 TTT-E2E 配置。
            values = {"model_type": "ttt_e2e", **values}
        return cls.from_dict(values)


class AutoModelForCausalLM:
    """根据配置构建已注册的因果语言模型。"""

    @classmethod
    def from_config(cls, config, **kwargs):
        _load_builtin_models()
        mode = kwargs.pop("mode", "base")
        if mode not in ("base", "train", "infer"):
            raise ValueError("mode 必须为 base、train 或 infer。")
        if mode == "train":
            from .ttt_e2e.modeling_ttt_e2e_train import TrainingTTTE2EForCausalLM

            model_class = TrainingTTTE2EForCausalLM
        elif mode == "infer":
            from .ttt_e2e.modeling_ttt_e2e_infer import InferenceTTTE2EForCausalLM

            model_class = InferenceTTTE2EForCausalLM
        else:
            model_class = get_model_class(config.model_type)
        return model_class(config, **kwargs)

    @classmethod
    def from_pretrained(cls, path: str | Path, **kwargs):
        """从 NPZ 或 Orbax 检查点读取配置和参数。"""
        source = Path(path).expanduser().resolve()
        if source.is_dir() and (source / "latest.json").is_file():
            import json

            latest = json.loads((source / "latest.json").read_text(encoding="utf-8"))
            checkpoint_name = latest.get("path")
            if not isinstance(checkpoint_name, str):
                raise ValueError("latest.json 缺少字符串字段 path。")
            source = (source / checkpoint_name).resolve()
        if not source.exists():
            raise FileNotFoundError(f"预训练检查点不存在：{source}")
        from ..checkpoint import load_checkpoint

        checkpoint = load_checkpoint(source, load_optimizer=False)
        model = cls.from_config(checkpoint.experiment.model, **kwargs)
        model.load_parameters(checkpoint.state.params)
        model.checkpoint = checkpoint
        return model
