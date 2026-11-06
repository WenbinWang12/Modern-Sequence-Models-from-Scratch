"""模型配置和模型类注册表。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

__all__ = [
    "register_config",
    "register_model",
    "get_config_class",
    "get_model_class",
]

ConfigClass = TypeVar("ConfigClass")
ModelClass = TypeVar("ModelClass")
_CONFIGS: dict[str, type] = {}
_MODELS: dict[str, type] = {}


def _names(model_type: str, aliases: Iterable[str] | None) -> tuple[str, ...]:
    if not isinstance(model_type, str) or not model_type.strip():
        raise ValueError("model_type 必须是非空字符串。")
    values = (model_type, *(aliases or ()))
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("模型注册别名必须是非空字符串。")
    return tuple(dict.fromkeys(values))


def register_config(config_class: ConfigClass, *, aliases: Iterable[str] | None = None) -> ConfigClass:
    """注册配置类，并拒绝覆盖已有模型类型。"""
    names = _names(config_class.model_type, aliases)
    for name in names:
        current = _CONFIGS.get(name)
        if current is not None and current is not config_class:
            raise ValueError(f"配置类型已注册：{name}")
    for name in names:
        _CONFIGS[name] = config_class
    return config_class


def register_model(model_class: ModelClass, *, model_type: str | None = None, aliases=None) -> ModelClass:
    """注册因果语言模型类。"""
    config_class = model_class.config_class
    names = _names(model_type or config_class.model_type, aliases)
    for name in names:
        current = _MODELS.get(name)
        if current is not None and current is not model_class:
            raise ValueError(f"模型类型已注册：{name}")
    for name in names:
        _MODELS[name] = model_class
    return model_class


def get_config_class(model_type: str) -> type:
    """按模型类型取得配置类。"""
    try:
        return _CONFIGS[model_type]
    except KeyError as error:
        raise ValueError(f"未注册的模型类型：{model_type}") from error


def get_model_class(model_type: str) -> type:
    """按模型类型取得模型类。"""
    try:
        return _MODELS[model_type]
    except KeyError as error:
        raise ValueError(f"未注册的因果语言模型类型：{model_type}") from error
