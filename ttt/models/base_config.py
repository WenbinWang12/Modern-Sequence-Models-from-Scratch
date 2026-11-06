"""可序列化的模型配置基类。"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

__all__ = ["PretrainedConfig"]


class PretrainedConfig:
    """提供统一的配置保存和加载接口。"""

    model_type: ClassVar[str] = "model"
    config_name: ClassVar[str] = "config.json"

    def to_dict(self) -> dict[str, Any]:
        """返回可写入 JSON 的配置字典。"""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]):
        """从配置字典创建实例。具体字段由子类定义。"""
        raise NotImplementedError

    def save_pretrained(self, save_directory: str | os.PathLike[str]) -> Path:
        """按预训练模型目录格式保存配置。"""
        directory = Path(save_directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / self.config_name
        handle_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory, prefix=f".{target.name}.", suffix=".tmp", delete=False
            ) as handle:
                handle_name = Path(handle.name)
                json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle_name, target)
        finally:
            if handle_name is not None and handle_name.exists():
                handle_name.unlink()
        return target

    @classmethod
    def from_pretrained(cls, path: str | os.PathLike[str]):
        """从目录或 JSON 配置文件加载配置。"""
        source = Path(path).expanduser().resolve()
        if source.is_dir():
            source = source / cls.config_name
        if source.suffix.lower() != ".json":
            raise ValueError(f"配置文件必须是 JSON：{source}")
        try:
            values = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"无法读取配置文件：{source}") from error
        if not isinstance(values, Mapping):
            raise ValueError("配置文件顶层必须是对象。")
        return cls.from_dict(values)
