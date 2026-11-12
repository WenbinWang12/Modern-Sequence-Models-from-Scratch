"""训练参数对象。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["TrainingArguments"]


@dataclass(frozen=True, kw_only=True)
class TrainingArguments:
    """描述一次训练作业的入口参数。

    该对象只管理作业级参数；模型结构、优化器和数据协议仍由实验配置文件负责。
    """

    config: str | Path
    data: str | Path
    output_dir: str | Path
    resume: str | Path | None = None
    auto_resume: bool = False
    initialize_from: str | Path | None = None

    def __post_init__(self) -> None:
        if self.auto_resume and (self.resume is not None or self.initialize_from is not None):
            raise ValueError("auto_resume 不能与 resume 或 initialize_from 同时使用。")
        if self.resume is not None and self.initialize_from is not None:
            raise ValueError("resume 不能与 initialize_from 同时使用。")

    @property
    def config_path(self) -> Path:
        """返回实验配置路径。"""

        return Path(self.config).expanduser()

    @property
    def data_path(self) -> Path:
        """返回训练数据路径。"""

        return Path(self.data).expanduser()

    @property
    def output_path(self) -> Path:
        """返回运行输出目录。"""

        return Path(self.output_dir).expanduser()
