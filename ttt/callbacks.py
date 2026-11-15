"""训练过程回调。

回调只负责观察训练生命周期，不持有模型参数，也不改变优化逻辑。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO

__all__ = ["TrainerCallback", "JsonlMetricsCallback"]


class TrainerCallback:
    """训练生命周期回调的兼容基类。"""

    def on_run_start(self, **_: Any) -> None:
        """运行开始时调用。"""

    def on_step_end(self, metrics: dict[str, Any]) -> None:
        """一次记录步结束时调用。"""

    def on_checkpoint(self, path: Path, step: int) -> None:
        """checkpoint 成功写入后调用。"""

    def on_run_end(self, status: str, step: int, checkpoint: Path | None) -> None:
        """运行正常结束时调用。"""

    def on_exception(self, error: BaseException, step: int) -> None:
        """运行发生异常时调用。"""

    def close(self) -> None:
        """释放回调持有的资源。"""


class JsonlMetricsCallback(TrainerCallback):
    """将结构化指标写入 JSONL，并可同步输出到标准输出。"""

    def __init__(self, path: str | Path, *, print_to_stdout: bool = True) -> None:
        self.path = Path(path)
        self.print_to_stdout = print_to_stdout
        self._handle: TextIO | None = None

    def on_run_start(self, **_: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def on_step_end(self, metrics: dict[str, Any]) -> None:
        message = json.dumps(metrics, ensure_ascii=False)
        if self.print_to_stdout:
            print(message, flush=True)
        if self._handle is not None:
            self._handle.write(message + "\n")
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
