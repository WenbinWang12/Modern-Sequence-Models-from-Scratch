"""训练器门面。

该模块提供接近业界 Trainer 的作业级接口，同时保留现有 ttt.train 的兼容入口。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig, load_experiment
from ..lifecycle import latest_checkpoint
from ..train import train as run_training
from .training_args import TrainingArguments

__all__ = ["Trainer", "TrainingResult"]


@dataclass(frozen=True)
class TrainingResult:
    """一次训练作业的最小结果。"""

    checkpoint: Path
    output_dir: Path


class Trainer:
    """管理实验配置、恢复策略和训练生命周期。"""

    def __init__(
        self,
        args: TrainingArguments,
        *,
        experiment: ExperimentConfig | None = None,
    ) -> None:
        self.args = args
        self.experiment = experiment or load_experiment(args.config_path)

    def _resume_path(self) -> str | Path | None:
        if self.args.auto_resume:
            return latest_checkpoint(self.args.output_path)
        return self.args.resume

    def train(self) -> TrainingResult:
        """执行训练并返回最后保存的 checkpoint。"""

        checkpoint = run_training(
            self.experiment,
            self.args.data_path,
            self.args.output_path,
            resume=self._resume_path(),
            initialize_from=self.args.initialize_from,
        )
        return TrainingResult(checkpoint=Path(checkpoint), output_dir=self.args.output_path)

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """评估入口预留给统一评估协议。"""

        raise NotImplementedError(
            "统一评估协议正在接入；当前请使用 ttt.eval_retrieval 或评估章节中的显式入口。"
        )

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """推理入口预留给统一生成协议。"""

        raise NotImplementedError("请使用 ttt.generate 或 ttt.serve；统一 Trainer 推理入口尚未接入。")
