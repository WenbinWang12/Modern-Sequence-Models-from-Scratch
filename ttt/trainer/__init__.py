"""训练器公共接口。"""

from .trainer import Trainer, TrainingResult
from .training_args import TrainingArguments

__all__ = ["Trainer", "TrainingArguments", "TrainingResult"]
