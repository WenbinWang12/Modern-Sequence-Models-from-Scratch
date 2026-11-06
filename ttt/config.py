"""实验配置的读写。"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .inner_loop import InnerLoopConfig
from .models.ttt_e2e.configuration_ttt_e2e import TTTE2EConfig
from .training import TrainConfig


@dataclass(frozen=True)
class ExperimentConfig:
    model: TTTE2EConfig
    inner: InnerLoopConfig
    training: TrainConfig
    tokenizer: str = "byte"

    def __post_init__(self):
        if self.training.sequence_length > self.model.max_position_embeddings:
            raise ValueError("训练长度不能超过模型位置上限。")
        if self.tokenizer == "byte" and self.model.vocab_size < 259:
            raise ValueError("byte tokenizer 需要至少 259 个词表位置。")

    def to_dict(self):
        return {
            "model": self.model.to_dict(),
            "inner": asdict(self.inner),
            "training": asdict(self.training),
            "tokenizer": self.tokenizer,
        }

    @classmethod
    def from_dict(cls, values):
        unknown = values.keys() - {"model", "inner", "training", "tokenizer"}
        if unknown:
            raise ValueError(f"未知实验字段：{sorted(unknown)}。")
        return cls(
            model=TTTE2EConfig.from_dict(values["model"]),
            inner=InnerLoopConfig(**values.get("inner", {})),
            training=TrainConfig(**values.get("training", {})),
            tokenizer=values.get("tokenizer", "byte"),
        )


def load_experiment(path):
    return ExperimentConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
