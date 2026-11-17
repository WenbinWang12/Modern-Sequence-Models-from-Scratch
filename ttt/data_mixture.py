"""按来源概率采样多个 token manifest。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .dataset import TokenDataset, file_sha256

__all__ = ["MixtureBatchStream"]


class MixtureBatchStream:
    """把多个 TokenDataset 组合成一个可恢复的窗口采样器。"""

    recipe_format = "ttt_mixture_v1"

    def __init__(self, recipe_path, *, sequence_length, batch_size, vocab_size, seed=0):
        self.recipe_path = Path(recipe_path).expanduser().resolve()
        recipe = json.loads(self.recipe_path.read_text(encoding="utf-8"))
        if recipe.get("format") != self.recipe_format:
            raise ValueError("数据混合 recipe 格式不受支持。")
        if recipe.get("selection_unit", "window") != "window":
            raise ValueError("当前混合采样器只支持按训练窗口选择来源。")
        sources = recipe.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError("混合 recipe 至少需要一个来源。")
        if recipe.get("exhaustion_policy", "with_replacement") != "with_replacement":
            raise ValueError("当前混合采样器只支持 with_replacement。")
        names, paths, probabilities = [], [], []
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("每个混合来源必须是对象。")
            name = source.get("name")
            relative_path = source.get("data")
            probability = source.get("probability")
            if not isinstance(name, str) or not name:
                raise ValueError("混合来源需要非空 name。")
            if name in names:
                raise ValueError(f"混合来源名称重复：{name}")
            if not isinstance(relative_path, str):
                raise ValueError(f"来源 {name} 缺少 data 路径。")
            if not isinstance(probability, (int, float)) or not math.isfinite(float(probability)) or probability <= 0:
                raise ValueError(f"来源 {name} 的 probability 必须为正数。")
            names.append(name)
            paths.append((self.recipe_path.parent / relative_path).resolve())
            probabilities.append(float(probability))
        total = sum(probabilities)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("混合来源的 probability 总和必须为有限正数。")
        self.names = tuple(names)
        self.probabilities = np.asarray(probabilities, dtype=np.float64) / total
        self.datasets = tuple(TokenDataset(path, vocab_size) for path in paths)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.recipe_sha256 = file_sha256(self.recipe_path)
        self.source_sha256 = tuple(dataset.fingerprint for dataset in self.datasets)
        digest = hashlib.sha256()
        digest.update(self.recipe_sha256.encode())
        for name, source_sha256 in zip(self.names, self.source_sha256):
            digest.update(name.encode())
            digest.update(source_sha256.encode())
        self.fingerprint = digest.hexdigest()
        self.size = max(dataset.size for dataset in self.datasets)

        if any(dataset.size < sequence_length + 1 for dataset in self.datasets):
            raise ValueError("混合来源中有数据不足一个训练窗口。")

    def _sample_one(self, rng):
        source_index = int(rng.choice(len(self.datasets), p=self.probabilities))
        dataset = self.datasets[source_index]
        start = int(rng.integers(0, dataset.size - self.sequence_length))
        tokens, masks = dataset.rows([start], self.sequence_length + 1)
        return source_index, tokens[0], masks[0]

    def next_batch_with_masks(self):
        rows = [self._sample_one(self.rng) for _ in range(self.batch_size)]
        source_ids, tokens, masks = zip(*rows)
        self.last_source_ids = tuple(source_ids)
        return np.asarray(tokens, dtype=np.int32), np.asarray(masks, dtype=np.float32)

    def sample_rows(self, step, indices):
        rows = []
        source_ids = []
        for index in indices:
            rng = np.random.default_rng(np.random.SeedSequence((self.seed, int(step), int(index))))
            source_index, tokens, masks = self._sample_one(rng)
            source_ids.append(source_index)
            rows.append((tokens, masks))
        self.last_source_ids = tuple(source_ids)
        return (
            np.asarray([row[0] for row in rows], dtype=np.int32),
            np.asarray([row[1] for row in rows], dtype=np.float32),
        )

    def state_dict(self):
        return {
            "kind": self.recipe_format,
            "source_names": list(self.names),
            "rng": self.rng.bit_generator.state,
            "recipe_sha256": self.recipe_sha256,
            "source_sha256": list(self.source_sha256),
            "probabilities": self.probabilities.tolist(),
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }

    def load_state_dict(self, state):
        expected = (
            self.recipe_sha256,
            tuple(self.source_sha256),
            tuple(self.names),
            tuple(float(value) for value in self.probabilities),
            self.sequence_length,
            self.batch_size,
            self.seed,
        )
        actual = (
            state.get("recipe_sha256"),
            tuple(state.get("source_sha256", ())),
            tuple(state.get("source_names", ())),
            tuple(float(value) for value in state.get("probabilities", ())),
            state.get("sequence_length"),
            state.get("batch_size"),
            state.get("seed", self.seed),
        )
        if actual != expected:
            raise ValueError("恢复训练时混合 recipe、来源摘要或采样配置不一致。")
        self.rng.bit_generator.state = state["rng"]
