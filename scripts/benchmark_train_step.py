#!/usr/bin/env python3
"""测量单设备训练步的编译时间、吞吐和峰值显存。

使用随机 token，不读取数据；用于在正式训练前确定 batch、序列长度和注意力后端，
也用于技术报告中的成本表。``--set`` 可以临时覆盖模型或训练字段，例如：

    python scripts/benchmark_train_step.py --config configs/report/ctx4k_ttt.json \
        --set training.batch_size=16 --set model.ttt_chunk_size=128
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ttt.config import ExperimentConfig
from ttt.models.ttt_e2e.modeling_ttt_e2e import TTTE2EForCausalLM
from ttt.training import TrainState, make_optimizer, make_train_step


def apply_overrides(values: dict, overrides: list[str]) -> dict:
    for item in overrides:
        key, raw = item.split("=", 1)
        section, name = key.split(".", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        values.setdefault(section, {})[name] = value
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="训练步吞吐基准")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()

    values = apply_overrides(json.loads(Path(args.config).read_text(encoding="utf-8")), args.overrides)
    experiment = ExperimentConfig.from_dict(values)
    training = experiment.training
    model = TTTE2EForCausalLM(experiment.model, training.gradient_checkpointing)
    optimizer = make_optimizer(training)
    params = model.init_parameters(jax.random.PRNGKey(training.seed))
    state = TrainState(params, optimizer.init(params), jnp.asarray(0, dtype=jnp.int32))
    step = make_train_step(model, experiment.inner, training, optimizer)
    rng = np.random.default_rng(0)
    shape = (training.batch_size, training.sequence_length + 1)
    tokens = jnp.asarray(rng.integers(0, experiment.model.vocab_size, size=shape), dtype=jnp.int32)
    masks = jnp.ones((training.batch_size, training.sequence_length), dtype=jnp.float32)

    started = time.perf_counter()
    state, metrics = step(state, tokens, masks)
    first_loss = float(metrics["loss"])
    compile_seconds = time.perf_counter() - started
    durations = []
    for _ in range(args.steps):
        started = time.perf_counter()
        state, metrics = step(state, tokens, masks)
        float(metrics["loss"])
        durations.append(time.perf_counter() - started)
    memory = jax.local_devices()[0].memory_stats() or {}
    step_seconds = float(np.median(durations))
    tokens_per_step = training.batch_size * training.sequence_length
    result = {
        "config": args.config,
        "overrides": args.overrides,
        "device": str(jax.local_devices()[0]),
        "parameters": int(sum(value.size for value in jax.tree.leaves(params))),
        "fast_parameters": int(sum(value.size for value in jax.tree.leaves(model.initial_fast_weights(params)))),
        "batch_size": training.batch_size,
        "sequence_length": training.sequence_length,
        "inner_enabled": experiment.inner.enabled,
        "first_loss": first_loss,
        "last_loss": float(metrics["loss"]),
        "compile_and_first_step_seconds": compile_seconds,
        "median_step_seconds": step_seconds,
        "tokens_per_second": tokens_per_step / step_seconds,
        "peak_bytes_in_use": memory.get("peak_bytes_in_use"),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
