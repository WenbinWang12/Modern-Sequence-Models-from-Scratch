#!/usr/bin/env python3
"""生成技术报告实验使用的全部配置（只依赖标准库）。

主实验（``main_*``）：约 125M 参数、GPT-2 BPE、上下文 4096、FineWeb-Edu/PG-19 各 50% 的
窗口混合，四个臂共享数据顺序、优化器和 token 预算：

- ``full``：12 层全注意力，上下文内无损可见；
- ``swa``：12 层滑窗注意力（k=512），窗口外信息不可见；
- ``ck``：与 TTT 相同的结构和参数量（末 3 层带 MLP′），但不做测试时更新；
- ``ttt``：TTT-E2E，末 3 层 MLP′ 每 256 个 token 做一次内循环更新。

消融（``abl_*``）：约 45M 参数、上下文 2048，逐项改变一个因素。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VOCAB = 50257
EOT = 50256


def model_config(*, hidden, intermediate, layers, heads, window, kind, chunk, suffix, backend="blocked"):
    layer_types = ["full_attention" if kind == "full" else "sliding_attention"] * layers
    fast = kind in ("ck", "ttt")
    return {
        "vocab_size": VOCAB,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "layer_types": layer_types,
        "sliding_window": window,
        "max_position_embeddings": 65536,
        "attention_backend": backend,
        "use_fast_mlp": fast,
        "ttt_suffix_length": suffix if fast else 0,
        "ttt_chunk_size": chunk if fast else 1024,
        "loss_token_block_size": min(chunk, 256) if fast else 256,
        "loss_vocab_block_size": 65536,
        "compute_dtype": "bfloat16",
        "post_norm": False,
        "bos_token_id": EOT,
        "eos_token_id": EOT,
        "pad_token_id": EOT,
    }


def training_config(*, steps, batch, length, seed=0, learning_rate=6e-4, checkpointing=False, checkpoint_every=1000):
    return {
        "steps": steps,
        "batch_size": batch,
        "sequence_length": length,
        "learning_rate": learning_rate,
        "weight_decay": 0.1,
        "clip_norm": 1.0,
        "warmup_steps": max(1, steps // 25),
        "inner_warmup_steps": max(1, steps // 10),
        "inner_lr_start": 0.1,
        "checkpoint_every": checkpoint_every,
        "checkpoint_format": "orbax",
        "gradient_checkpointing": checkpointing,
        "log_every": 20,
        "seed": seed,
    }


def experiment(model, training, *, inner_enabled, inner_lr=1.0, inner_clip=1.0):
    inner = {"enabled": inner_enabled, "learning_rate": inner_lr, "clip_norm": inner_clip}
    return {"model": model, "inner": inner, "training": training, "tokenizer": "tiktoken:gpt2"}


def main_arms(seed=0):
    size = dict(hidden=768, intermediate=2048, layers=12, heads=12, window=512, chunk=256, suffix=3)
    arms = {}
    for kind in ("full", "swa", "ck", "ttt"):
        training = training_config(steps=12000, batch=16, length=4096, seed=seed, checkpointing=kind == "full")
        arms[kind] = experiment(model_config(kind=kind, **size), training, inner_enabled=kind == "ttt")
    return arms


def ablation_arms():
    size = dict(hidden=512, intermediate=1408, layers=8, heads=8, window=256, chunk=128, suffix=2)
    training = dict(steps=6000, batch=32, length=2048, checkpoint_every=2000)

    def arm(kind="ttt", *, inner_lr=1.0, seed=0, **changes):
        model = model_config(kind=kind, **{**size, **changes})
        return experiment(model, training_config(seed=seed, **training), inner_enabled=kind == "ttt", inner_lr=inner_lr)

    arms = {
        "full": arm("full"),
        "swa": arm("swa"),
        "ck": arm("ck"),
        "ttt": arm("ttt"),
        "ttt_ilr0p1": arm(inner_lr=0.1),
        "ttt_ilr0p3": arm(inner_lr=0.3),
        "ttt_ilr3": arm(inner_lr=3.0),
        "ttt_chunk64": arm(chunk=64),
        "ttt_chunk256": arm(chunk=256),
        "ttt_suffix1": arm(suffix=1),
        "ttt_suffix4": arm(suffix=4),
        "ttt_window128": arm(window=128),
        "swa_window128": arm("swa", window=128),
        "swa_window512": arm("swa", window=512),
        "ttt_window512": arm(window=512),
        "ttt_seed1": arm(seed=1),
        "swa_seed1": arm("swa", seed=1),
        "full_seed1": arm("full", seed=1),
        "ttt_seed2": arm(seed=2),
        "swa_seed2": arm("swa", seed=2),
        "full_seed2": arm("full", seed=2),
    }
    return arms


def main() -> None:
    parser = argparse.ArgumentParser(description="生成技术报告实验配置")
    parser.add_argument("--output", default="configs/report")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    written = []
    groups = {"main": main_arms(), "abl": ablation_arms()}
    for seed in (1, 2):
        for name, value in main_arms(seed).items():
            groups["main"][f"{name}_seed{seed}"] = value
    for prefix, arms in groups.items():
        for name, value in arms.items():
            path = output / f"{prefix}_{name}.json"
            path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
            written.append(path.name)
    print(json.dumps({"written": len(written), "files": written}))


if __name__ == "__main__":
    main()
