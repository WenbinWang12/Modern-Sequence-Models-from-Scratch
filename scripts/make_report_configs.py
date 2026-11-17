#!/usr/bin/env python3
"""生成技术报告实验使用的全部配置（只依赖标准库）。

主实验（``main_*``）：约 125M 参数、GPT-2 BPE、上下文 4096、FineWeb-Edu/PG-19 各 50% 的
窗口混合，四个臂共享数据顺序、优化器和 token 预算：

- ``full``：12 层全注意力，上下文内无损可见；
- ``swa``：12 层滑窗注意力（k=512），窗口外信息不可见；
- ``ck``：与 TTT 相同的结构和参数量（末 3 层带 MLP′），但不做测试时更新；
- ``ttt``：TTT-E2E，末 3 层 MLP′ 每 256 个 token 做一次内循环更新。

消融（``abl_*``）：约 50M 参数、上下文 2048，每次只改变一个因素。
规模对照（``scale_*``）：三种模型规模在同一 token 预算下的 full / swa / ttt。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VOCAB = 50257
EOT = 50256

SIZES = {
    "xs": dict(hidden=384, intermediate=1024, layers=6, heads=6, suffix=2),
    "s": dict(hidden=512, intermediate=1408, layers=8, heads=8, suffix=2),
    "m": dict(hidden=768, intermediate=2048, layers=12, heads=12, suffix=3),
}


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


def experiment(model, training, *, inner_enabled, inner_lr=1.0, inner_clip=1.0, first_order=False):
    inner = {"enabled": inner_enabled, "learning_rate": inner_lr, "clip_norm": inner_clip}
    if first_order:
        inner["first_order"] = True
    return {"model": model, "inner": inner, "training": training, "tokenizer": "tiktoken:gpt2"}


def main_arms(seed=0):
    size = dict(SIZES["m"], window=512, chunk=256)
    arms = {}
    for kind in ("full", "swa", "ck", "ttt"):
        # 梯度检查点只改变显存/速度的取舍，不改变数值；全注意力不开会超出单卡显存。
        training = training_config(steps=12000, batch=16, length=4096, seed=seed, checkpointing=kind in ("full", "ck"))
        arms[kind] = experiment(model_config(kind=kind, **size), training, inner_enabled=kind == "ttt")
    return arms


def small_arm(kind="ttt", *, size="s", steps=5000, inner_lr=1.0, seed=0, first_order=False, **changes):
    """消融和规模对照共用的设置：上下文 2048、窗口 256、更新块 128。"""
    shape = {**SIZES[size], "window": 256, "chunk": 128, **changes}
    training = training_config(
        steps=steps, batch=16, length=2048, seed=seed, checkpointing=kind == "full", checkpoint_every=steps
    )
    return experiment(
        model_config(kind=kind, **shape),
        training,
        inner_enabled=kind == "ttt",
        inner_lr=inner_lr,
        first_order=first_order,
    )


def ablation_arms():
    arms = {
        "full": small_arm("full"),
        "swa": small_arm("swa"),
        "ck": small_arm("ck"),
        "ttt": small_arm("ttt"),
        "ttt_fo": small_arm(first_order=True),
        # 把所有 MLP 收窄到 1120，使总参数量与不带 MLP′ 的基线相同。
        "ttt_matched": small_arm(intermediate=1120),
        "ttt_ilr0p1": small_arm(inner_lr=0.1),
        "ttt_ilr0p3": small_arm(inner_lr=0.3),
        "ttt_ilr3": small_arm(inner_lr=3.0),
        "ttt_chunk64": small_arm(chunk=64),
        "ttt_chunk256": small_arm(chunk=256),
        "ttt_suffix1": small_arm(suffix=1),
        "ttt_suffix4": small_arm(suffix=4),
        "ttt_window128": small_arm(window=128),
        "swa_window128": small_arm("swa", window=128),
        "ttt_window512": small_arm(window=512),
        "swa_window512": small_arm("swa", window=512),
    }
    for seed in (1, 2):
        for kind in ("full", "swa", "ttt"):
            arms[f"{kind}_seed{seed}"] = small_arm(kind, seed=seed)
    return arms


def scaling_arms():
    return {f"{size}_{kind}": small_arm(kind, size=size) for size in ("xs", "m") for kind in ("full", "swa", "ttt")}


def main() -> None:
    parser = argparse.ArgumentParser(description="生成技术报告实验配置")
    parser.add_argument("--output", default="configs/report")
    parser.add_argument("--iso-steps", type=int, help="额外生成与 TTT 等训练时间的滑窗基线（步数由实测决定）")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.iso_steps:
        # 只写这一个文件，避免在其他训练进程读取配置时重写它们。
        groups = {"abl": {"swa_iso": small_arm("swa", steps=args.iso_steps)}}
    else:
        groups = {"main": main_arms(), "abl": ablation_arms(), "scale": scaling_arms()}
    written = []
    for prefix, arms in groups.items():
        for name, value in arms.items():
            path = output / f"{prefix}_{name}.json"
            path.write_bytes((json.dumps(value, indent=2) + "\n").encode("utf-8"))
            written.append(path.name)
    print(json.dumps({"written": len(written), "files": written}))


if __name__ == "__main__":
    main()
