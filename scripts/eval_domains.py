#!/usr/bin/env python3
"""一次载入检查点，逐个领域评测 held-out loss。

每个领域的 held-out 语料被当作一条连续 token 流，切成互不重叠的定长窗口（与训练采样
一致，窗口可以跨文档边界），窗口内从空状态开始。输出每个领域的 NLL、困惑度和
bits-per-byte，以及对各领域等权的宏平均。bits-per-byte 不依赖每个 token 平均覆盖多少
字符，可以跨领域比较。

用法：scripts/eval_domains.py --checkpoint <dir> --data web=data/tokens/v2/web-heldout ... --output out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ttt.checkpoint import load_checkpoint
from ttt.inner_loop import sequence_loss
from ttt.models.ttt_e2e.modeling_ttt_e2e import TTTE2EForCausalLM


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def window_starts(tokens: int, window: int, limit: int) -> np.ndarray:
    """互不重叠的窗口起点；窗口多于 ``limit`` 时等间隔抽取，覆盖整份语料。"""
    available = (tokens - 1) // window
    count = min(available, limit)
    return np.arange(count, dtype=np.int64) * (available // max(count, 1)) * window


def main() -> None:
    parser = argparse.ArgumentParser(description="逐领域评测 held-out loss")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", nargs="+", required=True, metavar="NAME=DIR")
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-windows", type=int, default=900)
    parser.add_argument("--no-adapt", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    experiment = checkpoint.experiment
    config = experiment.model
    if args.window > config.max_position_embeddings:
        config = replace(config, max_position_embeddings=args.window)
    model = TTTE2EForCausalLM(config)
    inner = replace(experiment.inner, enabled=False) if args.no_adapt else experiment.inner
    params = checkpoint.state.params
    pad_token = config.pad_token_id if config.pad_token_id is not None else 0
    evaluate = jax.jit(
        jax.vmap(lambda weights, ids, mask: sequence_loss(model, weights, ids, inner, mask).token_losses, (None, 0, 0))
    )

    domains = {}
    for item in args.data:
        name, _, directory = item.partition("=")
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["vocab_size"] != config.vocab_size:
            raise ValueError(f"{name} 的词表与模型不一致。")
        tokens = np.concatenate([np.load(directory / record["tokens"]) for record in manifest["shards"]])
        starts = window_starts(tokens.size, args.window, args.max_windows)
        if not starts.size:
            raise ValueError(f"{name} 不足一个评测窗口。")
        total = 0.0
        for offset in range(0, starts.size, args.batch_size):
            batch = starts[offset : offset + args.batch_size]
            ids = np.full((args.batch_size, args.window + 1), pad_token, dtype=np.int32)
            mask = np.zeros((args.batch_size, args.window), dtype=np.float32)
            for row, start in enumerate(batch):
                ids[row] = tokens[start : start + args.window + 1]
                mask[row] = 1.0
            values = np.asarray(evaluate(params, jnp.asarray(ids), jnp.asarray(mask)), dtype=np.float64)
            if not np.isfinite(values).all():
                raise FloatingPointError(f"{name} 评测出现非有限 loss。")
            total += float((values * mask).sum())
        count = int(starts.size) * args.window
        nll = total / count
        bytes_per_token = manifest["stats"]["bytes_per_token"]
        domains[name] = {
            "data": str(directory),
            "data_manifest_sha256": sha256_file(directory / "manifest.json"),
            "windows": int(starts.size),
            "tokens": count,
            "nll": nll,
            "perplexity": math.exp(nll),
            "bytes_per_token": bytes_per_token,
            "bits_per_byte": nll / math.log(2) / bytes_per_token,
        }
        print(json.dumps({"domain": name, **{key: domains[name][key] for key in ("nll", "bits_per_byte")}}), flush=True)

    metadata = args.checkpoint / "metadata.json"
    summary = {
        "protocol_version": "domain_heldout_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.state.step),
        "checkpoint_metadata_sha256": sha256_file(metadata) if metadata.is_file() else None,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "window": args.window,
        "adapt": bool(inner.enabled and config.use_fast_mlp),
        "domains": domains,
        "macro_nll": float(np.mean([entry["nll"] for entry in domains.values()])),
        "macro_bits_per_byte": float(np.mean([entry["bits_per_byte"] for entry in domains.values()])),
        "device": str(jax.local_devices()[0]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
