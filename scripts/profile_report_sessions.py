#!/usr/bin/env python3
"""测量单个推理会话的状态大小和逐 token 解码速度。

对一个检查点，在若干上下文长度下：先把上下文一次读入（prefill），再逐个 token 解码
``--decode-tokens`` 个 token（对 TTT-E2E 取更新块长度的整数倍，这样恰好包含整数次测试时更新）。
记录会话状态各部分占用的字节数——快权重、KV 缓存、未完成的更新块——以及解码速度。

状态大小由实际分配的数组算出，不是按公式估计。解码速度反映的是这份研究实现在批大小为 1 时的
表现，包含每个 token 一次的 Python 调度开销，只用于比较它随上下文长度的变化。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

from ttt.checkpoint import load_checkpoint
from ttt.models.ttt_e2e.modeling_ttt_e2e import TTTE2EForCausalLM
from ttt.runtime import TTTSession


def main() -> None:
    parser = argparse.ArgumentParser(description="推理会话的状态大小与解码速度")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path, help="tokenize_corpus.py 的输出目录，用作真实上下文")
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[2048, 8192, 32768])
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--no-adapt", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    experiment = checkpoint.experiment
    model = TTTE2EForCausalLM(experiment.model)
    inner = replace(experiment.inner, enabled=False) if args.no_adapt else experiment.inner
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    tokens = np.load(args.data / manifest["shards"][0]["tokens"])
    session = TTTSession(model, checkpoint.state.params, inner)
    rows = []
    for length in args.context_lengths:
        context = tokens[:length].astype(np.int32)
        continuation = tokens[length : length + args.decode_tokens].astype(np.int32)
        timings = []
        for _ in range(2):  # 第一遍用于编译，只记录第二遍
            session.reset()
            started = time.perf_counter()
            session.prefill(context).block_until_ready()
            prefill = time.perf_counter() - started
            started = time.perf_counter()
            for token in continuation:
                # 逐 token 喂入真实的后续文本：与自回归生成的计算量相同，但不依赖采样结果。
                session.decode(int(token)).block_until_ready()
            timings.append((prefill, time.perf_counter() - started))
        prefill, decode = timings[-1]
        state = session.state_bytes()
        rows.append(
            {
                "context_length": length,
                "decode_tokens": args.decode_tokens,
                "prefill_seconds": prefill,
                "prefill_tokens_per_second": length / prefill,
                "decode_seconds": decode,
                "decode_tokens_per_second": args.decode_tokens / decode,
                "inner_updates_total": session.state.updates,
                "state_bytes": {key: int(value) for key, value in state.items()},
                "persistent_state_bytes": int(state["fast_weights"] + state["kv_cache"] + state["pending_chunk"]),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    memory = jax.local_devices()[0].memory_stats() or {}
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.state.step),
        "adapt": bool(inner.enabled and experiment.model.use_fast_mlp),
        "layer_types": sorted(set(experiment.model.layer_types)),
        "sliding_window": experiment.model.sliding_window,
        "ttt_chunk_size": experiment.model.ttt_chunk_size,
        "fast_weight_parameters": int(sum(value.size for value in jax.tree.leaves(session.state.fast_weights))),
        "compute_dtype": experiment.model.compute_dtype,
        "device": str(jax.local_devices()[0]),
        "peak_bytes_in_use": memory.get("peak_bytes_in_use"),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
