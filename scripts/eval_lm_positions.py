#!/usr/bin/env python3
"""按上下文位置统计 held-out next-token loss。

把评测集的每篇文档切成互不重叠、长度为 ``--window`` 的窗口，窗口内从空状态开始
（KV 缓存和快权重都重置），记录每个位置的 NLL。位置 i 的 loss 就是“已有 i 个 token
上下文时”的预测损失，因此按位置分桶后可以直接画出 loss 随上下文长度的曲线。

同一份脚本用于全注意力、滑窗注意力和 TTT-E2E；TTT 的测试时更新遵循训练时的
先预测、后更新顺序。可用 ``--no-adapt`` 关闭更新，或用 ``--sliding-window`` /
``--force-sliding`` 在评测时改变注意力可见范围（参数不变）。

输出：JSON 摘要，以及 ``.npz``（每个窗口、每个位置桶的 loss 和计数，供配对 bootstrap）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ttt.checkpoint import load_checkpoint
from ttt.inner_loop import sequence_loss
from ttt.models.ttt_e2e.modeling_ttt_e2e import TTTE2EForCausalLM


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_corpus(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    tokens = np.concatenate([np.load(directory / record["tokens"]) for record in manifest["shards"]])
    starts = np.load(directory / "documents.npy")
    if starts[-1] != tokens.size:
        raise ValueError("文档索引与 token 总数不一致。")
    return manifest, tokens, starts


def enumerate_windows(starts, window, *, include_partial, windows_per_document, max_windows):
    rows = []
    for document in range(len(starts) - 1):
        begin, end = int(starts[document]), int(starts[document + 1])
        position, count = begin, 0
        while position + 1 < end:
            length = min(window, end - position - 1)
            if length < window and not include_partial:
                break
            rows.append((document, position, length))
            position += window
            count += 1
            if windows_per_document is not None and count >= windows_per_document:
                break
        if max_windows is not None and len(rows) >= max_windows:
            return rows[:max_windows]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="按位置统计 held-out loss")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path, help="tokenize_corpus.py 的输出目录")
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--bin", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--include-partial", action="store_true")
    parser.add_argument("--windows-per-document", type=int)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--stream", action="store_true", help="忽略文档边界，把语料当作一条连续 token 流切窗")
    parser.add_argument("--no-adapt", action="store_true")
    parser.add_argument("--force-adapt", action="store_true", help="对未做元训练的检查点也执行测试时更新")
    parser.add_argument("--inner-learning-rate", type=float)
    parser.add_argument("--sliding-window", type=int, help="评测时覆盖滑窗长度")
    parser.add_argument("--force-sliding", action="store_true", help="评测时把所有层改成滑窗注意力")
    parser.add_argument("--chunk-size", type=int, help="评测时覆盖 TTT 更新块长度")
    parser.add_argument("--label", default="")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.window % args.bin:
        raise ValueError("window 必须是 bin 的整数倍。")

    checkpoint = load_checkpoint(args.checkpoint)
    experiment = checkpoint.experiment
    config = experiment.model
    changes = {}
    if args.force_sliding:
        changes["layer_types"] = ("sliding_attention",) * config.num_hidden_layers
    if args.sliding_window is not None:
        changes["sliding_window"] = args.sliding_window
    if args.chunk_size is not None:
        changes["ttt_chunk_size"] = args.chunk_size
        changes["loss_token_block_size"] = min(config.loss_token_block_size, args.chunk_size)
    if args.window > config.max_position_embeddings:
        changes["max_position_embeddings"] = args.window
    if changes:
        config = replace(config, **changes)
    model = TTTE2EForCausalLM(config)
    inner = experiment.inner
    if args.no_adapt:
        inner = replace(inner, enabled=False)
    if args.force_adapt:
        # 朴素测试时更新（dynamic evaluation）：模型训练时没有见过内循环。
        if not config.use_fast_mlp:
            raise ValueError("--force-adapt 需要带快权重 MLP 的模型。")
        inner = replace(inner, enabled=True)
    if args.inner_learning_rate is not None:
        inner = replace(inner, learning_rate=args.inner_learning_rate)
    params = checkpoint.state.params

    manifest, tokens, starts = load_corpus(args.data)
    if manifest["vocab_size"] != config.vocab_size:
        raise ValueError("评测数据的词表与模型不一致。")
    if args.stream:
        # 与训练采样一致：窗口可以跨越文档边界，文档之间只有 EOT 分隔。
        starts = np.asarray([0, tokens.size], dtype=np.int64)
    windows = enumerate_windows(
        starts,
        args.window,
        include_partial=args.include_partial,
        windows_per_document=args.windows_per_document,
        max_windows=args.max_windows,
    )
    if not windows:
        raise ValueError("没有可评测的窗口。")
    pad_token = config.pad_token_id if config.pad_token_id is not None else 0
    bins = args.window // args.bin

    def losses(weights, ids, mask):
        output = sequence_loss(model, weights, ids, inner, mask)
        # 每个更新块的快权重范数、更新幅度和内循环梯度范数，用于检查长上下文下的适应是否稳定。
        return output.token_losses, jnp.stack((output.fast_weight_norms, output.update_norms, output.grad_norms))

    evaluate = jax.jit(jax.vmap(losses, in_axes=(None, 0, 0)))
    loss_sums = np.zeros((len(windows), bins), dtype=np.float64)
    counts = np.zeros((len(windows), bins), dtype=np.float64)
    diagnostics = []
    compile_seconds, run_seconds, timed_tokens = None, 0.0, 0
    for offset in range(0, len(windows), args.batch_size):
        batch = windows[offset : offset + args.batch_size]
        ids = np.full((args.batch_size, args.window + 1), pad_token, dtype=np.int32)
        mask = np.zeros((args.batch_size, args.window), dtype=np.float32)
        for row, (_, position, length) in enumerate(batch):
            ids[row, : length + 1] = tokens[position : position + length + 1]
            mask[row, :length] = 1.0
        started = time.perf_counter()
        values, chunk_statistics = evaluate(params, jnp.asarray(ids), jnp.asarray(mask))
        values = np.asarray(values, dtype=np.float64)
        elapsed = time.perf_counter() - started
        diagnostics.append(np.asarray(chunk_statistics, dtype=np.float32)[: len(batch)])
        if compile_seconds is None:
            compile_seconds = elapsed
        else:
            run_seconds += elapsed
            timed_tokens += int(mask.sum())
        if not np.isfinite(values).all():
            raise FloatingPointError("评测出现非有限 loss。")
        rows = len(batch)
        loss_sums[offset : offset + rows] = values[:rows].reshape(rows, bins, args.bin).sum(axis=-1)
        counts[offset : offset + rows] = mask[:rows].reshape(rows, bins, args.bin).sum(axis=-1)

    total_tokens = float(counts.sum())
    nll = float(loss_sums.sum() / total_tokens)
    per_bin = loss_sums.sum(axis=0) / np.maximum(counts.sum(axis=0), 1.0)
    # 形状为 [窗口, 3, 更新块]；只对完整窗口取平均，避免尾部不完整块拉低统计。
    chunk_statistics = np.concatenate(diagnostics)
    complete = np.asarray([length == args.window for _, _, length in windows])
    chunk_means = chunk_statistics[complete].mean(axis=0) if complete.any() else chunk_statistics.mean(axis=0)
    bytes_per_token = manifest.get("stats", {}).get("bytes_per_token")
    metadata_path = args.checkpoint / "metadata.json" if args.checkpoint.is_dir() else args.checkpoint
    summary = {
        "protocol_version": "lm_positions_v1",
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.state.step),
        "checkpoint_metadata_sha256": sha256_file(metadata_path) if metadata_path.is_file() else None,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "data": str(args.data.resolve()),
        "data_manifest_sha256": sha256_file(args.data / "manifest.json"),
        "window": args.window,
        "bin": args.bin,
        "include_partial": args.include_partial,
        "stream": args.stream,
        "force_adapt": args.force_adapt,
        "adapt": bool(inner.enabled and config.use_fast_mlp),
        "inner_learning_rate": inner.learning_rate,
        "eval_overrides": {key: (list(value) if isinstance(value, tuple) else value) for key, value in changes.items()},
        "layer_types": sorted(set(config.layer_types)),
        "sliding_window": config.sliding_window,
        "ttt_chunk_size": config.ttt_chunk_size,
        "windows": len(windows),
        "documents": len({document for document, _, _ in windows}),
        "tokens": total_tokens,
        "nll": nll,
        "perplexity": math.exp(nll),
        "bits_per_byte": None if not bytes_per_token else nll / math.log(2) / bytes_per_token,
        "nll_by_position_bin": per_bin.tolist(),
        "fast_weight_norm_by_chunk": chunk_means[0].tolist(),
        "inner_update_norm_by_chunk": chunk_means[1].tolist(),
        "inner_grad_norm_by_chunk": chunk_means[2].tolist(),
        "compile_and_first_batch_seconds": compile_seconds,
        "eval_tokens_per_second": timed_tokens / run_seconds if run_seconds else None,
        "batch_size": args.batch_size,
        "device": str(jax.local_devices()[0]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        loss_sums=loss_sums.astype(np.float32),
        counts=counts.astype(np.float32),
        document=np.asarray([document for document, _, _ in windows], dtype=np.int32),
        position=np.asarray([position for _, position, _ in windows], dtype=np.int64),
    )
    printed = {key: value for key, value in summary.items() if not key.endswith(("_bin", "_by_chunk"))}
    print(json.dumps(printed, ensure_ascii=False))


if __name__ == "__main__":
    main()
