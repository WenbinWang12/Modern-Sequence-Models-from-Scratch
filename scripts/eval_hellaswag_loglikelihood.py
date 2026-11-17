"""用 continuation log-likelihood 做 HellaSwag 选择题诊断。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp

from ttt.checkpoint import load_checkpoint
from ttt.data import ByteTokenizer
from ttt.inner_loop import sequence_loss
from ttt.models.ttt_e2e.modeling_ttt_e2e import TTTE2EForCausalLM


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _score(evaluate, tokenizer: ByteTokenizer, prompt: str, continuation: str, *, max_length: int) -> float:
    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(prompt + " " + continuation.strip())
    if len(full_ids) > max_length:
        raise ValueError("HellaSwag prompt exceeds the fixed model context window")
    padded = full_ids + [tokenizer.pad_token_id] * (max_length - len(full_ids))
    logits = evaluate(jnp.asarray(padded, dtype=jnp.int32))
    answer_ids = full_ids[len(prompt_ids) :]
    return math.fsum(
        float(jax.nn.log_softmax(logits[len(prompt_ids) - 1 + index])[token_id])
        for index, token_id in enumerate(answer_ids)
    )


def evaluate(checkpoint_path: Path, data_path: Path, *, limit: int | None = None) -> dict[str, object]:
    checkpoint = load_checkpoint(checkpoint_path, load_optimizer=False)
    experiment = checkpoint.experiment
    model = TTTE2EForCausalLM(experiment.model)
    inner = replace(experiment.inner, enabled=False)
    tokenizer = ByteTokenizer()
    max_length = min(experiment.model.max_position_embeddings, 512)
    evaluate_logits = jax.jit(
        lambda tokens: sequence_loss(
            model,
            checkpoint.state.params,
            tokens,
            inner,
            jnp.ones((max_length - 1,), dtype=jnp.float32),
            return_logits=True,
        ).logits
    )
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    correct = skipped = 0
    for row in rows:
        try:
            endings = row["endings"]
            label = int(row["label"])
            scores = [_score(evaluate_logits, tokenizer, row["ctx"], ending, max_length=max_length) for ending in endings]
            prediction = max(range(len(scores)), key=scores.__getitem__)
            correct += int(prediction == label)
        except (KeyError, TypeError, ValueError, IndexError):
            skipped += 1
    samples = len(rows) - skipped
    metadata_path = checkpoint_path / "metadata.json" if checkpoint_path.is_dir() else checkpoint_path
    return {
        "protocol_version": "hellaswag_loglikelihood_v1",
        "protocol": "HellaSwag validation split; zero-shot continuation log-likelihood; byte tokenizer; no adaptation",
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata_sha256": _sha256(metadata_path) if metadata_path.is_file() else None,
        "script_sha256": _sha256(Path(__file__).resolve()),
        "data": str(data_path.resolve()),
        "data_sha256": _sha256(data_path),
        "limit": limit,
        "max_length": max_length,
        "correct": correct,
        "samples": samples,
        "skipped": skipped,
        "accuracy": correct / samples if samples else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="评估 HellaSwag continuation log-likelihood")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.data, limit=args.limit)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
