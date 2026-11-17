"""用 continuation log-likelihood 评估 ARC 或 PIQA 的选择题子集。"""

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
        raise ValueError("选择题 prompt 超过固定上下文长度")
    padded = full_ids + [tokenizer.pad_token_id] * (max_length - len(full_ids))
    logits = evaluate(jnp.asarray(padded, dtype=jnp.int32))
    answer_ids = full_ids[len(prompt_ids) :]
    return math.fsum(
        float(jax.nn.log_softmax(logits[len(prompt_ids) - 1 + index])[token_id])
        for index, token_id in enumerate(answer_ids)
    )


def _arc_row(row: dict[str, object]) -> tuple[str, list[str], int]:
    question = row["question"]
    if isinstance(question, str):
        stem = question
        choices = row["choices"]
        if not isinstance(choices, dict):
            raise ValueError("ARC choices 格式无效")
        texts = [str(value) for value in choices["text"]]
        labels = [str(value) for value in choices["label"]]
        answer = str(row.get("answerKey", ""))
    else:
        stem = str(question["stem"])
        choices = question["choices"]
        texts = [str(choice["text"]) for choice in choices]
        labels = [str(choice.get("label", "")) for choice in choices]
        answer = str(question.get("answerKey", ""))
    if answer in labels:
        index = labels.index(answer)
    elif answer.upper() in ("A", "B", "C", "D"):
        index = "ABCD".index(answer.upper())
    elif answer.isdigit():
        index = int(answer) - 1
    else:
        raise ValueError("ARC answerKey 无法映射到选项")
    if not 0 <= index < len(texts):
        raise ValueError("ARC answerKey 超出选项范围")
    return stem + "\nAnswer:", texts, index


def _piqa_row(row: dict[str, object]) -> tuple[str, list[str], int]:
    label = int(row["label"])
    choices = [str(row["sol1"]), str(row["sol2"])]
    if label not in (0, 1):
        raise ValueError("PIQA label 必须为 0 或 1")
    return str(row["goal"]) + "\nAnswer:", choices, label


def _gpqa_row(row: dict[str, object]) -> tuple[str, list[str], int]:
    choices = [str(value) for value in row["choices"]]
    answer = str(row["answer"]).strip()
    labels = "ABCD"
    if answer.upper() in labels:
        index = labels.index(answer.upper())
    elif answer.isdigit():
        index = int(answer)
    else:
        raise ValueError("GPQA answer 无法映射到选项")
    if not 0 <= index < len(choices):
        raise ValueError("GPQA answer 超出选项范围")
    return str(row["question"]) + "\nAnswer:", choices, index


def evaluate(checkpoint_path: Path, data_path: Path, task: str, *, limit: int | None = None) -> dict[str, object]:
    checkpoint = load_checkpoint(checkpoint_path, load_optimizer=False)
    experiment = checkpoint.experiment
    model = TTTE2EForCausalLM(experiment.model)
    inner = replace(experiment.inner, enabled=False)
    tokenizer = ByteTokenizer()
    max_length = min(experiment.model.max_position_embeddings, 512)
    evaluate_logits = jax.jit(
        lambda tokens: (
            sequence_loss(
                model,
                checkpoint.state.params,
                tokens,
                inner,
                jnp.ones((max_length - 1,), dtype=jnp.float32),
                return_logits=True,
            ).logits
        )
    )
    with data_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    correct = skipped = 0
    parse_row = {"arc": _arc_row, "piqa": _piqa_row, "gpqa": _gpqa_row}[task]
    for row in rows:
        try:
            prompt, choices, answer = parse_row(row)
            scores = [_score(evaluate_logits, tokenizer, prompt, choice, max_length=max_length) for choice in choices]
            correct += int(max(range(len(scores)), key=scores.__getitem__) == answer)
        except (KeyError, TypeError, ValueError, IndexError):
            skipped += 1
    samples = len(rows) - skipped
    metadata_path = checkpoint_path / "metadata.json" if checkpoint_path.is_dir() else checkpoint_path
    split = "test" if task in ("arc", "gpqa") else "validation"
    return {
        "protocol_version": "arc_piqa_loglikelihood_v1",
        "protocol": f"{task} {split} split; zero-shot answer-text continuation log-likelihood; byte tokenizer; no adaptation",
        "task": task,
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
    parser = argparse.ArgumentParser(description="评估 ARC 或 PIQA continuation log-likelihood")
    parser.add_argument("--task", choices=("arc", "piqa", "gpqa"), required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.data, args.task, limit=args.limit)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
