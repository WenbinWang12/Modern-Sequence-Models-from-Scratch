"""用无适配的 continuation log-likelihood 评估 MMLU。

该脚本只实现可复核的零样本字母选择协议：问题和四个选项组成提示，
分别计算 `` A``、`` B``、`` C``、`` D`` 的条件对数概率，取分数最高者。
它不执行 instruction tuning，也不把 TTT 的会话内更新混入基准分数。
"""

from __future__ import annotations

import argparse
import csv
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

__all__ = ["evaluate_subjects", "main"]

LABELS = ("A", "B", "C", "D")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prompt(row: list[str]) -> str:
    """按 MMLU CSV 的六列格式构造零样本提示。"""
    question, *choices, _answer = row
    lines = [question.strip(), ""]
    lines.extend(f"{label}. {choice.strip()}" for label, choice in zip(LABELS, choices))
    lines.append("Answer:")
    return "\n".join(lines)


def _score_choice(
    evaluate,
    tokenizer: ByteTokenizer,
    prompt: str,
    label: str,
    *,
    max_length: int,
) -> float:
    """返回答案字母 continuation 的总对数概率。"""
    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(prompt + " " + label)
    answer_start = len(prompt_ids)
    if len(full_ids) > max_length:
        raise ValueError("MMLU prompt exceeds the fixed model context window")
    padded = full_ids + [tokenizer.pad_token_id] * (max_length - len(full_ids))
    logits = evaluate(jnp.asarray(padded, dtype=jnp.int32))
    token_ids = full_ids[answer_start:]
    scores = []
    for offset, token_id in enumerate(token_ids):
        position = answer_start - 1 + offset
        scores.append(float(jax.nn.log_softmax(logits[position])[token_id]))
    return math.fsum(scores)


def evaluate_subjects(
    checkpoint_path: Path,
    data_dir: Path,
    subjects: list[str],
    *,
    limit: int | None = None,
) -> dict[str, object]:
    """评估指定科目，并保留跳过样本以便审计有效样本数。"""
    checkpoint = load_checkpoint(checkpoint_path, load_optimizer=False)
    experiment = checkpoint.experiment
    model = TTTE2EForCausalLM(experiment.model)
    # MMLU 分数固定为 no-adapt；会话适配应单独报告，不能和公开榜单混用。
    inner = replace(experiment.inner, enabled=False)
    tokenizer = ByteTokenizer()
    max_length = min(experiment.model.max_position_embeddings, 512)
    evaluate = jax.jit(
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
    per_subject: dict[str, dict[str, object]] = {}
    total_correct = total_count = total_skipped = 0
    data_hashes: dict[str, str] = {}
    for subject in subjects:
        path = data_dir / "test" / f"{subject}_test.csv"
        data_hashes[subject] = _sha256(path)
        rows = list(csv.reader(path.open(encoding="utf-8")))
        if limit is not None:
            rows = rows[:limit]
        correct = skipped = 0
        for row in rows:
            if len(row) != 6:
                skipped += 1
                continue
            prompt = _prompt(row)
            try:
                scores = [_score_choice(evaluate, tokenizer, prompt, label, max_length=max_length) for label in LABELS]
            except ValueError:
                skipped += 1
                continue
            prediction = LABELS[max(range(len(scores)), key=scores.__getitem__)]
            correct += int(prediction == row[-1].strip().upper())
        evaluated = len(rows) - skipped
        per_subject[subject] = {
            "correct": correct,
            "samples": evaluated,
            "skipped": skipped,
            "accuracy": correct / evaluated if evaluated else None,
        }
        total_correct += correct
        total_count += evaluated
        total_skipped += skipped
    metadata_path = checkpoint_path / "metadata.json" if checkpoint_path.is_dir() else checkpoint_path
    return {
        "protocol_version": "mmlu_loglikelihood_v1",
        "protocol": "MMLU test split; zero-shot answer-letter continuation log-likelihood; byte tokenizer; no adaptation",
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata_sha256": _sha256(metadata_path) if metadata_path.is_file() else None,
        "script_sha256": _sha256(Path(__file__).resolve()),
        "data_dir": str(data_dir.resolve()),
        "data_sha256": data_hashes,
        "subjects": subjects,
        "limit_per_subject": limit,
        "max_length": max_length,
        "correct": total_correct,
        "samples": total_count,
        "skipped": total_skipped,
        "accuracy": total_correct / total_count if total_count else None,
        "per_subject": per_subject,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评估 MMLU 零样本 continuation log-likelihood")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = evaluate_subjects(args.checkpoint, args.data_dir, args.subjects, limit=args.limit)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
