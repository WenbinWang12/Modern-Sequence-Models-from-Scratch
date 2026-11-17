"""用贪心生成评估 GSM8K 的数值答案。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

from ttt.checkpoint import load_checkpoint
from ttt.data import load_tokenizer
from ttt.models.ttt_e2e.modeling_ttt_e2e import TTTE2EForCausalLM
from ttt.runtime import TTTSession

_ANSWER = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(value: str) -> str:
    return value.strip().replace(",", "")


def _gold_answer(answer: str) -> str | None:
    match = _ANSWER.search(answer)
    if match:
        return _normalize(match.group(1))
    numbers = _NUMBER.findall(answer)
    return _normalize(numbers[-1]) if numbers else None


def _predicted_answer(text: str) -> str | None:
    match = _ANSWER.search(text)
    if match:
        return _normalize(match.group(1))
    numbers = _NUMBER.findall(text)
    return _normalize(numbers[-1]) if numbers else None


def evaluate(
    checkpoint_path: Path, data_path: Path, *, limit: int | None, max_new_tokens: int, no_adapt: bool
) -> dict[str, object]:
    checkpoint = load_checkpoint(checkpoint_path, load_optimizer=False)
    experiment = checkpoint.experiment
    tokenizer = load_tokenizer(experiment.tokenizer)
    model = TTTE2EForCausalLM(experiment.model)
    inner = replace(experiment.inner, enabled=False) if no_adapt else experiment.inner
    session = TTTSession(model, checkpoint.state.params, inner)
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    correct = skipped = 0
    records = []
    for index, row in enumerate(rows):
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            skipped += 1
            continue
        prompt = "Question: " + question.strip() + "\nAnswer:"
        try:
            session.reset()
            generated = session.generate(
                tokenizer.encode(prompt),
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_k=0,
                seed=0,
                eos_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(generated)
        except (ValueError, RuntimeError):
            skipped += 1
            continue
        gold = _gold_answer(answer)
        predicted = _predicted_answer(text)
        correct += int(gold is not None and predicted == gold)
        records.append({"index": index, "gold": gold, "prediction": predicted, "text": text})
    samples = len(records)
    metadata_path = checkpoint_path / "metadata.json" if checkpoint_path.is_dir() else checkpoint_path
    return {
        "protocol_version": "gsm8k_generate_v1",
        "protocol": "GSM8K test split; greedy generation; exact normalized numeric match",
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata_sha256": _sha256(metadata_path) if metadata_path.is_file() else None,
        "data": str(data_path.resolve()),
        "data_sha256": _sha256(data_path),
        "tokenizer": experiment.tokenizer,
        "adaptation": not no_adapt,
        "max_new_tokens": max_new_tokens,
        "limit": limit,
        "correct": correct,
        "samples": samples,
        "skipped": skipped,
        "accuracy": correct / samples if samples else None,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="评估 GSM8K 数值答案生成")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--no-adapt", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(
        args.checkpoint, args.data, limit=args.limit, max_new_tokens=args.max_new_tokens, no_adapt=args.no_adapt
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
