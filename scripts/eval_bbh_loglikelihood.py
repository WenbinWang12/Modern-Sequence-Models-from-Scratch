"""用 continuation log-likelihood 评估 BBH 的单个子任务诊断。"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from ttt.checkpoint import load_checkpoint
from ttt.data import ByteTokenizer
from ttt.models.ttt_e2e.modeling_ttt_e2e import TTTE2EForCausalLM
from ttt.runtime import TTTSession


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def evaluate(checkpoint_path: Path, data_path: Path, *, limit: int | None) -> dict[str, object]:
    checkpoint = load_checkpoint(checkpoint_path, load_optimizer=False)
    experiment = checkpoint.experiment
    model = TTTE2EForCausalLM(experiment.model)
    inner = replace(experiment.inner, enabled=False)
    tokenizer = ByteTokenizer()
    max_length = min(experiment.model.max_position_embeddings, 512)
    session = TTTSession(model, checkpoint.state.params, inner)
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    rows = payload["examples"]
    if limit is not None:
        rows = rows[:limit]
    correct = skipped = 0
    for row in rows:
        try:
            prompt = str(row["input"])
            target = _normalize(str(row["target"]))
            if len(tokenizer.encode(prompt)) + 64 > model.config.max_position_embeddings:
                raise ValueError("BBH prompt exceeds the generation context window")
            session.reset()
            generated = session.generate(
                tokenizer.encode(prompt),
                max_new_tokens=64,
                temperature=0.0,
                top_k=0,
                seed=0,
                eos_token_id=tokenizer.eos_token_id,
            )
            prediction = _normalize(tokenizer.decode(generated))
            correct += int(prediction == target or prediction.startswith(target + " "))
        except (KeyError, TypeError, ValueError, RuntimeError):
            skipped += 1
    samples = len(rows) - skipped
    metadata_path = checkpoint_path / "metadata.json" if checkpoint_path.is_dir() else checkpoint_path
    return {
        "protocol_version": "bbh_greedy_v1",
        "protocol": "BBH single-task zero-shot greedy generation; normalized exact match; byte tokenizer; no adaptation",
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata_sha256": _sha256(metadata_path) if metadata_path.is_file() else None,
        "script_sha256": _sha256(Path(__file__).resolve()),
        "data": str(data_path.resolve()),
        "data_sha256": _sha256(data_path),
        "limit": limit,
        "max_length": max_length,
        "samples": samples,
        "skipped": skipped,
        "correct": correct,
        "accuracy": correct / samples if samples else None,
        "note": "Diagnostic only: standard BBH leaderboard scoring requires the official subtask, shot prompt, and answer parser.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 BBH 单子任务 continuation 诊断")
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
