"""运行可复现的长上下文 key-value 检索诊断。"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

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


def _prompt(tokenizer: ByteTokenizer, target_length: int, rng: np.random.Generator, sample: int) -> tuple[str, str]:
    key = f"KEY-{sample:06d}"
    value = f"{int(rng.integers(100000, 1000000))}"
    lines = [f"The target record is {key}: {value}."]
    filler = "The archive contains an ordinary record with no special meaning."
    while len(tokenizer.encode("\n".join(lines) + "\nQuestion: value for " + key + "?\nAnswer:")) < target_length - 16:
        lines.append(filler)
    rng.shuffle(lines)
    prompt = "\n".join(lines) + f"\nQuestion: value for {key}?\nAnswer:"
    return prompt, value


def _run_mode(model, params, inner, tokenizer, lengths, samples, seed, max_new_tokens):
    results = []
    rng = np.random.default_rng(seed)
    for length in lengths:
        correct = skipped = 0
        for sample in range(samples):
            prompt, answer = _prompt(tokenizer, length, rng, sample)
            tokens = tokenizer.encode(prompt)
            if len(tokens) + max_new_tokens > model.config.max_position_embeddings:
                skipped += 1
                continue
            session = TTTSession(model, params, inner)
            generated = session.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
            )
            prediction = tokenizer.decode(generated).strip()
            correct += int(prediction == answer)
        results.append(
            {
                "context_length_request": length,
                "correct": correct,
                "samples": samples - skipped,
                "skipped": skipped,
                "exact_match": correct / (samples - skipped) if samples > skipped else None,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic long-context retrieval diagnostic")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--lengths", nargs="+", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--adapt", action="store_true", help="enable TTT session updates")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint, load_optimizer=False)
    experiment = checkpoint.experiment
    model = TTTE2EForCausalLM(experiment.model)
    inner = experiment.inner if args.adapt else replace(experiment.inner, enabled=False)
    tokenizer = ByteTokenizer()
    result = {
        "protocol_version": "synthetic_kv_retrieval_v1",
        "protocol": "random key-value retrieval; greedy generation; exact match; no training-data claims",
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata_sha256": _sha256(args.checkpoint / "metadata.json") if args.checkpoint.is_dir() else None,
        "tokenizer": "byte",
        "adapt": args.adapt,
        "lengths": args.lengths,
        "samples_per_length": args.samples,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "results": _run_mode(model, checkpoint.state.params, inner, tokenizer, args.lengths, args.samples, args.seed, args.max_new_tokens),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
