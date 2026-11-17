"""统计数据混合采样器的实际来源比例。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttt.data_mixture import MixtureBatchStream


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe source proportions of a mixture recipe")
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--batches", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=259)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batches <= 0:
        raise SystemExit("--batches must be positive")
    stream = MixtureBatchStream(
        args.recipe,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    counts = [0] * len(stream.names)
    for _ in range(args.batches):
        stream.next_batch_with_masks()
        for source_id in stream.last_source_ids:
            counts[source_id] += 1
    total = sum(counts)
    result = {
        "recipe": str(args.recipe.resolve()),
        "recipe_sha256": stream.recipe_sha256,
        "source_names": list(stream.names),
        "target_probabilities": stream.probabilities.tolist(),
        "observed_counts": counts,
        "observed_probabilities": [count / total for count in counts],
        "windows": total,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
