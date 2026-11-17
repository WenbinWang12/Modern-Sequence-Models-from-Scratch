"""把公开 parquet split 转成可审计的 JSONL 子集。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a parquet split to JSONL")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise SystemExit("需要可选依赖 pyarrow；仅 benchmark 数据转换需要") from error
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    table = parquet.read_table(args.input)
    if args.limit is not None:
        table = table.slice(0, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = table.to_pylist()
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"input": str(args.input), "output": str(args.output), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
