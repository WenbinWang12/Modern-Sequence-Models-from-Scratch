#!/usr/bin/env python3
"""删除已经取用过的中间检查点，只保留后续实验仍然需要的那些。

一个 138M 模型的检查点含参数和优化器状态，约 1.4 GiB；一次 12,000 步的运行会写 12 个。
主目录有配额限制（2026-11-17 曾因此让两个训练同时写盘失败），所以曲线评测取完数值之后，
中间检查点就应该删掉。

保留规则：最终检查点（后续所有评测都用它）、``--keep`` 指定的里程碑步数，以及曲线评测
尚未产生结果的检查点。默认只打印将要删除的内容，加 ``--apply`` 才真正删除。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="修剪中间检查点")
    parser.add_argument("--runs", nargs="+", required=True, help="运行目录，例如 outputs/main-ttt")
    parser.add_argument("--evals", type=Path, default=Path("outputs/evals"))
    parser.add_argument("--keep", nargs="*", type=int, default=[], help="额外保留的步数")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    removed, kept, freed = [], [], 0
    for name in args.runs:
        run = Path(name)
        checkpoints = sorted(run.glob("step_*.orbax"))
        if not checkpoints:
            continue
        final = max(checkpoints, key=lambda path: int(path.stem.split("_")[-1]))
        arm = run.name.split("-", 1)[-1]
        for checkpoint in checkpoints:
            step = int(checkpoint.stem.split("_")[-1])
            curve = args.evals / f"curve_{arm}_s{step:05d}__pg19val.json"
            # 曲线结果还没算出来的检查点不能删：删了这个点就永远补不上。
            reason = None
            if checkpoint == final:
                reason = "final"
            elif step in args.keep:
                reason = "milestone"
            elif not curve.exists():
                reason = "curve eval pending"
            if reason:
                kept.append({"path": str(checkpoint), "reason": reason})
                continue
            size = directory_bytes(checkpoint)
            freed += size
            removed.append({"path": str(checkpoint), "bytes": size})
            if args.apply:
                shutil.rmtree(checkpoint)
    print(
        json.dumps(
            {
                "applied": args.apply,
                "removed": len(removed),
                "kept": len(kept),
                "freed_gib": round(freed / 2**30, 2),
                "detail": {"removed": removed, "kept": kept},
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
