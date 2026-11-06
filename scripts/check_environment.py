"""检查当前机器是否满足 TTT 运行前提。"""

import argparse
import json
import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ttt.provenance import environment_report


def main():
    parser = argparse.ArgumentParser(description="检查 TTT 运行环境")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--min-gpus", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = environment_report()
    gpu_count = sum(device.platform == "gpu" for device in jax.local_devices())
    report["gpu_count"] = gpu_count
    report["passed"] = not args.require_gpu or gpu_count >= args.min_gpus
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(f"GPU 数量不足：需要 {args.min_gpus}，实际 {gpu_count}")


if __name__ == "__main__":
    main()
