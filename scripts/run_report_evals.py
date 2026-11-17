#!/usr/bin/env python3
"""按阶段批量运行技术报告所需的 held-out 评测，已有结果的任务自动跳过。

阶段：

- ``main``：四个主实验臂在 PG-19 test（按文档切窗）和 FineWeb-Edu held-out（连续流）上的
  上下文 4096 评测，以及 TTT 关闭测试时更新的对照；
- ``long``：超出训练长度的评测（窗口 8K/16K/32K），考察滑窗、TTT 与截断全注意力的外推；
- ``curves``：每个保存的 checkpoint 在固定验证子集上的 loss，用于验证曲线；
- ``latency``：不同上下文长度下的评测吞吐，用于成本图；
- ``ablations``：消融臂和规模对照臂在 PG-19 validation 上的评测；
- ``dynamic``：对没有做元训练的对照模型直接做测试时更新，以及 TTT 测试时学习率的敏感性。

每个任务是一次独立的 ``eval_lm_positions.py`` 进程，只使用 ``--gpu`` 指定的单张卡。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MAIN_ARMS = ("full", "swa", "ck", "ttt")
PG19_TEST = "data/tokens/pg19-test"
PG19_VALIDATION = "data/tokens/pg19-validation"
FINEWEB_HELDOUT = "data/tokens/fineweb-edu-heldout-dedup"


def latest_checkpoint(run: Path) -> Path | None:
    candidates = sorted(path for path in run.glob("step_*.orbax") if (path / "metadata.json").exists())
    return candidates[-1] if candidates else None


def finished(run: Path) -> bool:
    status = run / "status.json"
    return status.exists() and json.loads(status.read_text(encoding="utf-8")).get("status") == "complete"


def task(name, checkpoint, data, window, *, bin_size=64, batch=16, extra=()):
    return name, [
        "--checkpoint",
        str(checkpoint),
        "--data",
        data,
        "--window",
        str(window),
        "--bin",
        str(bin_size),
        "--batch-size",
        str(batch),
        "--label",
        name,
        *extra,
    ]


def main_tasks(outputs: Path, arms, prefix="main"):
    tasks = []
    for arm in arms:
        run = outputs / f"{prefix}-{arm}"
        if not finished(run):
            continue
        checkpoint = latest_checkpoint(run)
        batch = 8 if arm == "full" else 16
        tasks.append(task(f"{prefix}_{arm}__pg19test_w4096", checkpoint, PG19_TEST, 4096, batch=batch))
        tasks.append(
            task(
                f"{prefix}_{arm}__fineweb_w4096",
                checkpoint,
                FINEWEB_HELDOUT,
                4096,
                batch=batch,
                extra=("--stream", "--max-windows", "2048"),
            )
        )
        if arm.startswith("ttt"):
            for data, label, extra in (
                (PG19_TEST, "pg19test", ()),
                (FINEWEB_HELDOUT, "fineweb", ("--stream", "--max-windows", "2048")),
            ):
                tasks.append(
                    task(f"{prefix}_{arm}_noadapt__{label}_w4096", checkpoint, data, 4096, extra=("--no-adapt", *extra))
                )
    return tasks


def long_tasks(outputs: Path, arms):
    tasks = []
    for arm in arms:
        run = outputs / f"main-{arm}"
        if not finished(run):
            continue
        checkpoint = latest_checkpoint(run)
        if arm == "full":
            # 截断到训练长度的全注意力：每个 token 只看最近 4096 个 token，成本随长度线性增长。
            extra = ("--force-sliding", "--sliding-window", "4096")
            tasks.append(
                task(
                    "main_full_sw4096__pg19test_w32768",
                    checkpoint,
                    PG19_TEST,
                    32768,
                    bin_size=256,
                    batch=2,
                    extra=extra,
                )
            )
            # 不截断的全注意力：直接外推到训练长度之外。
            tasks.append(task("main_full__pg19test_w8192", checkpoint, PG19_TEST, 8192, bin_size=256, batch=4))
            tasks.append(
                task(
                    "main_full__pg19test_w16384",
                    checkpoint,
                    PG19_TEST,
                    16384,
                    bin_size=256,
                    batch=1,
                    extra=("--max-windows", "160"),
                )
            )
        else:
            tasks.append(task(f"main_{arm}__pg19test_w32768", checkpoint, PG19_TEST, 32768, bin_size=256, batch=4))
            if arm == "ttt":
                tasks.append(
                    task(
                        "main_ttt_noadapt__pg19test_w32768",
                        checkpoint,
                        PG19_TEST,
                        32768,
                        bin_size=256,
                        batch=4,
                        extra=("--no-adapt",),
                    )
                )
    return tasks


def curve_tasks(outputs: Path, arms):
    tasks = []
    for arm in arms:
        run = outputs / f"main-{arm}"
        for checkpoint in sorted(run.glob("step_*.orbax")):
            if not (checkpoint / "metadata.json").exists():
                continue
            step = int(checkpoint.stem.split("_")[-1])
            batch = 8 if arm == "full" else 16
            tasks.append(
                task(
                    f"curve_{arm}_s{step:05d}__pg19val",
                    checkpoint,
                    PG19_VALIDATION,
                    4096,
                    batch=batch,
                    extra=("--windows-per-document", "5"),
                )
            )
            tasks.append(
                task(
                    f"curve_{arm}_s{step:05d}__fineweb",
                    checkpoint,
                    FINEWEB_HELDOUT,
                    4096,
                    batch=batch,
                    extra=("--stream", "--max-windows", "256"),
                )
            )
    return tasks


def latency_tasks(outputs: Path, arms):
    tasks = []
    for arm in arms:
        run = outputs / f"main-{arm}"
        if not finished(run):
            continue
        checkpoint = latest_checkpoint(run)
        lengths = (1024, 2048, 4096, 8192, 16384) if arm == "full" else (1024, 2048, 4096, 8192, 16384, 32768)
        for length in lengths:
            # 固定每批 token 数，避免把批大小的差异误读成上下文长度的影响。
            batch = max(1, 32768 // length)
            tasks.append(
                task(
                    f"latency_{arm}__w{length}",
                    checkpoint,
                    PG19_TEST,
                    length,
                    bin_size=256,
                    batch=batch,
                    extra=("--max-windows", str(batch * 6)),
                )
            )
    return tasks


def ablation_tasks(outputs: Path, only=None):
    tasks = []
    for run in sorted([*outputs.glob("abl-*"), *outputs.glob("scale-*")]):
        if not finished(run) or (only and run.name not in only):
            continue
        checkpoint = latest_checkpoint(run)
        name = run.name.replace("-", "_", 1)
        tasks.append(task(f"{name}__pg19val_w2048", checkpoint, PG19_VALIDATION, 2048, batch=32))
        tasks.append(
            task(
                f"{name}__fineweb_w2048",
                checkpoint,
                FINEWEB_HELDOUT,
                2048,
                batch=32,
                extra=("--stream", "--max-windows", "2048"),
            )
        )
        if "ttt" in name:
            tasks.append(
                task(
                    f"{name}_noadapt__pg19val_w2048", checkpoint, PG19_VALIDATION, 2048, batch=32, extra=("--no-adapt",)
                )
            )
    return tasks


def dynamic_tasks(outputs: Path):
    """朴素测试时更新基线，以及改变测试时内循环学习率的敏感性。"""
    tasks = []
    subset = ("--windows-per-document", "5")
    control = outputs / "main-ck"
    if finished(control):
        checkpoint = latest_checkpoint(control)
        for label, rate in (("0p01", 0.01), ("0p03", 0.03), ("0p1", 0.1), ("0p3", 0.3), ("1", 1.0)):
            options = ("--force-adapt", "--inner-learning-rate", str(rate))
            tasks.append(
                task(f"dynamic_ck_lr{label}__pg19val", checkpoint, PG19_VALIDATION, 4096, extra=(*options, *subset))
            )
            tasks.append(task(f"dynamic_ck_lr{label}__pg19test_w4096", checkpoint, PG19_TEST, 4096, extra=options))
        tasks.append(task("dynamic_ck_off__pg19val", checkpoint, PG19_VALIDATION, 4096, extra=subset))
    meta = outputs / "main-ttt"
    if finished(meta):
        checkpoint = latest_checkpoint(meta)
        for label, rate in (("0p25", 0.25), ("0p5", 0.5), ("1", 1.0), ("2", 2.0), ("4", 4.0)):
            options = ("--inner-learning-rate", str(rate), *subset)
            tasks.append(task(f"dynamic_ttt_lr{label}__pg19val", checkpoint, PG19_VALIDATION, 4096, extra=options))
        tasks.append(task("dynamic_ttt_off__pg19val", checkpoint, PG19_VALIDATION, 4096, extra=("--no-adapt", *subset)))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="批量运行技术报告评测")
    stages = ("main", "long", "curves", "latency", "ablations", "dynamic")
    parser.add_argument("--stage", choices=stages, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--arms", nargs="+", default=list(MAIN_ARMS))
    parser.add_argument("--runs", nargs="+", help="ablations 阶段只评测这些运行目录")
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--evals", type=Path, default=Path("outputs/evals"))
    parser.add_argument("--batch-size", type=int, help="覆盖所有任务的批大小（与训练共用显卡时调小）")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    builders = {
        "main": lambda: main_tasks(args.outputs, args.arms),
        "long": lambda: long_tasks(args.outputs, args.arms),
        "curves": lambda: curve_tasks(args.outputs, args.arms),
        "latency": lambda: latency_tasks(args.outputs, args.arms),
        "ablations": lambda: ablation_tasks(args.outputs, args.runs),
        "dynamic": lambda: dynamic_tasks(args.outputs),
    }
    tasks = builders[args.stage]()
    args.evals.mkdir(parents=True, exist_ok=True)
    logs = Path("logs/evals")
    logs.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "JAX_PLATFORMS": "cuda",
        "PYTHONPATH": ".",
        # 评测可能与训练进程共用一张卡：按需申请显存，不预占。
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "false"),
    }
    for name, arguments in tasks:
        if args.batch_size is not None:
            arguments[arguments.index("--batch-size") + 1] = str(args.batch_size)
        output = args.evals / f"{name}.json"
        if output.exists():
            continue
        command = [sys.executable, "scripts/eval_lm_positions.py", *arguments, "--output", str(output)]
        if args.dry_run:
            print(" ".join(command))
            continue
        started = time.time()
        with (logs / f"{name}.log").open("w", encoding="utf-8") as log:
            code = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT).returncode
        print(json.dumps({"task": name, "returncode": code, "seconds": round(time.time() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
