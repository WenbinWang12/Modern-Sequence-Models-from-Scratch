"""生成 HumanEval completion，并可显式执行官方测试。

默认只生成代码，不执行。只有传入 ``--run-tests`` 才会运行题目测试；
生成代码属于不可信输入，正式实验应在隔离容器中运行此选项。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from ttt.checkpoint import load_checkpoint
from ttt.data import load_tokenizer
from ttt.models.ttt_e2e.modeling_ttt_e2e import TTTE2EForCausalLM
from ttt.runtime import TTTSession


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_test(program: str, *, timeout: float) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="ttt-humaneval-") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        environment = {"PYTHONHASHSEED": "0", "PATH": os.environ.get("PATH", "")}
        try:
            result = subprocess.run(
                [os.environ.get("PYTHON", "python3"), "-I", str(path)],
                cwd=directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if result.returncode == 0:
            return True, ""
        return False, (result.stderr or result.stdout)[-2000:]


def evaluate(
    checkpoint_path: Path,
    data_path: Path,
    *,
    limit: int | None,
    max_new_tokens: int,
    no_adapt: bool,
    run_tests: bool,
    timeout: float,
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
    records = []
    skipped = passed = 0
    for index, row in enumerate(rows):
        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            skipped += 1
            continue
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
            completion = tokenizer.decode(generated)
        except (ValueError, RuntimeError):
            skipped += 1
            continue
        passed_case = None
        error = ""
        if run_tests:
            test = row.get("test")
            entry_point = row.get("entry_point")
            if not isinstance(test, str) or not isinstance(entry_point, str):
                skipped += 1
                continue
            program = prompt + completion + "\n\n" + test + f"\n\ncheck({entry_point})\n"
            passed_case, error = _run_test(program, timeout=timeout)
            passed += int(passed_case)
        records.append({"index": index, "completion": completion, "passed": passed_case, "error": error})
    samples = len(records)
    metadata_path = checkpoint_path / "metadata.json" if checkpoint_path.is_dir() else checkpoint_path
    return {
        "protocol_version": "humaneval_generate_v1",
        "protocol": "HumanEval; greedy completion; pass@1 when --run-tests is enabled",
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata_sha256": _sha256(metadata_path) if metadata_path.is_file() else None,
        "data": str(data_path.resolve()),
        "data_sha256": _sha256(data_path),
        "tokenizer": experiment.tokenizer,
        "adaptation": not no_adapt,
        "max_new_tokens": max_new_tokens,
        "run_tests": run_tests,
        "timeout_seconds": timeout if run_tests else None,
        "limit": limit,
        "passed": passed if run_tests else None,
        "samples": samples,
        "skipped": skipped,
        "pass_at_1": passed / samples if run_tests and samples else None,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="评估 HumanEval 代码生成")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-adapt", action="store_true")
    parser.add_argument("--run-tests", action="store_true", help="在隔离临时目录运行测试；生成代码仍是不可信输入")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(
        args.checkpoint,
        args.data,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        no_adapt=args.no_adapt,
        run_tests=args.run_tests,
        timeout=args.timeout,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
