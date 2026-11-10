"""依次执行数据章实验，保存命令状态、测试计数和来源一致性证据。"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_digests() -> dict[str, str]:
    """记录输入、执行脚本与测试配置；运行后重新读取，检查是否发生变化。"""
    sources = sorted(
        list((ROOT / "examples" / "book").glob("*.py"))
        + list((ROOT / "examples" / "book").rglob("*.jsonl"))
        + [ROOT / "examples/sample.jsonl", ROOT / "scripts/train_tokenizer.py",
           ROOT / "tests/test_tokenizer_training.py", ROOT / "pyproject.toml", ROOT / "uv.lock"]
    )
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }


def read_pytest_summary(path: Path) -> dict[str, int]:
    """读取 pytest 的 JUnit 计数；缺失或格式不完整的报告不能算作通过。"""
    root = ET.parse(path).getroot()
    if root.tag not in {"testsuites", "testsuite"}:
        raise ValueError("Unexpected JUnit root element")
    suites = [suite for suite in root.iter("testsuite") if not suite.findall("testsuite")]
    if not suites:
        raise ValueError("JUnit report has no test suite")
    summary = {name: sum(int(suite.attrib[name]) for suite in suites)
               for name in ("tests", "failures", "errors", "skipped")}
    if any(value < 0 for value in summary.values()):
        raise ValueError("JUnit counts must not be negative")
    if summary["tests"] != len(list(root.iter("testcase"))):
        raise ValueError("JUnit test count disagrees with testcase elements")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="新建的验证输出目录。")
    parser.add_argument("--tokenizer", action="store_true", help="同时检查正式 tokenizer；需要 tokenizer/dev 依赖。")
    parser.add_argument("--lint", action="store_true", help="同时调用环境中的 Ruff。")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    junit_path = output / "tokenizer-tests.junit.xml"
    commands = [
        ("data-boundaries", ["examples/book/data_checks.py"]),
        ("dedup-boundaries", ["examples/book/dedup_checks.py"]),
        ("acquisition", ["examples/book/acquisition_walkthrough.py",
                         "--input", "examples/sample.jsonl", "--output", str(output / "acquisition")]),
        ("cleaning", ["examples/book/cleaning_walkthrough.py", "--output", str(output / "cleaning")]),
        ("deduplication", ["examples/book/dedup_walkthrough.py", "--output", str(output / "deduplication")]),
        ("splits", ["examples/book/split_walkthrough.py", "--seed", "17", "--output", str(output / "splits")]),
        ("bpe", ["examples/book/bpe_walkthrough.py", "--output", str(output / "bpe.json")]),
    ]
    if args.tokenizer:
        # 不加载模型 conftest；该文件仅使用 pytest 内置 fixture。
        commands.extend([
            ("tokenizer-dependencies", ["-c", "import pytest, tokenizers, transformers; "
                                        "print(pytest.__version__, tokenizers.__version__, transformers.__version__)"]),
            ("tokenizer-tests", ["-m", "pytest", "--noconftest", "-c", "pyproject.toml", "-q",
                                 "--junitxml", str(junit_path), "tests/test_tokenizer_training.py"]),
        ])
    if args.lint:
        commands.append(("lint", ["-m", "ruff", "check", "examples/book",
                                  "scripts/train_tokenizer.py", "tests/test_tokenizer_training.py"]))
    report = {
        "scope": "data_chapter_examples_and_optional_tokenizer_checks",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "started_at_utc": now(),
        "status": "running",
        "source_sha256": None,
        "source_snapshot_unchanged": None,
        "requested_tokenizer_checks": args.tokenizer,
        "requested_lint": args.lint,
        "pytest_policy": {
            "load_conftest": False,
            "ignore_external_pytest_addopts": True,
            "require_nonzero_tests_and_zero_skips_failures_errors": True,
        } if args.tokenizer else None,
        "planned_commands": [{"name": name, "arguments": arguments} for name, arguments in commands],
        "commands": [],
        "validation_errors": [],
        "all_requested_commands_passed": False,
    }
    report_path = output / "validation.json"

    def save() -> None:
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    def fail(message: str) -> None:
        report["validation_errors"].append(message)
        report["validation_error"] = "\n".join(report["validation_errors"])

    save()
    exit_code = 0
    interrupted = False
    entry = None
    try:
        report["source_sha256"] = source_digests()
        save()
        environment = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONUTF8="1")
        for name, arguments in commands:
            entry = {
                "name": name, "arguments": arguments, "returncode": None,
                "status": "running", "started_at_utc": now(),
            }
            report["commands"].append(entry)
            save()
            command_environment = dict(environment)
            if name == "tokenizer-tests":
                # 外部 -k / --collect-only 会悄悄缩小检查范围，故本入口不用它。
                command_environment.pop("PYTEST_ADDOPTS", None)
            result = subprocess.run(
                [sys.executable, *arguments], cwd=ROOT, env=command_environment,
                capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
            )
            # 保留真实退出码，即使随后因 JUnit 审核未通过而令整个运行器失败。
            entry["returncode"] = result.returncode
            stdout = output / f"{name}.stdout.txt"
            stderr = output / f"{name}.stderr.txt"
            stdout.write_text(result.stdout, encoding="utf-8")
            stderr.write_text(result.stderr, encoding="utf-8")
            entry.update(stdout=stdout.name, stderr=stderr.name, finished_at_utc=now())
            validation_error = None
            if name == "tokenizer-tests":
                entry["junit_xml"] = junit_path.name
                try:
                    summary = read_pytest_summary(junit_path)
                    entry["pytest_counts"] = summary
                    if summary["tests"] <= 0 or any(summary[key] for key in ("failures", "errors", "skipped")):
                        validation_error = f"Tokenizer JUnit counts do not satisfy the required policy: {summary}"
                except (OSError, ET.ParseError, KeyError, ValueError) as error:
                    validation_error = f"Tokenizer JUnit report is missing or invalid: {error}"
            if result.returncode or validation_error:
                entry["status"] = "failed"
                if validation_error:
                    entry["validation_error"] = validation_error
                    fail(validation_error)
                if result.returncode:
                    fail(f"{name} exited with code {result.returncode}")
                exit_code = result.returncode if result.returncode > 0 else 1
                save()
                print(f"{name}: exit={result.returncode}; validation failed", flush=True)
                break
            entry["status"] = "passed"
            save()
            print(f"{name}: exit={result.returncode}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        exit_code = 130
        if entry is not None and entry["status"] == "running":
            entry.update(status="interrupted", finished_at_utc=now())
        fail("Runner interrupted before all requested commands completed")
    except Exception as error:
        exit_code = 1
        if entry is not None and entry["status"] == "running":
            entry.update(status="error", finished_at_utc=now(), execution_error=f"{type(error).__name__}: {error}")
        fail(f"Runner error: {type(error).__name__}: {error}")

    if report["source_sha256"] is not None:
        try:
            final_digests = source_digests()
            report["source_sha256_after"] = final_digests
            before = report["source_sha256"]
            changed = sorted(path for path in set(before) | set(final_digests)
                             if before.get(path) != final_digests.get(path))
            report["changed_sources"] = changed
            report["source_snapshot_unchanged"] = not changed
            if changed:
                fail("Source or input files changed during validation: " + ", ".join(changed))
                exit_code = exit_code or 1
        except OSError as error:
            report["source_snapshot_unchanged"] = False
            fail(f"Could not verify final source digests: {error}")
            exit_code = exit_code or 1
    report["status"] = "interrupted" if interrupted else ("failed" if exit_code else "completed")
    report["all_requested_commands_passed"] = report["status"] == "completed"
    report["finished_at_utc"] = now()
    save()
    print(f"report: {report_path}")
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
