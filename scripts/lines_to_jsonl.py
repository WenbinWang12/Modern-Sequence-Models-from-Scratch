"""把按行保存的文本转换为可追踪的 JSONL 文档。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

__all__ = ["convert_lines", "main"]


def file_sha256(path: Path) -> str:
    """分块计算文件摘要，避免把大输入一次性读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    """原子写出 JSON 报告。"""

    temporary = path.with_name(path.name + ".part")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def convert_lines(
    input_path: Path,
    output_path: Path,
    *,
    limit: int | None = None,
    report_path: Path | None = None,
) -> dict[str, object]:
    """将非空文本行转换为 ``{"text": ...}`` JSONL。"""

    if limit is not None and limit <= 0:
        raise ValueError("limit 必须是正整数。")
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    resolved_report: Path | None = None
    if report_path is not None:
        resolved_report = report_path.expanduser().resolve()
        if resolved_report.exists():
            raise FileExistsError(f"报告文件已存在：{resolved_report}")
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")
    if output_path.exists():
        raise FileExistsError(f"输出文件已存在：{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_before = file_sha256(input_path)
    temporary = output_path.with_name(output_path.name + ".part")
    records = 0
    blank_lines = 0
    consumed_lines = 0
    try:
        with input_path.open("r", encoding="utf-8", newline="") as source, temporary.open(
            "w", encoding="utf-8", newline="\n"
        ) as target:
            for raw_line in source:
                consumed_lines += 1
                text = raw_line.rstrip("\r\n")
                if not text.strip():
                    blank_lines += 1
                    continue
                target.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                records += 1
                if limit is not None and records >= limit:
                    break
            target.flush()
            os.fsync(target.fileno())
        input_after = file_sha256(input_path)
        if input_before != input_after:
            raise RuntimeError("输入文件在转换期间发生变化；未发布 JSONL。")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)

    output_sha256 = file_sha256(output_path)
    report: dict[str, object] = {
        "input": str(input_path),
        "input_sha256": input_after,
        "output": str(output_path),
        "output_sha256": output_sha256,
        "records": records,
        "blank_lines": blank_lines,
        "consumed_lines": consumed_lines,
        "limit": limit,
        "selection": "all non-empty lines" if limit is None else "first non-empty lines up to limit",
    }
    if resolved_report is not None:
        resolved_report.parent.mkdir(parents=True, exist_ok=True)
        _write_json(resolved_report, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="把纯文本逐行转换成 JSONL 文档")
    parser.add_argument("--input", required=True, type=Path, help="UTF-8 文本输入")
    parser.add_argument("--output", required=True, type=Path, help="JSONL 输出")
    parser.add_argument("--limit", type=int, help="最多保留多少条非空记录")
    parser.add_argument("--report", type=Path, help="可选的 JSON 统计报告路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = convert_lines(
            args.input,
            args.output,
            limit=args.limit,
            report_path=args.report,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"错误：{error}") from error
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
