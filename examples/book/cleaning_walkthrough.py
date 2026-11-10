"""清洗章的可运行小例子：保留原文，逐条记录变换、决策和计数。

仅使用标准库；演示文本规范化与规则记账，不提供 HTML 正文抽取或质量分类。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import unicodedata
from collections import Counter
from pathlib import Path


RULE_VERSION = "cleaning-walkthrough-v1"
RULE_ORDER = (
    "invalid_json",
    "invalid_schema",
    "invalid_unicode",
    "empty",
    "too_short",
    "too_long",
    "repeated_line",
)
DEFAULT_INPUT = Path(__file__).resolve().parent / "fixtures" / "cleaning_records.jsonl"


def normalize_text(text: str) -> str:
    """仅去掉开头一个 BOM、统一换行并作 NFC，不删除首尾或逐行空白。"""
    text = text.removeprefix("\ufeff")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


def schema_errors(record: object) -> list[str]:
    """教学 schema 的必需字段；完整生产 schema 需要单独设计。"""
    if not isinstance(record, dict):
        return ["record_must_be_object"]
    errors = []
    for field in ("id", "source", "language", "kind"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field}_must_be_nonempty_string")
    if not isinstance(record.get("text"), str):
        errors.append("text_must_be_string")
    if record.get("kind") not in ("prose", "code"):
        errors.append("kind_must_be_prose_or_code")
    return errors


def inspect_record(
    record: object,
    *,
    min_prose_chars: int,
    min_code_chars: int,
    max_chars: int,
    max_duplicate_line_fraction: float,
) -> dict:
    errors = schema_errors(record)
    if errors:
        return {
            "primary_reason": "invalid_schema",
            "diagnostics": ["invalid_schema"],
            "schema_errors": errors,
            "schema_valid": False,
        }

    # JSON 能表示孤立代理码点，但它们不能编码为有效 UTF-8 正文。
    try:
        normalized = normalize_text(record["text"])
        normalized.encode("utf-8")
    except UnicodeError:
        return {
            "primary_reason": "invalid_unicode",
            "diagnostics": ["invalid_unicode"],
            "schema_errors": [],
            "schema_valid": True,
        }

    # strip 只作用于计数/比较视图；保存给模型的 normalized 保持空白。
    count_view = normalized.strip()
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    content_chars = len(count_view)
    duplicate_fraction = 1 - len(set(lines)) / len(lines) if lines else 0.0
    minimum = min_code_chars if record["kind"] == "code" else min_prose_chars
    diagnostics = []
    if not count_view:
        diagnostics.append("empty")
    if content_chars < minimum:
        diagnostics.append("too_short")
    if content_chars > max_chars:
        diagnostics.append("too_long")
    # 代码中的重复行可能合法，因此该启发式只用于 prose。
    if (
        record["kind"] == "prose"
        and len(lines) >= 3
        and duplicate_fraction >= max_duplicate_line_fraction
    ):
        diagnostics.append("repeated_line")

    return {
        "primary_reason": diagnostics[0] if diagnostics else "kept",
        "diagnostics": diagnostics,
        "schema_errors": [],
        "schema_valid": True,
        "normalized_text": normalized,
        "text_changed": normalized != record["text"],
        "metrics": {
            "raw_codepoints": len(record["text"]),
            "normalized_codepoints": len(normalized),
            "content_chars": content_chars,
            "nonempty_lines": len(lines),
            "unique_comparison_lines": len(set(lines)),
            "duplicate_line_fraction": duplicate_fraction,
            "minimum_content_chars": minimum,
        },
    }


def unicode_examples() -> list[dict]:
    cases = (
        ("combining_accent", "Cafe\u0301"),
        ("fullwidth", "Ａ"),
        ("circled_digit", "①"),
    )
    result = []
    for name, text in cases:
        forms = {}
        for form, value in (
            ("original", text),
            ("NFC", unicodedata.normalize("NFC", text)),
            ("NFKC", unicodedata.normalize("NFKC", text)),
        ):
            forms[form] = {
                "text": value,
                "codepoints": [f"U+{ord(char):04X}" for char in value],
                "utf8_hex": value.encode("utf-8").hex(" "),
            }
        result.append({"case": name, "forms": forms})
    return result


def write_jsonl(path: Path, rows: list[dict]) -> None:
    # ASCII 转义保留 JSON 中原有的孤立代理码点诊断，不让拒绝日志写入失败。
    with path.open("x", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=True, sort_keys=True, allow_nan=False) + "\n")


def reject_json_constant(value: str) -> None:
    """拒绝 Python JSON 解析器默认允许、但 JSON 标准不允许的 NaN/Infinity。"""
    raise ValueError(f"nonstandard_json_constant: {value}")


def parse_finite_float(value: str) -> float:
    """拒绝溢出为无穷的 JSON 数字，避免保留的元数据变成非标准 JSON。"""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"json_number_outside_finite_float_range: {value}")
    return number


def run(args: argparse.Namespace) -> dict:
    if min(args.min_prose_chars, args.min_code_chars, args.max_chars) < 1:
        raise ValueError("长度阈值必须为正整数。")
    if max(args.min_prose_chars, args.min_code_chars) > args.max_chars:
        raise ValueError("最短长度不得超过最长长度。")
    if not 0 <= args.max_duplicate_line_fraction <= 1:
        raise ValueError("重复行比例阈值必须位于 [0, 1]。")
    if args.output.exists():
        raise FileExistsError("输出目录已存在；请为新清洗版本指定新目录。")

    raw_bytes = args.input.read_bytes()
    # 字节解码失败时整个输入无效；不先替换字符再伪装成合法文本。
    source_text = raw_bytes.decode("utf-8")
    kept, rejected, decisions = [], [], []
    primary_counts = Counter({reason: 0 for reason in ("kept",) + RULE_ORDER})
    diagnostic_counts = Counter({reason: 0 for reason in RULE_ORDER})
    groups = {}
    blank_lines = 0
    # JSONL 用 LF 分隔记录；正文里的 U+2028 等 Unicode 字符不是文件记录边界。
    input_lines = source_text.split("\n") if source_text else []
    if source_text.endswith("\n"):
        input_lines.pop()
    for line_number, raw_line in enumerate(input_lines, start=1):
        raw_line = raw_line.removesuffix("\r")
        if not raw_line.strip():
            blank_lines += 1
            continue
        try:
            record = json.loads(
                raw_line, parse_constant=reject_json_constant, parse_float=parse_finite_float
            )
        except ValueError as error:
            record = None
            decision = {
                "primary_reason": "invalid_json",
                "diagnostics": ["invalid_json"],
                "schema_errors": [],
                "schema_valid": False,
                "json_error": {
                    "column": getattr(error, "colno", None),
                    "message": getattr(error, "msg", str(error)),
                },
            }
        else:
            decision = inspect_record(
                record,
                min_prose_chars=args.min_prose_chars,
                min_code_chars=args.min_code_chars,
                max_chars=args.max_chars,
                max_duplicate_line_fraction=args.max_duplicate_line_fraction,
            )

        reason = decision["primary_reason"]
        primary_counts[reason] += 1
        diagnostic_counts.update(decision["diagnostics"])
        public_decision = {
            "line": line_number,
            "id": record.get("id") if isinstance(record, dict) else None,
            **{key: value for key, value in decision.items() if key != "normalized_text"},
        }
        decisions.append(public_decision)

        if decision["schema_valid"]:
            group_key = (record["source"], record["language"], record["kind"])
            group = groups.setdefault(group_key, {"input": 0, "kept": 0, "rejected": 0})
            group["input"] += 1
            group["kept" if reason == "kept" else "rejected"] += 1

        if reason == "kept":
            cleaned = dict(record)
            cleaned["original_text"] = record["text"]
            cleaned["text"] = decision["normalized_text"]
            cleaned["cleaning"] = {"rule_version": RULE_VERSION, "input_line": line_number}
            kept.append(cleaned)
        else:
            rejected.append(
                {
                    **public_decision,
                    "original_record": record,
                    "raw_line": raw_line,
                }
            )

    total = len(decisions)
    primary_rejected = sum(primary_counts[reason] for reason in RULE_ORDER)
    if total != len(kept) + primary_rejected or primary_rejected != len(rejected):
        raise RuntimeError("清洗主状态计数不守恒。")

    stages = [{"stage": "input_nonblank_lines", "removed": 0, "remaining": total}]
    remaining = total
    for reason in RULE_ORDER:
        removed = primary_counts[reason]
        remaining -= removed
        stages.append({"stage": reason, "removed": removed, "remaining": remaining})

    group_rows = []
    for (source, language, kind), counts in sorted(groups.items()):
        group_rows.append(
            {
                "source": source,
                "language": language,
                "kind": kind,
                **counts,
                "keep_rate": counts["kept"] / counts["input"],
            }
        )
    report = {
        "scope": "teaching_text_normalization_and_rule_accounting",
        "rule_version": RULE_VERSION,
        "input_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "unicode_version": unicodedata.unidata_version,
        "parameters": {
            "normalization": "NFC",
            "remove_initial_bom": True,
            "normalize_newlines": True,
            "preserve_training_whitespace": True,
            "min_prose_chars": args.min_prose_chars,
            "min_code_chars": args.min_code_chars,
            "max_chars": args.max_chars,
            "max_duplicate_line_fraction": args.max_duplicate_line_fraction,
            "duplicate_rule_kind": "prose",
            "duplicate_rule_minimum_lines": 3,
        },
        "rule_order": list(RULE_ORDER),
        "input_records": total,
        "ignored_blank_lines": blank_lines,
        "kept_records": len(kept),
        "rejected_records": len(rejected),
        "primary_counts": dict(primary_counts),
        "diagnostic_counts": dict(diagnostic_counts),
        "stages": stages,
        "group_scope": "schema_valid_records",
        "groups": group_rows,
        "unicode_examples": unicode_examples(),
        "decisions": decisions,
        "count_conservation": True,
    }

    # 先完成分析，再创建新目录；report.json 最后写，避免半成品被当作成功报告。
    args.output.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output / "kept.jsonl", kept)
    write_jsonl(args.output / "rejected.jsonl", rejected)
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="运行书中清洗与规则计数的小例子。")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-prose-chars", type=int, default=12)
    parser.add_argument("--min-code-chars", type=int, default=1)
    parser.add_argument("--max-chars", type=int, default=120)
    parser.add_argument("--max-duplicate-line-fraction", type=float, default=0.6)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "scope": report["scope"],
                "input_records": report["input_records"],
                "kept_records": report["kept_records"],
                "rejected_records": report["rejected_records"],
                "primary_counts": report["primary_counts"],
                "report": str(args.output / "report.json"),
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
