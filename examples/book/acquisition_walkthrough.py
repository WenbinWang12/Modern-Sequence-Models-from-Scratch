"""用本地教学语料演示来源记录、字节摘要与解析结果的区别。"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def read_records(data: bytes) -> list[dict]:
    """跳过空行，严格读取本例要求的 JSONL 文档。"""
    records = []
    seen = set()
    for number, line in enumerate(data.decode("utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            raise ValueError(f"第 {number} 行必须为含字符串 text 的对象。")
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError(f"第 {number} 行缺少唯一非空字符串 id。")
        seen.add(identifier)
        records.append(record)
    if not records:
        raise ValueError("输入没有可读取的记录。")
    return records


def file_record(path: Path, data: bytes) -> dict:
    return {"file": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def run(source: Path, output: Path) -> dict:
    source = source.resolve(strict=True)
    raw = source.read_bytes()
    records = read_records(raw)
    # 只追加一个 LF；原文件没有行末换行时，它只是补齐最后一行。
    changed = raw + b"\n"

    repo = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        commit = revision.stdout.strip() if revision.returncode == 0 else None
    except FileNotFoundError:
        commit = None
    output.mkdir(parents=True, exist_ok=False)
    original = output / "raw.jsonl"
    variant = output / "with-extra-newline.jsonl"
    original.write_bytes(raw)
    variant.write_bytes(changed)
    # 摘要计算针对实际落盘对象，不能只记录内存中的预期内容。
    original_bytes = original.read_bytes()
    variant_bytes = variant.read_bytes()
    original_info = file_record(original, original_bytes)
    variant_info = file_record(variant, variant_bytes)
    same_records = read_records(original_bytes) == read_records(variant_bytes)
    original_preserved = original_bytes == raw and source.read_bytes() == raw
    different_digest = original_info["sha256"] != variant_info["sha256"]
    if not original_preserved or variant_bytes != changed or not same_records or not different_digest:
        raise AssertionError("落盘文件未保留预期的原件、换行对照或解析结果。")
    report = {
        "schema_version": 1,
        "scope": "local_teaching_source_and_byte_identity",
        "source": {
            "source_id": "local-sha256:" + hashlib.sha256(raw).hexdigest(),
            "acquisition_method": "local_copy",
            "input_name": source.name,
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_commit": commit,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "terms_review": "not_performed_by_this_script",
        },
        "original": original_info,
        "variant": variant_info,
        "records": len(records),
        "record_ids": [record["id"] for record in records],
        "byte_counts_differ_by": variant_info["bytes"] - original_info["bytes"],
        "same_parsed_records": same_records,
        "different_byte_digest": different_digest,
        "original_preserved": original_preserved,
    }
    (output / "source-record.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("examples/sample.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
