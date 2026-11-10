"""演示按组切分、样本顺序稳定性与两类跨集合泄漏检查。"""

import argparse
import hashlib
import json
import math
import unicodedata
from collections import defaultdict
from pathlib import Path

POLICY = "book-group-split-v1"
SPLITS = ("train", "validation", "test")


def bucket_for(group: str, seed: int) -> int:
    # JSON 序列化明确各字段边界；不使用进程随机化的内置 hash。
    payload = json.dumps([POLICY, seed, group], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 10_000


def choose_split(group: str, seed: int) -> str:
    bucket = bucket_for(group, seed)
    if bucket < 6000:
        return "train"
    return "validation" if bucket < 8000 else "test"


def reject_json_constant(value: str) -> None:
    """拒绝 JSON 标准不支持的 NaN 和 Infinity。"""
    raise ValueError(f"nonstandard_json_constant: {value}")


def unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    """不让重复的 group、text 或嵌套元数据字段被解析器静默覆盖。"""
    record = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"duplicate_json_key: {key}")
        record[key] = value
    return record


def parse_finite_float(value: str) -> float:
    """本例仅接收可表示为有限浮点数的小数，避免元数据溢出后写出 Infinity。"""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"json_number_outside_finite_float_range: {value}")
    return number


def load_records(data: bytes) -> list[dict]:
    records = []
    identifiers = set()
    for number, line in enumerate(data.decode("utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        record = json.loads(
            line, object_pairs_hook=unique_json_object,
            parse_constant=reject_json_constant, parse_float=parse_finite_float,
        )
        if not isinstance(record, dict):
            raise ValueError(f"第 {number} 行不是对象。")
        for name in ("id", "split_group", "source", "text"):
            if not isinstance(record.get(name), str) or not record[name]:
                raise ValueError(f"第 {number} 行的 {name} 必须为非空字符串。")
        if record["id"] in identifiers:
            raise ValueError(f"重复文档 id：{record['id']}")
        identifiers.add(record["id"])
        records.append(record)
    if not records:
        raise ValueError("没有可切分的文档。")
    return records


def assignments_for(records: list[dict], seed: int) -> dict[str, str]:
    return {record["id"]: choose_split(record["split_group"], seed) for record in records}


def validate(records: list[dict], assignments: dict[str, str]) -> None:
    if set(assignments) != {record["id"] for record in records}:
        raise ValueError("assignment_ids: 分配表与文档 ID 不一致。")
    group_splits = defaultdict(set)
    text_splits = defaultdict(set)
    for record in records:
        split = assignments[record["id"]]
        if split not in SPLITS:
            raise ValueError("split_name: 未知集合。")
        group_splits[record["split_group"]].add(split)
        normalized = unicodedata.normalize("NFC", record["text"])
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        text_splits[digest].add(split)
    leaked_groups = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaked_groups:
        raise ValueError("group_leak: " + ", ".join(leaked_groups))
    if any(len(splits) > 1 for splits in text_splits.values()):
        raise ValueError("normalized_content_leak: NFC 全文摘要横跨集合。")


def expect_rejection(kind: str, records: list[dict], assignments: dict[str, str]) -> dict:
    try:
        validate(records, assignments)
    except ValueError as error:
        if not str(error).startswith(kind + ":"):
            raise AssertionError("失败探针触发了错误的检查分支。") from error
        return {"expected": kind, "rejected": True, "message": str(error)}
    raise AssertionError("故意制造的泄漏没有被拒绝。")


def run(source: Path, output: Path, seed: int) -> dict:
    # 解析和摘要引用同一份输入快照，避免处理中源文件变化导致身份错配。
    input_bytes = source.read_bytes()
    records = load_records(input_bytes)
    assignments = assignments_for(records, seed)
    validate(records, assignments)
    if assignments_for(list(reversed(records)), seed) != assignments:
        raise AssertionError("改变输入遍历顺序后，原有组归属改变了。")
    suffix = 0
    existing_groups = {record["split_group"] for record in records}
    while f"new-document-{suffix}" in assignments or f"new-group-{suffix}" in existing_groups:
        suffix += 1
    added = {
        "id": f"new-document-{suffix}", "split_group": f"new-group-{suffix}",
        "source": "demo", "text": "New document.",
    }
    extended = assignments_for(records + [added], seed)
    if any(extended[identifier] != split for identifier, split in assignments.items()):
        raise AssertionError("增加一个新组改变了既有分配。")

    groups = defaultdict(list)
    for record in records:
        groups[record["split_group"]].append(record)
    repeated = next((rows for rows in groups.values() if len(rows) > 1), None)
    if repeated is None or len(set(assignments.values())) < 2:
        raise ValueError("失败演示需要一个多文档组以及至少两个非空集合。")
    # 故意把同组的一条文档移入另一集合，验证组边界检查。
    bad_assignments = dict(assignments)
    original_split = assignments[repeated[0]["id"]]
    bad_assignments[repeated[0]["id"]] = next(split for split in SPLITS if split != original_split)
    group_probe = expect_rejection("group_leak", records, bad_assignments)
    # 分组不变，但把另一集合的一条正文替换为相同文本，验证内容检查。
    first = records[0]
    target = next(record for record in records if assignments[record["id"]] != assignments[first["id"]])
    bad_records = [
        dict(record, text=first["text"]) if record["id"] == target["id"] else dict(record)
        for record in records
    ]
    text_probe = expect_rejection("normalized_content_leak", bad_records, assignments)

    output.mkdir(parents=True, exist_ok=False)
    totals = {}
    for split in SPLITS:
        subset = sorted((r for r in records if assignments[r["id"]] == split), key=lambda r: r["id"])
        text = "".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in subset)
        path = output / f"{split}.jsonl"
        path.write_text(text, encoding="utf-8")
        totals[split] = {
            "documents": len(subset),
            "groups": len({record["split_group"] for record in subset}),
            "utf8_content_bytes": sum(len(record["text"].encode("utf-8")) for record in subset),
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    report = {
        "scope": "group_split_and_nfc_exact_overlap_only",
        "policy": POLICY,
        "seed": seed,
        "bucket_count": 10_000,
        "thresholds": {"train_end": 6000, "validation_end": 8000, "test_end": 10000},
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "groups": [
            {
                "id": group,
                "bucket": bucket_for(group, seed),
                "split": choose_split(group, seed),
                "documents": len(groups[group]),
                "utf8_content_bytes": sum(len(record["text"].encode("utf-8")) for record in groups[group]),
            }
            for group in sorted(groups)
        ],
        "totals": totals,
        "order_independent": True,
        "existing_assignments_stable_after_append": True,
        "failure_probes": [group_probe, text_probe],
        "limitations": ["No near-duplicate or semantic contamination detector.", "No model quality evaluation."],
    }
    (output / "split-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "fixtures" / "split_records.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.seed), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
