"""去重章的五文档教学实验；仅依赖 Python 标准库。"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import unicodedata
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Callable


FIXTURE = Path(__file__).with_name("dedup_documents.jsonl")
SHINGLE_LABELS = {
    ("red", "blue"): "a",
    ("blue", "green"): "b",
    ("green", "amber"): "c",
    ("amber", "gold"): "d",
    ("green", "violet"): "e",
    ("violet", "silver"): "f",
}
# 为手算明确选定的排列，不冒充独立均匀随机抽样。
PERMUTATIONS = (
    tuple("abcdef"),
    tuple("bcdaef"),
    tuple("dcbaef"),
    tuple("cdabef"),
)
THRESHOLD = Fraction(3, 4)


def comparison_text(text: str) -> str:
    """只统一 NFC 和 CRLF；原文和有意义的空白另行保留。"""
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def exact_deduplicate(
    records: list[dict[str, str]],
    digest: Callable[[str], str] = sha256_text,
) -> tuple[list[dict[str, str]], dict[str, str], dict[str, str]]:
    """按 ID 选择代表；同摘要之后仍比较完整的规范化文本。"""
    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Document IDs must be unique")
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    unique = []
    representatives = {}
    keys = {}
    for record in sorted(records, key=lambda item: item["id"]):
        text = comparison_text(record["text"])
        key = digest(text)
        keys[record["id"]] = key
        representative = next(
            (doc_id for prior, doc_id in buckets[key] if prior == text), None
        )
        if representative is None:
            representative = record["id"]
            buckets[key].append((text, representative))
            unique.append(record)
        representatives[record["id"]] = representative
    return unique, representatives, keys


def word_shingles(text: str, width: int = 2) -> set[tuple[str, ...]]:
    """相邻词组构成集合；此处的按空白切词不是模型 tokenizer。"""
    if width < 1:
        raise ValueError("Shingle width must be positive")
    words = comparison_text(text).split()
    return {tuple(words[index : index + width]) for index in range(len(words) - width + 1)}


def minhash_signature(
    elements: set[str], orders: tuple[tuple[str, ...], ...]
) -> tuple[str, ...]:
    """保存每个共享排列中最早出现的集合元素，而非该元素的排名。"""
    if not elements or not orders:
        raise ValueError("MinHash needs a nonempty set and at least one order")
    universe = set(orders[0])
    if not elements <= universe:
        raise ValueError("Every shingle must belong to the permutation universe")
    if any(len(order) != len(universe) or set(order) != universe for order in orders):
        raise ValueError("Each order must permute the same universe exactly once")
    return tuple(next(element for element in order if element in elements) for order in orders)


def lsh_candidates(
    signatures: dict[str, tuple[str, ...]], rows_per_band: int
) -> tuple[list[tuple[str, str]], list[dict[str, object]]]:
    """同一 band 的全部行一致才进入同一桶；对候选文档对去重。"""
    lengths = {len(signature) for signature in signatures.values()}
    if rows_per_band < 1 or len(lengths) != 1:
        raise ValueError("Signatures must have a shared length and positive band size")
    length = next(iter(lengths))
    if length == 0 or length % rows_per_band:
        raise ValueError("Signature length must be a positive multiple of band size")
    buckets: dict[tuple[int, tuple[str, ...]], list[str]] = defaultdict(list)
    for doc_id, signature in sorted(signatures.items()):
        for start in range(0, length, rows_per_band):
            buckets[(start // rows_per_band, signature[start : start + rows_per_band])].append(doc_id)
    candidates = set()
    bucket_report = []
    for (band, values), doc_ids in sorted(buckets.items()):
        candidates.update(itertools.combinations(doc_ids, 2))
        bucket_report.append({"band_index": band, "values": list(values), "document_ids": doc_ids})
    return sorted(candidates), bucket_report


def jaccard(left: set[str], right: set[str]) -> Fraction:
    union = left | right
    if not union:
        raise ValueError("Empty shingle sets need a separate policy")
    return Fraction(len(left & right), len(union))


def select_representatives(
    sets: dict[str, set[str]], candidates: list[tuple[str, str]], threshold: Fraction
) -> tuple[list[str], dict[str, str], list[dict[str, object]]]:
    """只允许直接匹配已保留代表的删除，不沿相似边传递删除。"""
    if not 0 < threshold <= 1:
        raise ValueError("Threshold must belong to (0, 1]")
    candidate_set = {tuple(sorted(pair)) for pair in candidates}
    kept = []
    representatives = {}
    decisions = []
    for doc_id in sorted(sets):
        matches = [
            representative for representative in kept
            if tuple(sorted((doc_id, representative))) in candidate_set
            and jaccard(sets[doc_id], sets[representative]) >= threshold
        ]
        representative = matches[0] if matches else doc_id
        representatives[doc_id] = representative
        if not matches:
            kept.append(doc_id)
        decisions.append({
            "document_id": doc_id,
            "representative_id": representative,
            "decision": "near_removed" if matches else "kept",
            "qualifying_retained_representatives": matches,
        })
    return kept, representatives, decisions


def build_report(records: list[dict[str, str]]) -> tuple[dict[str, object], list[dict[str, str]]]:
    unique, exact_representatives, keys = exact_deduplicate(records)
    sets = {}
    for record in unique:
        shingles = word_shingles(record["text"])
        if not shingles or not shingles <= SHINGLE_LABELS.keys():
            raise ValueError("This walkthrough only supports its nonempty six-shingle fixture")
        sets[record["id"]] = {SHINGLE_LABELS[shingle] for shingle in shingles}
    signatures = {doc_id: minhash_signature(elements, PERMUTATIONS) for doc_id, elements in sets.items()}
    candidates, buckets = lsh_candidates(signatures, rows_per_band=2)
    pair_reviews = []
    for left, right in candidates:
        similarity = jaccard(sets[left], sets[right])
        pair_reviews.append({
            "document_ids": [left, right],
            "intersection": sorted(sets[left] & sets[right]),
            "union": sorted(sets[left] | sets[right]),
            "jaccard_fraction": str(similarity),
            "jaccard": float(similarity),
            "at_or_above_threshold": similarity >= THRESHOLD,
        })
    kept_ids, near_representatives, decisions = select_representatives(sets, candidates, THRESHOLD)
    final_representatives = {
        doc_id: near_representatives[representative]
        for doc_id, representative in exact_representatives.items()
    }
    # 穷举 6! 个排列用于检查这组小集合；生产系统不会穷举排列。
    exhaustive = []
    for left, right in itertools.combinations(sorted(sets), 2):
        matches = 0
        total = 0
        for order in itertools.permutations(sorted(SHINGLE_LABELS.values())):
            matches += minhash_signature(sets[left], (order,)) == minhash_signature(sets[right], (order,))
            total += 1
        exhaustive.append({
            "document_ids": [left, right], "permutations": total, "matches": matches,
            "collision_fraction": str(Fraction(matches, total)),
            "jaccard_fraction": str(jaccard(sets[left], sets[right])),
        })
    counts = {
        "input_documents": len(records),
        "kept_documents": len(kept_ids),
        "exact_removed_documents": len(records) - len(unique),
        "near_removed_documents": len(unique) - len(kept_ids),
        "candidate_pairs_reviewed": len(candidates),
        "pairs_at_or_above_threshold": sum(review["at_or_above_threshold"] for review in pair_reviews),
    }
    report = {
        "rule_version": "dedup-teaching-v1",
        "python_version": platform.python_version(),
        "rules": {
            "exact": "NFC; CRLF to LF; SHA256 bucket followed by full text equality; ascending ID representative",
            "near": "whitespace split; word bigram set; fixed illustrative orders; 2 rows per band",
            "threshold_fraction": str(THRESHOLD),
            "selection": "ascending ID; delete only against an already-retained direct candidate representative",
            "scope": "five-document teaching fixture; no production deduplication integration",
        },
        "exact_keys": keys,
        "exact_representatives": exact_representatives,
        "shingle_labels": {label: list(shingle) for shingle, label in SHINGLE_LABELS.items()},
        "shingle_sets": {doc_id: sorted(elements) for doc_id, elements in sets.items()},
        "illustrative_permutations": [list(order) for order in PERMUTATIONS],
        "signatures": {doc_id: list(signature) for doc_id, signature in signatures.items()},
        "band_buckets": buckets,
        "candidate_pair_reviews": pair_reviews,
        "representative_decisions": decisions,
        "final_representatives": final_representatives,
        "exhaustive_permutation_checks": exhaustive,
        "counts": counts,
    }
    return report, [record for record in unique if record["id"] in kept_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are refused")
    args = parser.parse_args()
    raw = FIXTURE.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    report, kept = build_report(records)
    report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output / "deduped.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in kept), encoding="utf-8"
    )
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
