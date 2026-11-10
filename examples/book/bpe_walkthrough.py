"""逐轮记录字符级 BPE；仅依赖标准库，不代替正式 tokenizer。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

END = "</w>"
WORD_COUNTS = {"cat": 5, "cats": 2, "cap": 3, "caps": 1}
Pair = tuple[str, str]
Corpus = dict[str, tuple[str, ...]]


def count_pairs(corpus: Corpus, frequencies: dict[str, int]) -> Counter[Pair]:
    """相邻位置逐个计数，再乘该词在语料中的出现次数。"""
    counts: Counter[Pair] = Counter()
    for word, symbols in corpus.items():
        for pair in zip(symbols, symbols[1:]):
            counts[pair] += frequencies[word]
    return counts


def merge_pair(symbols: tuple[str, ...], pair: Pair) -> tuple[str, ...]:
    """从左到右合并互不重叠的位置。"""
    result: list[str] = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and symbols[index : index + 2] == pair:
            result.append(pair[0] + pair[1])
            index += 2
        else:
            result.append(symbols[index])
            index += 1
    return tuple(result)


def _positions(corpus: Corpus, frequencies: dict[str, int]) -> int:
    return sum(len(symbols) * frequencies[word] for word, symbols in corpus.items())


def learn_bpe(frequencies: dict[str, int], num_merges: int) -> dict:
    if num_merges < 0:
        raise ValueError("num_merges 不能为负数。")
    if not frequencies:
        raise ValueError("词频表不能为空。")
    for word, count in frequencies.items():
        if not word or any(char.isspace() or char in "<>" for char in word):
            raise ValueError("教学词项必须非空，不能含空白、< 或 >。")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("词频必须是正整数。")

    corpus = {word: (*word, END) for word in sorted(frequencies)}
    vocabulary = sorted({symbol for symbols in corpus.values() for symbol in symbols})
    initial_positions = _positions(corpus, frequencies)
    steps = []
    merges: list[Pair] = []
    for rank in range(num_merges):
        counts = count_pairs(corpus, frequencies)
        if not counts:
            break
        # 同频时按二元组的 Unicode 字典序选取，不依赖 dict 的插入顺序。
        chosen = min(counts, key=lambda pair: (-counts[pair], pair))
        before = corpus
        corpus = {word: merge_pair(symbols, chosen) for word, symbols in before.items()}
        merged_token = "".join(chosen)
        if merged_token not in vocabulary:
            vocabulary.append(merged_token)
        merges.append(chosen)
        steps.append(
            {
                "rank": rank,
                "round": rank + 1,
                "pair": chosen,
                "pair_count": counts[chosen],
                "pair_counts": [
                    {"pair": pair, "count": counts[pair]}
                    for pair in sorted(counts, key=lambda pair: (-counts[pair], pair))
                ],
                "before": before,
                "after": corpus,
                "positions_before": _positions(before, frequencies),
                "positions_after": _positions(corpus, frequencies),
            }
        )
    return {
        "word_counts": dict(sorted(frequencies.items())),
        "initial_positions": initial_positions,
        "final_positions": _positions(corpus, frequencies),
        "merges": merges,
        "vocabulary": vocabulary,
        "steps": steps,
    }


def encode_word(word: str, model: dict) -> dict:
    """只查询已经学到的 rank；编码阶段不计数、不更新词表。"""
    base_symbols = {char for training_word in model["word_counts"] for char in training_word}
    if not word or set(word) - base_symbols:
        raise ValueError("词含训练字母表之外的字符，或为空。")
    symbols = (*word, END)
    ranks = {tuple(pair): rank for rank, pair in enumerate(model["merges"])}
    trace = []
    while True:
        candidates = {pair for pair in zip(symbols, symbols[1:]) if pair in ranks}
        if not candidates:
            break
        pair = min(candidates, key=lambda candidate: ranks[candidate])
        symbols = merge_pair(symbols, pair)
        trace.append({"rank": ranks[pair], "pair": pair, "symbols": symbols})
    token_to_id = {token: index for index, token in enumerate(model["vocabulary"])}
    return {
        "word": word,
        "tokens": symbols,
        "ids": [token_to_id[symbol] for symbol in symbols],
        "trace": trace,
        "decoded": "".join(symbols).removesuffix(END),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="保存本次计算的完整 JSON。")
    parser.add_argument("--merges", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"输出文件已存在，请选择新的报告路径：{args.output}")
    model = learn_bpe(WORD_COUNTS, args.merges)
    report = {
        "schema_version": 1,
        "scope": "character BPE, isolated words, explicit end marker; no Unicode normalization",
        "tie_rule": "maximum weighted adjacent-pair count; then ascending Unicode pair tuple",
        "python_version": platform.python_version(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": model,
        "encodings": [encode_word(word, model) for word in ("cat", "cats", "cap", "caps", "catt", "catcap")],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 独占创建，避免检查后另一个进程写入同名报告时被覆盖。
    with args.output.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "merges": len(model["merges"]),
                      "positions": [model["initial_positions"], model["final_positions"]]}))


if __name__ == "__main__":
    main()
