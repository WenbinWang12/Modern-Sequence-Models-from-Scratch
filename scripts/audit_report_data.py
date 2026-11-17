#!/usr/bin/env python3
"""训练数据审计：实际采样配比、语料覆盖率和评测污染检查。

三个子命令，全部只读：

- ``sampler``：按训练配置重放混合采样器的随机数（不读 token），统计每个来源实际被抽到的
  窗口数、被覆盖的语料比例和重复采样程度；
- ``duplicates``：held-out 文档与训练文档的规范化全文哈希比对（精确重复）；
- ``batches``：按各个实验配置实例化采样器，核对它们是否真的读到同一批训练数据；
- ``benchmarks``：把每道 benchmark 题目的规范化题干片段放进 Aho-Corasick 自动机，
  扫描全部训练语料原文，报告出现精确匹配的题目数。

规范化 = 小写并把连续空白压成一个空格。精确匹配检不出改写后的近似重复，这一点在报告中
如实说明。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import re
from pathlib import Path

import numpy as np

_SPACE = re.compile(r"\s+")
_AUTOMATON = None


def normalise(text: str) -> str:
    return _SPACE.sub(" ", text.lower()).strip()


def digest(text: str) -> bytes:
    return hashlib.blake2b(normalise(text).encode("utf-8"), digest_size=12).digest()


# --------------------------------------------------------------------------- sampler


def audit_sampler(args) -> dict:
    recipe_path = Path(args.recipe).resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    experiment = json.loads(Path(args.config).read_text(encoding="utf-8"))
    training = experiment["training"]
    length, batch, steps, seed = (
        training["sequence_length"],
        training["batch_size"],
        training["steps"],
        training["seed"],
    )
    names, sizes, probabilities = [], [], []
    for source in recipe["sources"]:
        manifest = json.loads((recipe_path.parent / source["data"]).resolve().read_text(encoding="utf-8"))
        names.append(source["name"])
        sizes.append(sum(record["count"] for record in manifest["shards"]))
        probabilities.append(float(source["probability"]))
    probabilities = np.asarray(probabilities, dtype=np.float64) / sum(probabilities)
    # 与 MixtureBatchStream._sample_one 相同的随机数调用顺序：先选来源，再选窗口起点。
    rng = np.random.default_rng(seed)
    block = 1024
    coverage = [np.zeros(size // block + 1, dtype=np.int32) for size in sizes]
    counts = [0] * len(names)
    for _ in range(steps * batch):
        index = int(rng.choice(len(names), p=probabilities))
        start = int(rng.integers(0, sizes[index] - length))
        counts[index] += 1
        coverage[index][start // block : (start + length) // block + 1] += 1
    total = sum(counts)
    sources = {}
    for name, size, count, blocks in zip(names, sizes, counts, coverage):
        sources[name] = {
            "corpus_tokens": int(size),
            "windows": int(count),
            "window_fraction": count / total,
            "sampled_tokens": int(count * length),
            "sampled_tokens_over_corpus": count * length / size,
            "corpus_fraction_seen": float((blocks > 0).mean()),
            "seen_blocks_sampled_more_than_once": float((blocks > 1).sum() / max((blocks > 0).sum(), 1)),
        }
    return {
        "recipe": str(recipe_path),
        "recipe_sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
        "config": args.config,
        "seed": seed,
        "steps": steps,
        "batch_size": batch,
        "sequence_length": length,
        "windows": total,
        "target_probabilities": dict(zip(names, probabilities.tolist())),
        "coverage_block_tokens": block,
        "sources": sources,
    }


# --------------------------------------------------------------------------- duplicates


def audit_batches(args) -> dict:
    """核对若干实验配置是否真的读到同一批训练数据：对前若干个 batch 的 token 和掩码取摘要。"""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ttt.config import load_experiment
    from ttt.data import BatchStream

    runs = {}
    for path in args.configs:
        experiment = load_experiment(path)
        training = experiment.training
        stream = BatchStream(
            args.recipe,
            sequence_length=training.sequence_length,
            batch_size=training.batch_size,
            vocab_size=experiment.model.vocab_size,
            seed=training.seed,
        )
        digest = hashlib.sha256()
        for _ in range(args.batches):
            tokens, masks = stream.next_batch_with_masks()
            digest.update(tokens.tobytes())
            digest.update(masks.tobytes())
        runs[Path(path).stem] = {
            "batches_sha256": digest.hexdigest(),
            "seed": training.seed,
            "steps": training.steps,
            "batch_size": training.batch_size,
            "sequence_length": training.sequence_length,
            "learning_rate": training.learning_rate,
            "data_fingerprint": stream.fingerprint,
            "layer_types": sorted(set(experiment.model.layer_types)),
            "use_fast_mlp": experiment.model.use_fast_mlp,
            "ttt_suffix_length": experiment.model.ttt_suffix_length,
            "ttt_chunk_size": experiment.model.ttt_chunk_size,
            "inner_enabled": experiment.inner.enabled,
        }
    identical = len({entry["batches_sha256"] for entry in runs.values()}) == 1
    budget = {
        key: len({entry[key] for entry in runs.values()}) == 1
        for key in ("seed", "steps", "batch_size", "sequence_length", "learning_rate", "data_fingerprint")
    }
    return {
        "recipe": str(Path(args.recipe).resolve()),
        "batches_compared": args.batches,
        "identical_batches": identical,
        "identical_budget": budget,
        "runs": runs,
    }


def _parquet_digests(task):
    import pyarrow.parquet as pq

    path, row_group, limit = task
    texts = pq.ParquetFile(path).read_row_group(row_group, columns=["text"]).column("text").to_pylist()
    if limit is not None:
        texts = texts[:limit]
    return [digest(text) for text in texts if isinstance(text, str) and text.strip()]


def _text_digests(paths):
    return [(Path(path).name, digest(Path(path).read_text(encoding="utf-8", errors="replace"))) for path in paths]


def parquet_tasks(paths, max_documents=None):
    import pyarrow.parquet as pq

    tasks, remaining = [], max_documents
    for path in paths:
        metadata = pq.ParquetFile(path).metadata
        for row_group in range(metadata.num_row_groups):
            rows = metadata.row_group(row_group).num_rows
            limit = None
            if remaining is not None:
                if remaining <= 0:
                    return tasks
                limit = min(rows, remaining)
                remaining -= limit
            tasks.append((str(path), row_group, limit))
    return tasks


def audit_duplicates(args) -> dict:
    result = {}
    with multiprocessing.get_context("spawn").Pool(args.processes) as pool:
        train = set()
        for digests in pool.imap_unordered(_parquet_digests, parquet_tasks(args.fineweb_train), chunksize=1):
            train.update(digests)
        heldout = []
        for digests in pool.imap(_parquet_digests, parquet_tasks(args.fineweb_heldout, args.heldout_documents)):
            heldout.extend(digests)
        duplicated = sorted({item.hex() for item in heldout if item in train})
        result["fineweb_edu"] = {
            "train_documents": len(train),
            "heldout_documents": len(heldout),
            "heldout_exact_duplicates_of_train": sum(item in train for item in heldout),
            "heldout_internal_duplicates": len(heldout) - len(set(heldout)),
            # 供 tokenize_corpus.py --exclude-digests 使用：从 held-out 中剔除这些文档。
            "heldout_duplicate_digests": duplicated,
        }
        root = Path(args.pg19)
        splits = {}
        for split in ("train", "validation", "test"):
            files = sorted(str(path) for path in (root / split).glob("*.txt"))
            batches = [files[index : index + 64] for index in range(0, len(files), 64)]
            splits[split] = dict(pair for pairs in pool.imap(_text_digests, batches) for pair in pairs)
        train_digests = set(splits["train"].values())
        result["pg19"] = {
            "train_books": len(splits["train"]),
            **{
                f"{split}_exact_duplicates_of_train": sorted(
                    name for name, value in splits[split].items() if value in train_digests
                )
                for split in ("validation", "test")
            },
        }
    metadata = root / "metadata.csv"
    if metadata.exists():
        import csv

        titles = {}
        with metadata.open(encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.reader(handle):
                if len(row) >= 2:
                    titles[row[0]] = normalise(row[1])
        train_titles = {titles.get(Path(name).stem) for name in splits["train"]} - {None}
        for split in ("validation", "test"):
            shared = sorted(Path(name).stem for name in splits[split] if titles.get(Path(name).stem) in train_titles)
            result["pg19"][f"{split}_books_with_title_also_in_train"] = shared
        result["pg19"]["note"] = (
            "same normalised title + author string in metadata.csv; indicates possible other editions"
        )
    return result


# --------------------------------------------------------------------------- benchmarks


def benchmark_probes(directory: Path, minimum: int, length: int) -> dict[str, list[tuple[int, str]]]:
    def rows(name):
        text = (directory / f"{name}.jsonl").read_text(encoding="utf-8")
        return [json.loads(line) for line in text.split("\n") if line.strip()]

    fields = {
        "hellaswag": lambda row: row["ctx"],
        "piqa": lambda row: row["goal"],
        "arc_easy": lambda row: row["question"],
        "arc_challenge": lambda row: row["question"],
        "winogrande": lambda row: row["sentence"].replace("_", row["option1"]),
        "lambada_openai": lambda row: row["text"][-400:],
        "sciq": lambda row: row["question"],
        "openbookqa": lambda row: row["question_stem"],
        "mmlu": lambda row: row["question"],
        "wikitext": lambda row: row["page"][200:1200],
    }
    probes = {}
    for name, extract in fields.items():
        items = []
        for index, row in enumerate(rows(name)):
            text = normalise(extract(row))
            if len(text) >= minimum:
                items.append((index, text[:length]))
        probes[name] = items
    return probes


def _initialise_automaton(patterns):
    global _AUTOMATON
    import ahocorasick

    _AUTOMATON = ahocorasick.Automaton()
    for key, pattern in patterns:
        _AUTOMATON.add_word(pattern, key)
    _AUTOMATON.make_automaton()


def _scan(texts):
    hits = set()
    for text in texts:
        for _, key in _AUTOMATON.iter(normalise(text)):
            hits.add(key)
    return hits


def _scan_parquet(task):
    import pyarrow.parquet as pq

    path, row_group, _ = task
    return _scan(pq.ParquetFile(path).read_row_group(row_group, columns=["text"]).column("text").to_pylist())


def _scan_files(paths):
    return _scan(Path(path).read_text(encoding="utf-8", errors="replace") for path in paths)


def audit_benchmarks(args) -> dict:
    probes = benchmark_probes(Path(args.benchmarks), args.minimum_characters, args.probe_characters)
    # 相同的题干片段只入库一次，命中时归到所有同文的题目上。
    by_pattern = {}
    for name, items in probes.items():
        for index, pattern in items:
            by_pattern.setdefault(pattern, []).append((name, index))
    patterns = [(key, pattern) for key, pattern in enumerate(by_pattern)]
    owners = [by_pattern[pattern] for _, pattern in patterns]
    files = sorted(str(path) for path in Path(args.pg19_train).glob("*.txt"))
    hits = set()
    with multiprocessing.get_context("spawn").Pool(
        args.processes, initializer=_initialise_automaton, initargs=(patterns,)
    ) as pool:
        for found in pool.imap_unordered(_scan_parquet, parquet_tasks(args.fineweb_train), chunksize=1):
            hits |= found
        batches = [files[index : index + 32] for index in range(0, len(files), 32)]
        for found in pool.imap_unordered(_scan_files, batches, chunksize=1):
            hits |= found
    matched = {}
    for key in hits:
        for name, index in owners[key]:
            matched.setdefault(name, set()).add(index)
    return {
        "method": "Aho-Corasick exact match of normalised probe strings against normalised training text",
        "probe_characters": args.probe_characters,
        "minimum_characters": args.minimum_characters,
        "corpora": {"fineweb_edu_train": [str(path) for path in args.fineweb_train], "pg19_train_files": len(files)},
        "tasks": {
            name: {
                "items_probed": len(items),
                "items_with_exact_match": len(matched.get(name, ())),
                "matched_indices": sorted(matched.get(name, ()))[:200],
            }
            for name, items in probes.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="训练数据审计")
    commands = parser.add_subparsers(dest="command", required=True)
    sampler = commands.add_parser("sampler")
    sampler.add_argument("--recipe", required=True)
    sampler.add_argument("--config", required=True)
    batches = commands.add_parser("batches")
    batches.add_argument("--recipe", required=True)
    batches.add_argument("--configs", nargs="+", required=True)
    batches.add_argument("--batches", type=int, default=20)
    duplicates = commands.add_parser("duplicates")
    benchmarks = commands.add_parser("benchmarks")
    for command in (duplicates, benchmarks):
        command.add_argument("--fineweb-train", nargs="+", required=True)
        command.add_argument("--processes", type=int, default=64)
    duplicates.add_argument("--fineweb-heldout", nargs="+", required=True)
    duplicates.add_argument("--heldout-documents", type=int, default=20000)
    duplicates.add_argument("--pg19", required=True)
    benchmarks.add_argument("--pg19-train", required=True)
    benchmarks.add_argument("--benchmarks", default="data/benchmarks")
    benchmarks.add_argument("--probe-characters", type=int, default=80)
    benchmarks.add_argument("--minimum-characters", type=int, default=50)
    for command in (sampler, batches, duplicates, benchmarks):
        command.add_argument("--output", required=True)
    args = parser.parse_args()
    audits = {
        "sampler": audit_sampler,
        "batches": audit_batches,
        "duplicates": audit_duplicates,
        "benchmarks": audit_benchmarks,
    }
    result = audits[args.command](args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
