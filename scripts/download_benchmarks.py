#!/usr/bin/env python3
"""下载公开 benchmark 的评测划分并转成 JSONL，同时记录来源 revision 和 SHA256。

只读取 Hugging Face datasets-server 导出的 parquet 数据文件，不执行数据集脚本。
这些文件只用于评测，不进入任何训练或 tokenizer 语料。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

# 任务名 -> (数据集, config 或 None 表示全部 config, split)
TASKS = {
    "hellaswag": ("Rowan/hellaswag", "default", "validation"),
    "piqa": ("baber/piqa", "default", "validation"),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation"),
    "blimp": ("nyu-mll/blimp", None, "train"),
    "lambada_openai": ("EleutherAI/lambada_openai", "default", "test"),
    "mmlu": ("cais/mmlu", "all", "test"),
    "sciq": ("allenai/sciq", "default", "test"),
    "openbookqa": ("allenai/openbookqa", "main", "test"),
    "wikitext": ("EleutherAI/wikitext_document_level", "wikitext-103-raw-v1", "test"),
}


def fetch(url: str, *, retries: int = 6) -> bytes:
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "benchmark-downloader"})
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 benchmark 评测划分")
    parser.add_argument("--output", default="data/benchmarks")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    args = parser.parse_args()
    import pyarrow.parquet as pq

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    listings, revisions = {}, {}
    for task in args.tasks:
        dataset, config, split = TASKS[task]
        if dataset not in listings:
            quoted = urllib.parse.quote(dataset, safe="")
            listings[dataset] = json.loads(fetch(f"https://datasets-server.huggingface.co/parquet?dataset={quoted}"))[
                "parquet_files"
            ]
            revisions[dataset] = json.loads(fetch(f"https://huggingface.co/api/datasets/{dataset}")).get("sha")
        files = [
            item
            for item in listings[dataset]
            if item["split"] == split and (config is None or item["config"] == config)
        ]
        if not files:
            raise ValueError(f"{task}: 没有找到 {dataset} {config} {split} 的 parquet 文件。")
        files.sort(key=lambda item: (item["config"], item["filename"]))
        rows = []
        for item in files:
            table = pq.read_table(io.BytesIO(fetch(item["url"])))
            for row in table.to_pylist():
                if config is None:
                    row = {"config": item["config"], **row}
                rows.append(row)
        path = output / f"{task}.jsonl"
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
        path.write_bytes(payload)
        manifest[task] = {
            "dataset": dataset,
            "config": config if config is not None else "all configs",
            "split": split,
            "dataset_revision": revisions[dataset],
            "parquet_files": len(files),
            "rows": len(rows),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        print(json.dumps({"task": task, **manifest[task]}), flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
