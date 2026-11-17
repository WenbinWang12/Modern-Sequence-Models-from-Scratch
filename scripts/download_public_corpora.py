#!/usr/bin/env python3
"""下载公开预训练语料并写出带 SHA256 的来源清单。

支持两个来源：

- ``pg19``：DeepMind 官方 GCS 桶中的 PG-19（Project Gutenberg 长篇书籍），
  train/validation/test 划分沿用官方文件列表；
- ``fineweb-edu``：HuggingFaceFW/fineweb-edu 的 ``sample/10BT`` parquet 分片。

脚本只下载数据文件，不执行任何数据集自带代码。重复运行会跳过已完成的文件。
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

PG19_REPO_REVISION = "4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b"
PG19_LIST_URL = "https://huggingface.co/datasets/deepmind/pg19/resolve/{revision}/data/{split}_files.txt"
PG19_FILE_URL = "https://storage.googleapis.com/deepmind-gutenberg/{name}"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
FINEWEB_URL = (
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/{revision}/sample/10BT/{index:03d}_00000.parquet"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_bytes(url: str, *, retries: int = 6, timeout: int = 120) -> bytes:
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "corpus-downloader"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def fetch_small_file(url: str, path: Path) -> dict:
    """下载单个小文件；已存在则只计算摘要。"""
    if not path.exists():
        payload = fetch_bytes(url)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(path)
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def download_pg19(output: Path, splits: tuple[str, ...], workers: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "PG-19",
        "source": "https://storage.googleapis.com/deepmind-gutenberg",
        "file_lists": "deepmind/pg19",
        "file_lists_revision": PG19_REPO_REVISION,
        "license": "apache-2.0 (dataset card); texts are public-domain Project Gutenberg books",
        "splits": {},
    }
    for split in splits:
        listing = fetch_bytes(PG19_LIST_URL.format(revision=PG19_REPO_REVISION, split=split)).decode("utf-8")
        names = [line.strip() for line in listing.splitlines() if line.strip()]
        (output / split).mkdir(exist_ok=True)

        def job(name: str) -> tuple[str, dict]:
            return name, fetch_small_file(PG19_FILE_URL.format(name=name), output / name)

        records, started = {}, time.time()
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for index, (name, record) in enumerate(pool.map(job, names), start=1):
                records[name] = record
                if index % 2000 == 0 or index == len(names):
                    print(
                        json.dumps(
                            {"split": split, "done": index, "total": len(names), "seconds": time.time() - started}
                        ),
                        flush=True,
                    )
        manifest["splits"][split] = {
            "files": len(records),
            "bytes": sum(record["bytes"] for record in records.values()),
            "records": records,
        }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {name: {k: v for k, v in split.items() if k != "records"} for name, split in manifest["splits"].items()}
        )
    )


def download_fineweb(output: Path, indices: list[int]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for index in indices:
        url = FINEWEB_URL.format(revision=FINEWEB_REVISION, index=index)
        path = output / f"{index:03d}_00000.parquet"
        marker = path.with_suffix(".done")
        if not marker.exists():
            subprocess.run(
                ["curl", "-L", "--fail", "--retry", "8", "--retry-delay", "5", "-C", "-", "-sS", "-o", str(path), url],
                check=True,
            )
            marker.write_text("ok", encoding="utf-8")
        records[path.name] = {"url": url, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        print(json.dumps({"file": path.name, **records[path.name]}), flush=True)
    manifest = {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample/10BT",
        "revision": FINEWEB_REVISION,
        "license": "odc-by",
        "files": records,
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载公开语料并记录来源摘要")
    parser.add_argument("--dataset", choices=("pg19", "fineweb-edu"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+", default=["test", "validation", "train"])
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--shards", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    if args.dataset == "pg19":
        download_pg19(Path(args.output), tuple(args.splits), args.workers)
    else:
        download_fineweb(Path(args.output), args.shards)


if __name__ == "__main__":
    main()
