#!/usr/bin/env python3
"""多进程 BPE 分词，写出 TokenDataset 可读的分片清单和文档边界索引。

每篇文档编码为 ``[EOT] + tokens``；EOT 作为文档分隔符放在文档开头（GPT-2 约定）。
输出目录包含：

- ``<version>-NNNNN.npy``：uint16 token 分片；
- ``manifest.json``：与 ``ttt.dataset.TokenDataset`` 兼容的清单（含每个分片的 SHA256）；
- ``documents.npy``：int64 文档起始偏移，长度为文档数 + 1，最后一项是 token 总数；
- ``documents.txt``：文档标识，每行一个，与 ``documents.npy`` 对齐。

输入可以是 PG-19 风格的纯文本目录，也可以是带 ``text`` 列的 parquet 文件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import re
import time
import uuid
from pathlib import Path

import numpy as np

_ENCODING = None
_EXCLUDED = frozenset()


def _initialize(encoding_name: str, excluded=()) -> None:
    global _ENCODING, _EXCLUDED
    import tiktoken

    _ENCODING = tiktoken.get_encoding(encoding_name)
    _EXCLUDED = frozenset(excluded)


def document_digest(text: str) -> str:
    """与 audit_report_data.py 相同的规范化全文摘要：小写、连续空白压成一个空格。"""
    normalised = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.blake2b(normalised.encode("utf-8"), digest_size=12).hexdigest()


def _encode_documents(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    eot = _ENCODING.eot_token
    arrays, lengths = [], []
    for text in texts:
        tokens = np.fromiter(_ENCODING.encode_ordinary(text), dtype=np.uint16)
        arrays.append(np.concatenate((np.asarray([eot], dtype=np.uint16), tokens)))
        lengths.append(arrays[-1].size)
    if not arrays:
        return np.zeros((0,), dtype=np.uint16), np.zeros((0,), dtype=np.int64)
    return np.concatenate(arrays), np.asarray(lengths, dtype=np.int64)


def _text_file_task(task):
    paths = task
    texts = [Path(path).read_text(encoding="utf-8", errors="replace") for path in paths]
    tokens, lengths = _encode_documents(texts)
    return tokens, lengths, [Path(path).name for path in paths], sum(len(text.encode("utf-8")) for text in texts), 0


def _parquet_task(task):
    import pyarrow.parquet as pq

    path, row_group, column, limit = task
    table = pq.ParquetFile(path).read_row_group(row_group, columns=[column])
    texts = table.column(column).to_pylist()
    if limit is not None:
        texts = texts[:limit]
    texts = [text for text in texts if isinstance(text, str) and text.strip()]
    kept = [text for text in texts if document_digest(text) not in _EXCLUDED] if _EXCLUDED else texts
    tokens, lengths = _encode_documents(kept)
    names = [f"{Path(path).name}:{row_group}:{index}" for index in range(len(kept))]
    return tokens, lengths, names, sum(len(text.encode("utf-8")) for text in kept), len(texts) - len(kept)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_tasks(args):
    if args.source == "text-dir":
        files = sorted(Path(args.input[0]).glob("*.txt"), key=lambda path: path.name)
        if args.max_documents is not None:
            files = files[: args.max_documents]
        if not files:
            raise ValueError("输入目录中没有 .txt 文件。")
        batch = args.documents_per_task
        return _text_file_task, [[str(path) for path in files[i : i + batch]] for i in range(0, len(files), batch)]
    import pyarrow.parquet as pq

    tasks, remaining = [], args.max_documents
    for path in args.input:
        metadata = pq.ParquetFile(path).metadata
        for row_group in range(metadata.num_row_groups):
            rows = metadata.row_group(row_group).num_rows
            limit = None
            if remaining is not None:
                if remaining <= 0:
                    return _parquet_task, tasks
                limit = min(rows, remaining)
                remaining -= limit
            tasks.append((str(path), row_group, args.text_column, limit))
    return _parquet_task, tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="多进程分词并写出分片清单")
    parser.add_argument("--source", choices=("text-dir", "parquet"), required=True)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoding", default="gpt2")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--shard-tokens", type=int, default=100_000_000)
    parser.add_argument("--processes", type=int, default=48)
    parser.add_argument("--documents-per-task", type=int, default=32)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--exclude-digests", help="JSON 文件，内含需要剔除的文档摘要列表")
    parser.add_argument("--exclude-key", default="fineweb_edu.heldout_duplicate_digests")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("清单已存在；新版本请使用新目录。")
    import tiktoken

    encoding = tiktoken.get_encoding(args.encoding)
    if encoding.n_vocab > 65536:
        raise ValueError("uint16 分片只支持不超过 65536 的词表。")
    worker, tasks = build_tasks(args)
    excluded = ()
    if args.exclude_digests:
        excluded = json.loads(Path(args.exclude_digests).read_text(encoding="utf-8"))
        for key in args.exclude_key.split("."):
            excluded = excluded[key]
    version = uuid.uuid4().hex
    records, starts, names = [], [0], []
    buffer, buffered, total_tokens, total_bytes, skipped = [], 0, 0, 0, 0
    started = time.time()

    def flush(count: int) -> None:
        nonlocal buffer, buffered
        merged = np.concatenate(buffer) if len(buffer) > 1 else buffer[0]
        shard, rest = merged[:count], merged[count:]
        path = output / f"{version}-{len(records):05d}.npy"
        with path.open("xb") as handle:
            np.save(handle, shard, allow_pickle=False)
        records.append({"tokens": path.name, "count": int(shard.size), "sha256": sha256_file(path)})
        buffer, buffered = ([rest] if rest.size else []), int(rest.size)

    with multiprocessing.get_context("spawn").Pool(
        args.processes, initializer=_initialize, initargs=(args.encoding, tuple(excluded))
    ) as pool:
        for index, (tokens, lengths, task_names, byte_count, dropped) in enumerate(
            pool.imap(worker, tasks, chunksize=1), 1
        ):
            skipped += dropped
            if tokens.size:
                buffer.append(tokens)
                buffered += int(tokens.size)
                total_bytes += byte_count
                names.extend(task_names)
                for length in lengths.tolist():
                    total_tokens += length
                    starts.append(total_tokens)
            while buffered >= args.shard_tokens:
                flush(args.shard_tokens)
            if index % 50 == 0 or index == len(tasks):
                print(
                    json.dumps(
                        {
                            "tasks": index,
                            "of": len(tasks),
                            "documents": len(names),
                            "tokens": total_tokens,
                            "seconds": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
    if buffered:
        flush(buffered)
    if total_tokens < 2:
        raise ValueError("分词结果不足两个 token。")
    np.save(output / "documents.npy", np.asarray(starts, dtype=np.int64), allow_pickle=False)
    (output / "documents.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    manifest = {
        "format_version": 1,
        "version": version,
        "vocab_size": encoding.n_vocab,
        "tokenizer": f"tiktoken:{args.encoding}",
        "supervised": False,
        "document_format": "[EOT] + encode_ordinary(text)",
        "shards": records,
        "stats": {
            "documents": len(names),
            "tokens": total_tokens,
            "utf8_bytes": total_bytes,
            "bytes_per_token": total_bytes / total_tokens,
            "excluded_duplicate_documents": skipped,
        },
        "inputs": [str(path) for path in args.input],
        "max_documents": args.max_documents,
        "exclude_digests": args.exclude_digests,
        "documents_index_sha256": sha256_file(output / "documents.npy"),
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    temporary.replace(output / "manifest.json")
    print(json.dumps({"manifest": str(output / "manifest.json"), **manifest["stats"]}))


if __name__ == "__main__":
    main()
