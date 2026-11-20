#!/usr/bin/env python3
"""构建多领域预训练语料：下载固定版本的公开数据，去重、去污染、分词并写出统计。

子命令（需要 ``PYTHONPATH=.``）：

- ``download``：按 spec 选出每个来源的原始文件，下载到临时目录并记录 SHA256；
- ``index-evals``：把 benchmark 题目和 held-out 书籍做成词级 n-gram 索引；
- ``build``：逐来源清洗、全文去重、去污染、分词，写出 ``TokenDataset`` 兼容的分片。数据流
  超过 ``train_tokens`` 上限的部分被截掉（计入 ``over_cap``），所以按文件有序排列的来源应把
  上限设得高于实际产出。``--exclude`` 指向另一份已建好的语料时，两份语料的训练集和 held-out
  互不重叠。来源可以写明 ``keep_repeats``（保留配比有意重复的文档）和 ``heldout_one_in``
  （按文档摘要固定 held-out 的归属，与抽样率无关）；
- ``verify``：把分片解码回文本，复查去污染和跨来源去重确实成立；
- ``report``：汇总各来源的规模、文档长度和过滤统计。

每个来源写出 ``<name>-train`` 和 ``<name>-heldout`` 两个目录，格式与 ``tokenize_corpus.py``
相同（``manifest.json``、uint16 分片、``documents.npy``、``documents.txt``），另有
``digests.npy`` 保存每篇文档的全文摘要，供跨来源去重使用。
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as futures
import gzip
import hashlib
import json
import math
import multiprocessing
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from ttt.corpus import (
    PRIMARY_KEYS,
    EvalIndex,
    EvalIndexBuilder,
    clean_text,
    document_digest,
    iter_strings,
    render_math_problem,
    render_repository,
    sampled,
)

HF_FILE_URL = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
HF_LISTING_URL = "https://huggingface.co/api/datasets/{repo}/revision/{revision}?blobs=true"
PG19_LIST_URL = "https://huggingface.co/datasets/deepmind/pg19/resolve/{revision}/data/train_files.txt"
PG19_FILE_URL = "https://storage.googleapis.com/deepmind-gutenberg/{name}"
DIGEST_BYTES = 12
_STATE = {}


def load_spec(path) -> dict:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("format") != "ttt_corpus_v1":
        raise ValueError("语料 spec 格式不受支持。")
    names = [source["name"] for source in spec["sources"]]
    if len(set(names)) != len(names):
        raise ValueError("语料 spec 中来源名称重复。")
    return spec


def selected_sources(spec: dict, only) -> list[dict]:
    unknown = set(only or ()) - {source["name"] for source in spec["sources"]}
    if unknown:
        raise ValueError(f"spec 中没有这些来源：{sorted(unknown)}")
    return [source for source in spec["sources"] if not only or source["name"] in only]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- download


def _open(url: str, timeout: int = 120):
    request = urllib.request.Request(url, headers={"User-Agent": "corpus-downloader"})
    return urllib.request.urlopen(request, timeout=timeout)


def retry_delay(error: Exception, failures: int) -> float:
    """限流（HTTP 429）时等到服务器给出的窗口结束，其余错误指数退避。"""
    if isinstance(error, urllib.error.HTTPError) and error.code == 429:
        window = re.search(r"t=(\d+)", error.headers.get("RateLimit") or "")
        after = error.headers.get("Retry-After") or ""
        return float(after if after.isdigit() else window.group(1) if window else 60) + 1.0
    return float(2**failures)


def fetch(url: str, path: Path, retries: int = 6) -> dict:
    """流式下载到 ``path``（已存在则跳过），返回字节数和 SHA256。"""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        failures = waits = 0
        while True:
            try:
                with _open(url) as response, temporary.open("wb") as sink:
                    for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                        sink.write(block)
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
                # Hugging Face 对匿名下载按 5 分钟窗口限流，文件多的来源会被要求等待。
                limited = isinstance(error, urllib.error.HTTPError) and error.code == 429
                waits, failures = waits + limited, failures + (not limited)
                if failures >= retries or waits > 24:
                    raise
                time.sleep(retry_delay(error, failures))
        temporary.replace(path)
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def select_files(names: list[str], rule: dict) -> list[str]:
    """按前后缀筛出文件名并排序，再按 ``offset``/``stride``/``limit`` 等间隔取样。"""
    prefix, suffix = rule.get("prefix", ""), rule.get("suffix", "")
    chosen = sorted(name for name in names if name.startswith(prefix) and name.endswith(suffix))
    chosen = chosen[rule.get("offset", 0) :: rule.get("stride", 1)]
    return chosen[: rule["limit"]] if "limit" in rule else chosen


def plan_downloads(source: dict) -> list[tuple[str, str]]:
    """返回 ``(相对路径, URL)`` 列表；选择规则只依赖固定版本的文件清单，因此可复现。"""
    if source["reader"] == "pg19":
        with _open(PG19_LIST_URL.format(revision=source["revision"])) as response:
            names = [line.strip() for line in response.read().decode("utf-8").splitlines() if line.strip()]
        keep = 1.0 / source["select"]["one_in"]
        return [(name, PG19_FILE_URL.format(name=name)) for name in sorted(names) if sampled(name.encode(), keep)]
    with _open(HF_LISTING_URL.format(repo=source["repo"], revision=source["revision"])) as response:
        listing = json.loads(response.read())
    names = select_files([item["rfilename"] for item in listing["siblings"]], source["select"])
    return [(name, HF_FILE_URL.format(repo=source["repo"], revision=source["revision"], path=name)) for name in names]


def command_download(args) -> None:
    spec = load_spec(args.spec)
    for source in selected_sources(spec, args.only):
        root = Path(args.raw) / source["name"]
        plan = plan_downloads(source)
        if not plan:
            raise ValueError(f"来源 {source['name']} 没有选中任何文件。")
        started, records = time.time(), {}
        # PG-19 是几千个小文件，多开一些连接；其余来源是少量大文件。
        workers = 32 if source["reader"] == "pg19" else args.workers
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(fetch, url, root / name): (name, url) for name, url in plan}
            for job in futures.as_completed(jobs):
                name, url = jobs[job]
                records[name] = {"url": url, **job.result()}
        manifest = {
            "name": source["name"],
            "repo": source.get("repo", "deepmind/pg19"),
            "revision": source["revision"],
            "license": source["license"],
            "select": source["select"],
            "files": {name: records[name] for name, _ in plan},
        }
        (root / "source_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        total = sum(record["bytes"] for record in records.values())
        summary = {"source": source["name"], "files": len(plan), "bytes": total, "seconds": time.time() - started}
        print(json.dumps(summary), flush=True)


# --------------------------------------------------------------------------- eval index


def command_index_evals(args) -> None:
    builder, inputs = EvalIndexBuilder(), []

    def add_record(owner: str, value) -> int:
        return sum(builder.add(owner, text, primary=key in PRIMARY_KEYS) for key, text in iter_strings(value))

    for path in map(Path, args.jsonl or ()):
        owner, records, ngrams = f"{path.parent.name}/{path.stem}", 0, 0
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    records += 1
                    ngrams += add_record(owner, json.loads(line))
        inputs.append({"path": str(path), "sha256": sha256_file(path), "records": records, "ngrams": ngrams})
    for path in map(Path, args.json or ()):
        ngrams = add_record(f"{path.parent.name}/{path.stem}", json.loads(path.read_text(encoding="utf-8")))
        inputs.append({"path": str(path), "sha256": sha256_file(path), "records": 1, "ngrams": ngrams})
    for directory in map(Path, args.text_dir or ()):
        owner, files, ngrams = f"{directory.parent.name}/{directory.name}", sorted(directory.glob("*.txt")), 0
        digest = hashlib.sha256()
        for path in files:
            payload = path.read_bytes()
            digest.update(payload)
            ngrams += builder.add_document(owner, clean_text(payload.decode("utf-8", errors="replace")))
        inputs.append({"path": str(directory), "sha256": digest.hexdigest(), "records": len(files), "ngrams": ngrams})
    index = builder.finish()
    index.save(args.output, inputs=inputs)
    print(json.dumps({"output": args.output, "inputs": len(inputs), **index.sizes}))


def index_fingerprint(directory: Path) -> str:
    digest = hashlib.sha256()
    for name in ("index.json", *(f"{table}.npy" for table in EvalIndex.tables)):
        digest.update(sha256_file(directory / name).encode())
    return digest.hexdigest()


# --------------------------------------------------------------------------- readers


def _read_parquet(source, task, counters):
    import pyarrow.parquet as pq

    path, group = task
    reader = source["reader"]
    if reader == "parquet":
        column = source.get("text_column", "text")
        for row, text in enumerate(pq.ParquetFile(path).read_row_group(group, columns=[column]).column(0).to_pylist()):
            if isinstance(text, str):
                yield f"{Path(path).name}:{group}:{row}", text
        return
    render = render_math_problem if reader == "numinamath" else render_repository
    for row, record in enumerate(pq.ParquetFile(path).read_row_group(group).to_pylist()):
        text = render(record)
        if text is None:
            counters["rejected_by_reader"] += 1
        else:
            yield str(record.get("repo_path") or f"{Path(path).name}:{group}:{row}"), text


def _read_jsonl_gz(source, task, counters):
    path, rate = task
    field = source.get("text_column", "text")
    with gzip.open(path, "rb") as lines:
        for row, line in enumerate(lines):
            if not sampled(line[:256], rate):
                counters["not_sampled"] += 1
                continue
            text = json.loads(line).get(field)
            if isinstance(text, str):
                yield f"{Path(path).name}:{row}", text


def zst_lines(path):
    """逐行读 zstd 压缩的文本文件；用 pyarrow 自带的解码器，不需要额外依赖。"""
    import pyarrow as pa

    rest = b""
    with pa.CompressedInputStream(pa.OSFile(str(path)), "zstd") as stream:
        while block := stream.read(16 * 1024 * 1024):
            lines = (rest + block).split(b"\n")
            rest = lines.pop()
            yield from lines
    if rest:
        yield rest


def _read_jsonl_zst(source, task, counters):
    path, rate = task
    field = source.get("text_column", "text")
    label = f"{Path(path).parent.name}/{Path(path).name}"
    for row, line in enumerate(zst_lines(path)):
        # 按行号抽样：配比里被上采样的文档在文件中重复出现，每一份独立决定去留，
        # 所以抽出来的仍是原配比的均匀样本。
        if not sampled(f"{label}:{row}".encode(), rate):
            counters["not_sampled"] += 1
            continue
        text = json.loads(line).get(field) if line.strip() else None
        if isinstance(text, str):
            yield f"{label}:{row}", text


def _read_text_files(source, task, counters):
    for path in task:
        yield Path(path).name, Path(path).read_text(encoding="utf-8", errors="replace")


READERS = {
    "parquet": _read_parquet,
    "numinamath": _read_parquet,
    "stack_v3": _read_parquet,
    "jsonl_gz": _read_jsonl_gz,
    "jsonl_zst": _read_jsonl_zst,
    "pg19": _read_text_files,
}


def build_tasks(source: dict, raw: Path) -> list:
    manifest = json.loads((raw / "source_manifest.json").read_text(encoding="utf-8"))
    files = [(raw / name, record) for name, record in manifest["files"].items()]
    reader = source["reader"]
    if reader == "pg19":
        paths = [str(path) for path, _ in files]
        return [paths[index : index + 16] for index in range(0, len(paths), 16)]
    if reader == "jsonl_gz":
        # sqrt 规则：每个文件的取样量与其体积的平方根成正比，大站点不至于淹没小站点。
        scale = source.get("sample_sqrt_scale")
        return [(str(path), min(1.0, scale / math.sqrt(record["bytes"])) if scale else 1.0) for path, record in files]
    if reader == "jsonl_zst":
        return [(str(path), source.get("sample_rate", 1.0)) for path, _ in files]
    import pyarrow.parquet as pq

    return [(str(path), group) for path, _ in files for group in range(pq.ParquetFile(path).metadata.num_row_groups)]


# --------------------------------------------------------------------------- build


def _initialise(encoding_name: str, index_directory: str, source: dict) -> None:
    import tiktoken

    _STATE.update(encoding=tiktoken.get_encoding(encoding_name), index=EvalIndex.load(index_directory), source=source)


def _process(task):
    """工作进程：读一个任务里的文档，清洗、去污染、分词；返回拼接后的 token 和逐文档信息。"""
    source, encoding, index = _STATE["source"], _STATE["encoding"], _STATE["index"]
    counters = collections.Counter()
    eot, minimum = encoding.eot_token, source.get("min_chars", 200)
    arrays, lengths, digests, names, sizes = [], [], [], [], []
    for name, text in READERS[source["reader"]](source, task, counters):
        counters["read"] += 1
        text = clean_text(text)
        if len(text) < minimum:
            counters["too_short"] += 1
            continue
        hit = index.match(text)
        if hit is not None:
            counters[f"contaminated|{hit[0]}|{hit[1]}"] += 1
            continue
        tokens = np.fromiter(encoding.encode_ordinary(text), dtype=np.uint16)
        arrays.append(np.concatenate((np.asarray([eot], dtype=np.uint16), tokens)))
        lengths.append(arrays[-1].size)
        digests.append(document_digest(text))
        names.append(name)
        sizes.append(len(text.encode("utf-8")))
    tokens = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.uint16)
    return tokens, lengths, digests, names, sizes, dict(counters)


class ShardWriter:
    """按文档追加 token，写出定长分片、文档边界、文档名和全文摘要。"""

    def __init__(self, directory: Path, version: str, shard_tokens: int):
        self.directory, self.version, self.shard_tokens = directory, version, shard_tokens
        self.buffer, self.buffered = [], 0
        self.records, self.starts, self.names, self.digests = [], [0], [], []
        self.tokens = self.bytes = 0

    def add(self, tokens: np.ndarray, name: str, digest: bytes, size: int) -> None:
        self.buffer.append(tokens)
        self.buffered += int(tokens.size)
        self.tokens += int(tokens.size)
        self.bytes += size
        self.starts.append(self.tokens)
        self.names.append(name)
        self.digests.append(digest)
        while self.buffered >= self.shard_tokens:
            self._flush(self.shard_tokens)

    def _flush(self, count: int) -> None:
        merged = np.concatenate(self.buffer) if len(self.buffer) > 1 else self.buffer[0]
        shard, rest = merged[:count], merged[count:]
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{self.version}-{len(self.records):05d}.npy"
        with path.open("wb") as handle:
            np.save(handle, shard, allow_pickle=False)
        self.records.append({"tokens": path.name, "count": int(shard.size), "sha256": sha256_file(path)})
        self.buffer, self.buffered = ([rest] if rest.size else []), int(rest.size)

    def finish(self, manifest: dict) -> dict | None:
        if self.tokens < 2:
            return None
        if self.buffered:
            self._flush(self.buffered)
        starts = np.asarray(self.starts, dtype=np.int64)
        np.save(self.directory / "documents.npy", starts, allow_pickle=False)
        (self.directory / "documents.txt").write_text("\n".join(self.names) + "\n", encoding="utf-8")
        digests = np.frombuffer(b"".join(self.digests), dtype=np.uint8).reshape(-1, DIGEST_BYTES)
        np.save(self.directory / "digests.npy", digests, allow_pickle=False)
        stats = {
            "documents": len(self.names),
            "tokens": self.tokens,
            "utf8_bytes": self.bytes,
            "bytes_per_token": self.bytes / self.tokens,
            **length_statistics(starts),
        }
        manifest = {
            **manifest,
            "version": self.version,
            "shards": self.records,
            "stats": stats,
            "documents_index_sha256": sha256_file(self.directory / "documents.npy"),
        }
        temporary = self.directory / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        temporary.replace(self.directory / "manifest.json")
        return stats


def length_statistics(starts: np.ndarray) -> dict:
    """文档长度分布，以及落在长文档里的 token 占比（决定训练窗口里有多少真实长程上下文）。"""
    lengths = np.diff(starts)
    total = int(lengths.sum())
    return {
        "document_tokens_mean": float(lengths.mean()),
        "document_tokens_median": float(np.median(lengths)),
        "document_tokens_p90": float(np.percentile(lengths, 90)),
        "document_tokens_max": int(lengths.max()),
        "token_share_in_documents_ge_2048": float(lengths[lengths >= 2048].sum() / total),
        "token_share_in_documents_ge_4096": float(lengths[lengths >= 4096].sum() / total),
    }


def existing_digests(output: Path) -> set:
    """读入输出根目录下已建好的全部文档摘要，使分批构建也能跨来源去重。"""
    seen = set()
    for path in sorted(output.glob("*/digests.npy")):
        seen.update(row.tobytes() for row in np.load(path, allow_pickle=False))
    return seen


def foreign_digests(roots) -> tuple[set, set]:
    """其他语料的文档摘要，分成 held-out 和训练集两组。

    held-out 里出现过的文档不进本语料；训练集里出现过的文档不进本语料的 held-out。
    这样两份语料各自训练出的模型可以互相用对方的 held-out 评测。
    """
    heldout, train = set(), set()
    for root in roots or ():
        for path in sorted(Path(root).glob("*/digests.npy")):
            target = heldout if path.parent.name.endswith("-heldout") else train
            target.update(row.tobytes() for row in np.load(path, allow_pickle=False))
    return heldout, train


def nested_counters(counters: collections.Counter) -> dict:
    """把 ``contaminated|层级|评测来源`` 形式的计数整理成嵌套字典。"""
    result, contaminated = {}, {}
    for key, value in sorted(counters.items()):
        if key.startswith("contaminated|"):
            _, tier, owner = key.split("|", 2)
            contaminated.setdefault(tier, {})[owner] = value
        else:
            result[key] = value
    result["contaminated"] = contaminated
    result["contaminated_total"] = sum(sum(tier.values()) for tier in contaminated.values())
    return result


def build_source(spec: dict, source: dict, args, seen: set, blocked=frozenset(), foreign=frozenset()) -> None:
    import tiktoken

    name, output = source["name"], Path(args.output)
    if (output / f"{name}-train" / "manifest.json").exists():
        print(json.dumps({"source": name, "status": "exists"}), flush=True)
        return
    encoding = tiktoken.get_encoding(spec["tokenizer"])
    if encoding.n_vocab > 65536:
        raise ValueError("uint16 分片只支持不超过 65536 的词表。")
    tasks = build_tasks(source, Path(args.raw) / name)
    caps = {"train": source["train_tokens"], "heldout": source.get("heldout_tokens", 0)}
    # 大语料在 spec 里写明更大的分片：载入器为每个分片保持一个打开的文件，分片太多会超出进程的文件数上限。
    shard_tokens = spec.get("shard_tokens", args.shard_tokens)
    writers = {split: ShardWriter(output / f"{name}-{split}", f"{name}-{split}", shard_tokens) for split in caps}
    # held-out 按文档摘要抽取，均匀取自整个数据流。来源写明 heldout_one_in 时，被抽中的文档永远不进
    # 训练集（held-out 写满后直接丢弃），所以以后调整抽样率，旧的 held-out 文档也不会混进训练集；
    # 没写明时抽取率按两个上限的比例定，held-out 写满后其余被抽中的文档归入训练集。
    strict = "heldout_one_in" in source
    derived = max(2, caps["train"] // max(caps["heldout"], 1))
    one_in = source.get("heldout_one_in") or spec.get("heldout_one_in") or derived
    # 写明 keep_repeats 的来源保留来源内部的重复文档：那是配比有意做的上采样，去掉会改变配比。
    keep_repeats, own = bool(source.get("keep_repeats")), set()
    counters, started = collections.Counter(), time.time()
    context = multiprocessing.get_context("spawn")
    with context.Pool(args.processes, _initialise, (spec["tokenizer"], args.evals, source)) as pool:
        for done, result in enumerate(pool.imap(_process, tasks, chunksize=1), 1):
            tokens, lengths, digests, names, sizes, task_counters = result
            counters.update(task_counters)
            offset = 0
            for length, digest, document, size in zip(lengths, digests, names, sizes):
                document_tokens = tokens[offset : offset + length]
                offset += length
                if digest in blocked:
                    counters["in_other_heldout"] += 1
                    continue
                repeat = keep_repeats and digest in own
                if digest in seen and not repeat:
                    counters["duplicate"] += 1
                    continue
                split = "train"
                if int.from_bytes(digest[:4], "little") % one_in == 0 and digest not in foreign:
                    full = writers["heldout"].tokens >= caps["heldout"]
                    if repeat or (strict and full):
                        counters["heldout_repeat" if repeat else "heldout_over_cap"] += 1
                        continue
                    split = "train" if full else "heldout"
                if writers[split].tokens >= caps[split]:
                    counters["over_cap"] += 1
                    continue
                if repeat:
                    counters["repeat_kept"] += 1
                seen.add(digest)
                own.add(digest)
                writers[split].add(document_tokens, document, digest, size)
            if done % 25 == 0 or done == len(tasks):
                progress = {split: writer.tokens for split, writer in writers.items()}
                print(json.dumps({"source": name, "tasks": done, "of": len(tasks), **progress}), flush=True)
            if all(writers[split].tokens >= caps[split] for split in caps):
                counters["tasks_not_needed"] = len(tasks) - done
                break
    raw_manifest = json.loads((Path(args.raw) / name / "source_manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "format_version": 1,
        "vocab_size": encoding.n_vocab,
        "tokenizer": f"tiktoken:{spec['tokenizer']}",
        "supervised": False,
        "document_format": "[EOT] + encode_ordinary(text)",
        "source": source,
        "raw_files": {path: record["sha256"] for path, record in raw_manifest["files"].items()},
        "eval_index_sha256": index_fingerprint(Path(args.evals)),
        "filters": nested_counters(counters),
    }
    if args.exclude:
        manifest["excluded_corpora"] = [Path(root).name for root in args.exclude]
    summary = {split: writer.finish({**manifest, "split": split}) for split, writer in writers.items()}
    summary = {"source": name, "seconds": round(time.time() - started, 1), **summary, "filters": manifest["filters"]}
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def command_build(args) -> None:
    spec = load_spec(args.spec)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seen = existing_digests(output)
    blocked, foreign = foreign_digests(args.exclude)
    for source in selected_sources(spec, args.only):
        build_source(spec, source, args, seen, blocked, foreign)


# --------------------------------------------------------------------------- verify


def _verify(task):
    """把一段文档解码回文本，重新计算摘要并复查去污染。"""
    directory, first, last = task
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    # 每个任务只看一小段文档，索引和分片都按需映射，不整份读入。
    starts = np.load(directory / "documents.npy", mmap_mode="r")
    digests = np.load(directory / "digests.npy", mmap_mode="r")
    shards = [np.load(directory / record["tokens"], mmap_mode="r") for record in manifest["shards"]]
    ends = np.cumsum([shard.size for shard in shards])
    encoding, index = _STATE["encoding"], _STATE["index"]
    problems = []
    for document in range(first, last):
        begin, end = int(starts[document]), int(starts[document + 1])
        pieces, position = [], begin
        while position < end:
            shard = int(np.searchsorted(ends, position, side="right"))
            base = int(ends[shard - 1]) if shard else 0
            stop = min(end, int(ends[shard]))
            pieces.append(np.asarray(shards[shard][position - base : stop - base]))
            position = stop
        tokens = np.concatenate(pieces)
        if tokens[0] != encoding.eot_token:
            problems.append((document, "missing_eot"))
            continue
        text = encoding.decode(tokens[1:].tolist())
        if document_digest(text) != digests[document].tobytes():
            problems.append((document, "digest_mismatch"))
        hit = index.match(text)
        if hit is not None:
            problems.append((document, f"contaminated|{hit[0]}|{hit[1]}"))
    return str(directory), last - first, problems


def command_verify(args) -> None:
    output = Path(args.output)
    directories = sorted(path.parent for path in output.glob("*/manifest.json"))
    spec = load_spec(args.spec)
    # 只有写明 keep_repeats 的来源，训练集里才允许同一篇文档出现多次。
    repeatable = {f"{source['name']}-train" for source in spec["sources"] if source.get("keep_repeats")}
    seen, duplicates, repeats, tasks = set(), 0, {}, []
    blocked, foreign = foreign_digests(args.exclude)
    overlap = {"train_in_other_heldout": 0, "heldout_in_other_train": 0}
    for directory in directories:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for record in manifest["shards"]:
            if sha256_file(directory / record["tokens"]) != record["sha256"]:
                raise ValueError(f"分片校验失败：{directory.name}/{record['tokens']}")
        heldout = directory.name.endswith("-heldout")
        digests = np.load(directory / "digests.npy")
        local = {row.tobytes() for row in digests}
        if len(local) != len(digests):
            repeats[directory.name] = len(digests) - len(local)
        duplicates += len(local & seen)
        seen |= local
        overlap["heldout_in_other_train" if heldout else "train_in_other_heldout"] += len(
            local & (foreign if heldout else blocked)
        )
        documents = int(manifest["stats"]["documents"])
        step = max(1, min(args.documents_per_task, math.ceil(documents / args.processes)))
        tasks += [(str(directory), first, min(first + step, documents)) for first in range(0, documents, step)]
    unexpected = sum(count for name, count in repeats.items() if name not in repeatable)
    checked, problems = collections.Counter(), collections.Counter()
    context = multiprocessing.get_context("spawn")
    with context.Pool(args.processes, _initialise, (spec["tokenizer"], args.evals, {})) as pool:
        for directory, count, found in pool.imap_unordered(_verify, tasks, chunksize=1):
            checked[Path(directory).name] += count
            for _, kind in found:
                problems[f"{Path(directory).name}:{kind.split('|')[0]}"] += 1
    result = {
        "directories": len(directories),
        "documents_checked": dict(sorted(checked.items())),
        "unique_digests": len(seen),
        "cross_source_duplicates": int(duplicates),
        "problems": dict(sorted(problems.items())),
        "eval_index_sha256": index_fingerprint(Path(args.evals)),
        "passed": not duplicates and not unexpected and not problems and not any(overlap.values()),
    }
    if repeats:
        result["repeats_within_source"] = repeats
    if args.exclude:
        result["excluded_corpora"] = {"roots": [Path(root).name for root in args.exclude], **overlap}
    Path(args.report).write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


# --------------------------------------------------------------------------- report


def command_report(args) -> None:
    spec, output = load_spec(args.spec), Path(args.output)
    sources = {}
    for source in spec["sources"]:
        entry = {key: source.get(key) for key in ("domain", "repo", "revision", "license", "description")}
        for split in ("train", "heldout"):
            path = output / f"{source['name']}-{split}" / "manifest.json"
            if path.exists():
                manifest = json.loads(path.read_text(encoding="utf-8"))
                entry[split] = manifest["stats"]
                entry["filters"] = manifest["filters"]
                entry["eval_index_sha256"] = manifest["eval_index_sha256"]
                entry[f"{split}_manifest_sha256"] = sha256_file(path)
        sources[source["name"]] = entry
    report = {"spec": str(args.spec), "spec_sha256": sha256_file(Path(args.spec)), "sources": sources}
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name, entry in sources.items():
        train = entry.get("train")
        if train:
            print(
                f"{name:14s} docs={train['documents']:9d} tokens={train['tokens'] / 1e6:8.1f}M "
                f"bytes/token={train['bytes_per_token']:.2f} median={train['document_tokens_median']:.0f} "
                f">=4096:{train['token_share_in_documents_ge_4096']:.1%} "
                f"dup={entry['filters'].get('duplicate', 0)} repeats={entry['filters'].get('repeat_kept', 0)} "
                f"contaminated={entry['filters']['contaminated_total']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="构建多领域预训练语料")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--workers", type=int, default=8)
    index = commands.add_parser("index-evals")
    index.add_argument("--jsonl", nargs="*", help="每行一条评测记录的 JSONL 文件")
    index.add_argument("--json", nargs="*", help="整份 JSON 的评测文件")
    index.add_argument("--text-dir", nargs="*", help="held-out 书籍目录，每个 .txt 一篇")
    index.add_argument("--output", required=True)
    build = commands.add_parser("build")
    build.add_argument("--shard-tokens", type=int, default=100_000_000)
    verify = commands.add_parser("verify")
    verify.add_argument("--documents-per-task", type=int, default=2000)
    report = commands.add_parser("report")
    for command in (download, build, verify, report):
        command.add_argument("--spec", required=True)
    for command in (download, build):
        command.add_argument("--raw", required=True, help="原始文件的临时目录")
        command.add_argument("--only", nargs="+", help="只处理这些来源")
    for command in (build, verify):
        command.add_argument("--evals", required=True, help="index-evals 的输出目录")
        command.add_argument("--processes", type=int, default=64)
        command.add_argument("--exclude", nargs="*", default=[], help="其他语料的分片根目录，两边互不重叠")
    for command in (build, verify, report):
        command.add_argument("--output", required=True, help="token 分片的根目录")
    for command in (verify, report):
        command.add_argument("--report", required=True, help="结果 JSON 的输出路径")
    args = parser.parse_args()
    handlers = {
        "download": command_download,
        "index-evals": command_index_evals,
        "build": command_build,
        "verify": command_verify,
        "report": command_report,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
