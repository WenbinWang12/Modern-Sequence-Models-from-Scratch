import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ttt.corpus import (
    EvalIndex,
    EvalIndexBuilder,
    clean_text,
    document_digest,
    iter_strings,
    keep_code_file,
    ngram_hashes,
    render_math_problem,
    render_repository,
    sampled,
)

ROOT = Path(__file__).resolve().parents[1]
QUESTION = (
    "A student pushes a wooden block across a rough table and then lets go; "
    "which force is most responsible for slowing the block down until it finally stops?"
)


def build_index(*records):
    builder = EvalIndexBuilder()
    for owner, text, primary in records:
        builder.add(owner, text, primary=primary)
    return builder.finish()


def code_file(path, content, **changes):
    item = {
        "file_path": path,
        "content_id": path,
        "content": content,
        "language": "Python",
        "license_type": "permissive",
        "is_vendor": False,
    }
    return {**item, **changes}


def test_clean_text_and_digest_ignore_formatting():
    assert clean_text("  a\x00b" + " " * 500 + "c\n") == "ab" + " " * 128 + "c"
    assert clean_text("bad \udc80 surrogate") == "bad ? surrogate"
    assert document_digest("Hello   World\n") == document_digest("hello world")
    assert document_digest("hello world") != document_digest("hello worlds")
    assert len(document_digest("x")) == 12


def test_sampling_is_deterministic_and_close_to_rate():
    keys = [f"thread-{index}".encode() for index in range(20000)]
    kept = [key for key in keys if sampled(key, 0.25)]
    assert kept == [key for key in keys if sampled(key, 0.25)]
    assert 0.22 < len(kept) / len(keys) < 0.28
    assert all(sampled(key, 1.0) for key in keys[:10])


def test_position_keys_sample_at_the_requested_rate():
    keys = [f"common_crawl-health-0017/shard_00000875.jsonl.zst:{row}".encode() for row in range(200000)]
    kept = np.asarray([sampled(key, 0.032) for key in keys])
    assert 0.029 < kept.mean() < 0.035
    # 相邻两行是否同时被抽中应接近独立。
    assert (kept[1:] & kept[:-1]).mean() < 3 * 0.032**2


def test_ngram_hashes_depend_on_order_and_skip_numeric_windows():
    index = build_index(("eval", QUESTION, True))
    assert index.sizes == {"ngram13": 16, "ngram8": 0, "span13": 0}
    words = np.arange(2, 60, 2, dtype=np.uint64)
    assert ngram_hashes(words, 13, 1).size == 0
    assert ngram_hashes(words | np.uint64(1), 13, 8).size == words.size - 12
    forward = ngram_hashes(np.asarray([3, 5, 7], dtype=np.uint64), 3, 0)
    backward = ngram_hashes(np.asarray([7, 5, 3], dtype=np.uint64), 3, 0)
    assert forward[0] != backward[0]


def test_eval_index_matches_reformatted_copies_only():
    index = build_index(
        ("arc", QUESTION, True),
        ("piqa", "how do you remove a splinter from under your fingernail", True),
        ("choices", "the force of gravity acting on the wooden block", False),
    )
    copied = "## Quiz\n\n**A student pushes a wooden block** across a rough table, and then lets go;\nwhich force " + (
        "is most responsible for slowing the block down until it finally stops?"
    )
    assert index.match(copied) == ("ngram13", "arc")
    assert index.match("Tip: How do you remove a splinter from under your fingernail? Use tape.") == ("ngram8", "piqa")
    # 非题干字段不足 13 个词，不进索引。
    assert index.match("Here the force of gravity acting on the wooden block is small.") is None
    assert index.match("A student pushes a wooden block across a smooth floor and measures its speed.") is None
    assert index.match("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16") is None


def book_words(first, last):
    return " ".join(f"tale{index}" for index in range(first, last))


def test_heldout_documents_need_a_long_verbatim_passage():
    builder = EvalIndexBuilder()
    builder.add_document("books", book_words(0, 300))
    index = builder.finish()
    assert index.sizes == {"ngram13": 0, "ngram8": 0, "span13": 288}
    filler = " and then some unrelated commentary follows here "
    # 49 个词逐字重合不算，50 个词算；两段各 30 个词但不相连也不算。
    assert index.match("intro " + book_words(100, 149) + filler) is None
    assert index.match("intro " + book_words(100, 150) + filler) == ("span50", "books")
    assert index.match(book_words(10, 40) + filler + book_words(200, 230)) is None
    assert index.match(book_words(0, 300).upper().replace(" ", ",\n")) == ("span50", "books")


def test_eval_index_roundtrip(tmp_path):
    builder = EvalIndexBuilder()
    builder.add("arc", QUESTION, primary=True)
    builder.add("piqa", "how do you remove a splinter from under your nail", primary=True)
    builder.add_document("books", book_words(0, 120))
    index = builder.finish()
    index.save(tmp_path, inputs=[])
    loaded = EvalIndex.load(tmp_path)
    assert loaded.sizes == index.sizes == {"ngram13": 16, "ngram8": 3, "span13": 108}
    assert loaded.match(QUESTION.upper()) == ("ngram13", "arc")
    assert loaded.match("see " + book_words(20, 90)) == ("span50", "books")
    assert loaded.match("unrelated text about cooking rice in a pot with a lid on low heat today") is None


def test_iter_strings_reports_leaf_keys():
    row = {"question": "q", "choices": {"text": ["a", "b"], "label": ["A", "B"]}, "id": 3}
    assert list(iter_strings(row)) == [("question", "q"), ("text", "a"), ("text", "b"), ("label", "A"), ("label", "B")]


def test_code_file_filters():
    body = "def add(a, b):\n    return a + b\n\n\nprint(add(1, 2))\n" * 3
    assert keep_code_file(code_file("a.py", body))
    assert not keep_code_file(code_file("a.py", body, license_type="no_license"))
    assert not keep_code_file(code_file("a.py", body, is_vendor=True))
    assert not keep_code_file(code_file("a.json", body, language="JSON"))
    assert not keep_code_file(code_file("a.py", "x = 1\n"))
    assert not keep_code_file(code_file("a.py", "x = '" + "a" * 1500 + "'\n"))
    assert not keep_code_file(code_file("a.py", "# Generated by protoc. DO NOT EDIT.\n" + body))
    assert not keep_code_file(code_file("a.py", "(){}[];,.:" * 40))


def test_render_repository_orders_and_deduplicates_files():
    body = "def add(a, b):\n    return a + b\n\n\nprint(add(1, 2))\n" * 3
    files = [
        code_file("src/z.py", body + "# z\n"),
        code_file("README.md", body + "# readme\n", language="Markdown"),
        code_file("src/a.py", body + "# a\n"),
        code_file("src/a.py", body + "# a\n"),
        code_file("src/lib/b.py", body + "# b\n"),
        code_file("data.py", body, license_type="no_license"),
    ]
    text = render_repository({"repo_path": "owner/project", "files": files})
    assert text.startswith("Repository: owner/project\n\nFile: README.md\n")
    order = [line[6:] for line in text.split("\n") if line.startswith("File: ")]
    assert order == ["README.md", "src/a.py", "src/z.py", "src/lib/b.py"]
    assert render_repository({"repo_path": "x/y", "files": files[-1:]}) is None
    capped = render_repository({"repo_path": "owner/project", "files": files}, max_chars=len(body) + 20)
    assert [line[6:] for line in capped.split("\n") if line.startswith("File: ")] == ["README.md"]


def test_render_math_problem_requires_valid_rows():
    row = {"problem": " 1+1? ", "solution": " 2 ", "problem_is_valid": "Yes", "solution_is_valid": "Yes"}
    assert render_math_problem(row) == "Problem:\n1+1?\n\nSolution:\n2"
    assert render_math_problem({**row, "solution_is_valid": "Incomplete"}) is None
    assert render_math_problem({**row, "solution": "  "}) is None


def run_cli(*arguments):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_corpus.py"), *map(str, arguments)],
        check=True,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )


def test_build_pipeline_end_to_end(tmp_path):
    """用本地小文件走一遍：建评测索引、构建两个来源、复查、汇总。"""
    pytest.importorskip("tiktoken")
    filler = "Rivers carry sediment from the mountains to the sea, shaping valleys over thousands of years. "
    documents = [f"Document {index}. " + filler * (3 + index % 4) for index in range(60)]
    documents[7] = "Study guide. " + QUESTION + " Friction is the answer. " + filler * 3
    documents[9] = documents[8].upper()
    # 第 11 篇照抄 held-out 书里连续 80 个词，第 12 篇只引用了 30 个词。
    documents[11] = "Chapter one. " + book_words(100, 180) + " " + filler * 3
    documents[12] = "A short quotation. " + book_words(10, 40) + " " + filler * 3
    (tmp_path / "heldout_books").mkdir()
    (tmp_path / "heldout_books" / "book.txt").write_text(book_words(0, 300), encoding="utf-8")
    raw = tmp_path / "raw"
    for name, rows in (("alpha", documents[:40]), ("beta", documents[30:])):
        (raw / name).mkdir(parents=True)
        with gzip.open(raw / name / "part.jsonl.gz", "wt", encoding="utf-8") as sink:
            for index, text in enumerate(rows):
                sink.write(json.dumps({"id": f"{name}-{index}", "text": text}) + "\n")
        size = (raw / name / "part.jsonl.gz").stat().st_size
        manifest = {"files": {"part.jsonl.gz": {"bytes": size, "sha256": "unused"}}}
        (raw / name / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    source = {"reader": "jsonl_gz", "revision": "test", "license": "test", "select": {}, "train_tokens": 10**6}
    spec = {
        "format": "ttt_corpus_v1",
        "tokenizer": "gpt2",
        "heldout_one_in": 3,
        "sources": [
            {"name": "alpha", "domain": "a", "heldout_tokens": 300, **source},
            {"name": "beta", "domain": "b", "heldout_tokens": 0, **source},
        ],
    }
    (tmp_path / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (tmp_path / "bench.jsonl").write_text(json.dumps({"question": QUESTION, "answer": "friction"}) + "\n")
    evals, tokens = tmp_path / "evals", tmp_path / "tokens"
    run_cli(
        "index-evals", "--jsonl", tmp_path / "bench.jsonl", "--text-dir", tmp_path / "heldout_books", "--output", evals
    )
    common = ("--spec", tmp_path / "spec.json", "--evals", evals, "--output", tokens, "--processes", 2)
    run_cli("build", *common, "--raw", raw, "--shard-tokens", 2000)
    run_cli("verify", *common, "--report", tmp_path / "verify.json")
    run_cli("report", "--spec", tmp_path / "spec.json", "--output", tokens, "--report", tmp_path / "report.json")

    alpha = json.loads((tokens / "alpha-train" / "manifest.json").read_text(encoding="utf-8"))
    beta = json.loads((tokens / "beta-train" / "manifest.json").read_text(encoding="utf-8"))
    heldout = json.loads((tokens / "alpha-heldout" / "manifest.json").read_text(encoding="utf-8"))
    assert alpha["filters"]["contaminated"] == {
        "ngram13": {f"{tmp_path.name}/bench": 1},
        "span50": {f"{tmp_path.name}/heldout_books": 1},
    }
    # alpha 内部有一篇只差大小写的重复；beta 的前 10 篇与 alpha 完全重叠，跨来源去重后全部剔除。
    assert alpha["filters"]["duplicate"] == 1
    assert beta["filters"]["duplicate"] == 10
    assert alpha["stats"]["documents"] + heldout["stats"]["documents"] == 37
    assert beta["stats"]["documents"] == 20
    assert 300 <= heldout["stats"]["tokens"] < 600
    assert len(alpha["shards"]) > 1 and alpha["vocab_size"] == 50257
    starts = np.load(tokens / "alpha-train" / "documents.npy")
    shard = np.concatenate([np.load(tokens / "alpha-train" / record["tokens"]) for record in alpha["shards"]])
    assert starts[-1] == shard.size == alpha["stats"]["tokens"]
    assert (shard[starts[:-1]] == 50256).all()
    verify = json.loads((tmp_path / "verify.json").read_text(encoding="utf-8"))
    assert verify["passed"] and verify["cross_source_duplicates"] == 0 and verify["unique_digests"] == 57
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["sources"]["beta"]["train"]["tokens"] == beta["stats"]["tokens"]

    # 载入器能直接读这些分片。
    from ttt.dataset import TokenDataset

    dataset = TokenDataset(tokens / "alpha-train" / "manifest.json", 50257)
    assert dataset.size == alpha["stats"]["tokens"]


def load_digests(directory):
    return {row.tobytes() for row in np.load(directory / "digests.npy")}


def test_kept_repeats_and_fixed_heldout_rule(tmp_path):
    """配比里重复的文档原样保留；写明 heldout_one_in 时，被抽中的文档写不进 held-out 也不进训练集。"""
    pytest.importorskip("tiktoken")
    pa = pytest.importorskip("pyarrow")
    filler = "Tides rise and fall twice a day as the moon pulls on the oceans of the turning earth. "
    documents = [f"Note {index}. " + filler * (3 + index % 4) for index in range(40)]
    raw = tmp_path / "raw"
    (raw / "mix" / "group").mkdir(parents=True)
    # 前 6 篇各出现 3 次。
    with pa.output_stream(str(raw / "mix" / "group" / "shard_0.jsonl.zst"), compression="zstd") as sink:
        sink.write("".join(json.dumps({"text": text}) + "\n" for text in documents + documents[:6] * 2).encode())
    size = (raw / "mix" / "group" / "shard_0.jsonl.zst").stat().st_size
    manifest = {"files": {"group/shard_0.jsonl.zst": {"bytes": size, "sha256": "unused"}}}
    (raw / "mix" / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    source = {
        "name": "mix",
        "domain": "a",
        "reader": "jsonl_zst",
        "sample_rate": 1.0,
        "revision": "test",
        "license": "test",
        "select": {},
        "train_tokens": 10**6,
        "heldout_tokens": 150,
        "heldout_one_in": 3,
        "keep_repeats": True,
    }
    spec = {"format": "ttt_corpus_v1", "tokenizer": "gpt2", "shard_tokens": 1000, "sources": [source]}
    (tmp_path / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    strict = {**spec, "sources": [{key: value for key, value in source.items() if key != "keep_repeats"}]}
    (tmp_path / "no_repeats.json").write_text(json.dumps(strict), encoding="utf-8")
    (tmp_path / "bench.jsonl").write_text(json.dumps({"question": QUESTION}) + "\n")
    evals, tokens = tmp_path / "evals", tmp_path / "tokens"
    run_cli("index-evals", "--jsonl", tmp_path / "bench.jsonl", "--output", evals)
    common = ("--evals", evals, "--output", tokens, "--processes", 2)
    run_cli("build", "--spec", tmp_path / "spec.json", "--raw", raw, *common)
    run_cli("verify", "--spec", tmp_path / "spec.json", *common, "--report", tmp_path / "verify.json")

    digests = [document_digest(clean_text(text)) for text in documents]
    wanted = {digest for digest in digests if int.from_bytes(digest[:4], "little") % 3 == 0}
    tripled = set(digests[:6])
    train = np.load(tokens / "mix-train" / "digests.npy")
    train_unique, heldout = load_digests(tokens / "mix-train"), load_digests(tokens / "mix-heldout")
    manifest = json.loads((tokens / "mix-train" / "manifest.json").read_text(encoding="utf-8"))
    filters = manifest["filters"]
    stats = json.loads((tokens / "mix-heldout" / "manifest.json").read_text(encoding="utf-8"))["stats"]
    # 分片大小取自 spec：除最后一个外每个分片恰好 1000 个 token。
    assert len(manifest["shards"]) > 2 and {record["count"] for record in manifest["shards"][:-1]} == {1000}
    # 被抽中的文档只可能在 held-out 里；held-out 写满（150 个 token）后其余的直接丢弃。
    assert heldout and heldout < wanted and stats["tokens"] >= 150
    assert train_unique == set(digests) - wanted
    assert len(np.load(tokens / "mix-heldout" / "digests.npy")) == len(heldout)
    copies = len(wanted) + 2 * len(tripled & wanted)
    assert filters["heldout_over_cap"] + filters.get("heldout_repeat", 0) + len(heldout) == copies
    # 训练集保留重复：前 6 篇里没被抽中的每篇有 3 份。
    repeats = 2 * len(tripled - wanted)
    assert repeats > 0 and filters["repeat_kept"] == repeats and "duplicate" not in filters
    assert len(train) == len(train_unique) + repeats
    verify = json.loads((tmp_path / "verify.json").read_text(encoding="utf-8"))
    assert verify["passed"] and verify["repeats_within_source"] == {"mix-train": repeats}
    assert verify["cross_source_duplicates"] == 0 and verify["unique_digests"] == len(train_unique) + len(heldout)
    # 来源没有写明 keep_repeats 时，训练集里出现重复文档算作错误。
    with pytest.raises(subprocess.CalledProcessError):
        run_cli("verify", "--spec", tmp_path / "no_repeats.json", *common, "--report", tmp_path / "strict.json")
    assert not json.loads((tmp_path / "strict.json").read_text(encoding="utf-8"))["passed"]


def test_zst_source_and_cross_corpus_exclusion(tmp_path):
    """zstd 来源按行抽样；第二份语料与第一份的训练集和 held-out 互不重叠。"""
    pytest.importorskip("tiktoken")
    pa = pytest.importorskip("pyarrow")
    filler = "Glaciers carve wide valleys and leave long ridges of gravel behind them as they slowly retreat. "
    documents = [f"Entry {index}. " + filler * (3 + index % 4) for index in range(30)]
    raw = tmp_path / "raw"
    (raw / "first").mkdir(parents=True)
    with gzip.open(raw / "first" / "part.jsonl.gz", "wt", encoding="utf-8") as sink:
        sink.writelines(json.dumps({"text": text}) + "\n" for text in documents[:20])
    # 第二份语料与第一份重叠 10 篇，另有 4 篇各出现 3 次（配比里的上采样）。
    rows = documents[10:] + documents[20:24] * 2
    (raw / "second" / "group-0019").mkdir(parents=True)
    with pa.output_stream(str(raw / "second" / "group-0019" / "shard_0.jsonl.zst"), compression="zstd") as sink:
        sink.write("".join(json.dumps({"id": str(row), "text": text}) + "\n" for row, text in enumerate(rows)).encode())
    for name, path in (("first", "part.jsonl.gz"), ("second", "group-0019/shard_0.jsonl.zst")):
        manifest = {"files": {path: {"bytes": (raw / name / path).stat().st_size, "sha256": "unused"}}}
        (raw / name / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    base = {"revision": "test", "license": "test", "select": {}, "train_tokens": 10**6, "heldout_tokens": 600}
    specs = {
        "first": {"heldout_one_in": 3, "sources": [{"name": "first", "domain": "a", "reader": "jsonl_gz", **base}]},
        "second": {
            "heldout_one_in": 2,
            "sources": [{"name": "second", "domain": "b", "reader": "jsonl_zst", "sample_rate": 1.0, **base}],
        },
    }
    for name, spec in specs.items():
        payload = {"format": "ttt_corpus_v1", "tokenizer": "gpt2", **spec}
        (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "bench.jsonl").write_text(json.dumps({"question": QUESTION}) + "\n")
    evals, first, second = tmp_path / "evals", tmp_path / "tokens_first", tmp_path / "tokens_second"
    run_cli("index-evals", "--jsonl", tmp_path / "bench.jsonl", "--output", evals)
    common = ("--evals", evals, "--processes", 2)
    target = ("--spec", tmp_path / "second.json", "--output", second, *common)
    report = tmp_path / "verify.json"
    run_cli("build", "--spec", tmp_path / "first.json", "--raw", raw, "--output", first, *common)
    run_cli("build", *target, "--raw", raw, "--exclude", first)
    run_cli("verify", *target, "--report", report, "--exclude", first)

    first_train, first_heldout = load_digests(first / "first-train"), load_digests(first / "first-heldout")
    second_train, second_heldout = load_digests(second / "second-train"), load_digests(second / "second-heldout")
    shared = {document_digest(clean_text(text)) for text in documents[10:20]}
    fresh = {document_digest(clean_text(text)) for text in documents[20:]}
    manifest = json.loads((second / "second-train" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["excluded_corpora"] == ["tokens_first"]
    assert manifest["filters"]["duplicate"] == 8
    # 第一份语料 held-out 里的文档不进第二份；第一份训练集里的文档只能进第二份的训练集。
    assert 0 < len(shared & first_heldout) == manifest["filters"]["in_other_heldout"]
    assert not (second_train | second_heldout) & first_heldout
    assert second_heldout and second_heldout <= fresh
    assert shared & first_train <= second_train
    assert len(second_train) + len(second_heldout) == 20 - len(shared & first_heldout)
    names = (second / "second-train" / "documents.txt").read_text(encoding="utf-8").split()
    assert all(name.startswith("group-0019/shard_0.jsonl.zst:") for name in names)
    verify = json.loads(report.read_text(encoding="utf-8"))
    assert verify["passed"]
    assert verify["excluded_corpora"] == {
        "roots": ["tokens_first"],
        "train_in_other_heldout": 0,
        "heldout_in_other_train": 0,
    }
