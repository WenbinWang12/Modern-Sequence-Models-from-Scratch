"""分片边界、监督掩码和数据篡改检查。"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from ttt import dataset as dataset_module
from ttt.data import BatchStream, ByteTokenizer, prepare_data
from ttt.dataset import TokenDataset, file_sha256, prepare_sharded_data


def test_supervised_shards_and_resume(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps({"prompt": "Q:", "response": "ABCDE"}) + "\n", encoding="utf-8")
    path = prepare_sharded_data(source, tmp_path / "dataset", ByteTokenizer(), shard_tokens=3, supervised=True)
    dataset = TokenDataset(path, 259)
    tokens, masks = dataset.rows([1], 6)
    np.testing.assert_array_equal(tokens[0], ByteTokenizer().encode("Q:ABC", add_bos=False) + [ord("D") + 3])
    np.testing.assert_array_equal(masks[0], [0, 1, 1, 1, 1])
    stream = BatchStream(path, sequence_length=4, batch_size=2, vocab_size=259, seed=9)
    state = stream.state_dict()
    expected = stream.next_batch_with_masks()
    stream.load_state_dict(state)
    for actual, reference in zip(stream.next_batch_with_masks(), expected):
        np.testing.assert_array_equal(actual, reference)
    manifest = json.loads(path.read_text())
    shard = path.parent / manifest["shards"][0]["tokens"]
    with shard.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\xff")
    with pytest.raises(ValueError, match="校验"):
        TokenDataset(path, 259)


def test_global_rows_independent_of_partition(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(100, dtype=np.int32) % 19)
    stream = BatchStream(path, sequence_length=8, batch_size=4, vocab_size=19, seed=31)
    full = stream.sample_rows(7, [0, 1, 2, 3])
    halves = [stream.sample_rows(7, [0, 1]), stream.sample_rows(7, [2, 3])]
    for index in range(2):
        np.testing.assert_array_equal(full[index], np.concatenate([item[index] for item in halves]))


def write_shards(directory, arrays, masks=None):
    """把若干 uint16 数组写成分片和清单，返回清单路径。"""
    records = []
    for index, array in enumerate(arrays):
        np.save(directory / f"part-{index}.npy", array, allow_pickle=False)
        record = {"tokens": f"part-{index}.npy", "count": int(array.size)}
        record["sha256"] = file_sha256(directory / f"part-{index}.npy")
        if masks is not None:
            np.save(directory / f"part-{index}.mask.npy", masks[index], allow_pickle=False)
            record["mask"] = f"part-{index}.mask.npy"
            record["mask_sha256"] = file_sha256(directory / f"part-{index}.mask.npy")
        records.append(record)
    manifest = {"format_version": 1, "vocab_size": 300, "shards": records}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory / "manifest.json"


def test_rows_match_concatenated_tokens_across_shard_boundaries(tmp_path):
    rng = np.random.default_rng(0)
    arrays = [rng.integers(0, 300, size=size, dtype=np.uint16) for size in (7, 1, 12, 3, 40)]
    masks = [rng.integers(0, 2, size=array.size, dtype=np.uint8) for array in arrays]
    dataset = TokenDataset(write_shards(tmp_path, arrays, masks), 300)
    tokens, labels = np.concatenate(arrays), np.concatenate(masks)
    assert dataset.size == tokens.size == 63
    starts = [0, 5, 6, 7, 8, 19, 22, 23, 50]
    rows, row_masks = dataset.rows(starts, 13)
    assert rows.dtype == np.int32 and row_masks.dtype == np.float32
    for row, start in enumerate(starts):
        np.testing.assert_array_equal(rows[row], tokens[start : start + 13])
        np.testing.assert_array_equal(row_masks[row], labels[start + 1 : start + 13])
    whole, _ = dataset.rows(np.asarray([0]), 63)
    np.testing.assert_array_equal(whole[0], tokens)
    for bad in ([51], [-1], [63]):
        with pytest.raises(ValueError, match="越界"):
            dataset.rows(bad, 13)


def test_verification_receipt_skips_rehash_but_detects_changes(tmp_path, monkeypatch):
    arrays = [np.arange(10, dtype=np.uint16), np.arange(10, 30, dtype=np.uint16)]
    manifest = write_shards(tmp_path, arrays)
    calls = []
    original = dataset_module.file_sha256
    monkeypatch.setattr(dataset_module, "file_sha256", lambda path: calls.append(Path(path).name) or original(path))

    TokenDataset(manifest, 300)
    assert sorted(calls) == ["manifest.json", "part-0.npy", "part-1.npy"]
    assert (tmp_path / "verified.json").exists()
    calls.clear()
    reopened = TokenDataset(manifest, 300)
    # 回执有效时只对清单本身取摘要，不再读分片内容。
    assert calls == ["manifest.json"]
    np.testing.assert_array_equal(reopened.rows([8], 4)[0][0], [8, 9, 10, 11])
    # Windows 不允许在仍有 mmap 句柄时覆盖同一个分片。
    del reopened

    # 分片被改写（大小不变）后回执失效，重新校验并报错。
    changed = arrays[1].copy()
    changed[3] = 299
    np.save(tmp_path / "part-1.npy", changed, allow_pickle=False)
    os.utime(tmp_path / "part-1.npy", ns=(1, 1))
    calls.clear()
    with pytest.raises(ValueError, match="分片校验失败：part-1.npy"):
        TokenDataset(manifest, 300)
    assert "part-1.npy" in calls

    # 词表大小不同的回执不能复用。
    np.save(tmp_path / "part-1.npy", arrays[1], allow_pickle=False)
    TokenDataset(manifest, 300)
    wide = json.loads(manifest.read_text(encoding="utf-8"))
    receipt = json.loads((tmp_path / "verified.json").read_text(encoding="utf-8"))
    assert receipt["vocab_size"] == 300 and set(receipt["files"]) == {"part-0.npy", "part-1.npy"}
    wide["vocab_size"] = 20
    manifest.write_text(json.dumps(wide), encoding="utf-8")
    with pytest.raises(ValueError, match="超出词表"):
        TokenDataset(manifest, 20)


def test_source_digest_covers_actual_bytes_including_crlf_and_blank_lines(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_bytes(b'{"text":"A"}\r\n\r\n{"text":"B"}\n')
    manifest_path = prepare_sharded_data(source, tmp_path / "dataset", ByteTokenizer(), shard_tokens=3)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["input_sha256"] == file_sha256(source)
    dataset = TokenDataset(manifest_path, 259)
    tokens, _ = dataset.rows([0], 6)
    np.testing.assert_array_equal(tokens[0], [1, 68, 2, 1, 69, 2])


def test_source_changed_during_encoding_does_not_publish_manifest(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_text('{"text":"A"}\n', encoding="utf-8")

    class MutatingTokenizer(ByteTokenizer):
        def encode(self, text, *, add_bos=True):
            result = super().encode(text, add_bos=add_bos)
            source.write_text('{"text":"Z"}\n', encoding="utf-8")
            return result

    output = tmp_path / "dataset"
    with pytest.raises(RuntimeError, match="输入语料在分片准备期间发生变化"):
        prepare_sharded_data(source, output, MutatingTokenizer(), shard_tokens=3)
    assert not (output / "manifest.json").exists()


def test_consumed_digest_detects_changed_then_restored_source(tmp_path, monkeypatch):
    source = tmp_path / "records.jsonl"
    original = b'{"text":"A"}\n'
    source.write_bytes(original)
    original_digest = file_sha256(source)
    changed = False
    consumed_texts = []

    def change_after_initial_digest(path):
        nonlocal changed
        digest = file_sha256(path)
        if path == source and not changed:
            source.write_bytes(b'{"text":"Z"}\n')
            changed = True
        return digest

    class RestoringTokenizer(ByteTokenizer):
        def encode(self, text, *, add_bos=True):
            consumed_texts.append(text)
            result = super().encode(text, add_bos=add_bos)
            source.write_bytes(original)
            return result

    # 前后两次源摘要都为 A，但实际编码的是 Z；必须由消费摘要发现不一致。
    monkeypatch.setattr(dataset_module, "file_sha256", change_after_initial_digest)
    output = tmp_path / "dataset"
    with pytest.raises(RuntimeError, match="输入语料在分片准备期间发生变化"):
        prepare_sharded_data(source, output, RestoringTokenizer(), shard_tokens=3)
    assert consumed_texts == ["Z"]
    assert file_sha256(source) == original_digest
    assert not (output / "manifest.json").exists()


def test_single_npy_and_sharded_paths_have_distinct_document_policies(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_text('{"text":"A"}\n{"text":"B"}\n', encoding="utf-8")
    plain = tmp_path / "tokens.npy"
    prepare_data(source, plain, ByteTokenizer())
    sharded = prepare_sharded_data(source, tmp_path / "dataset", ByteTokenizer(), shard_tokens=3)
    plain_tokens, plain_mask = TokenDataset(plain, 259).rows([0], 6)
    sharded_tokens, sharded_mask = TokenDataset(sharded, 259).rows([0], 6)
    np.testing.assert_array_equal(plain_tokens, sharded_tokens)
    np.testing.assert_array_equal(plain_tokens[0], [1, 68, 2, 1, 69, 2])
    np.testing.assert_array_equal(plain_mask[0], [1, 1, 1, 1, 1])
    np.testing.assert_array_equal(sharded_mask[0], [1, 1, 0, 1, 1])

    # 空文本在单文件入口保留 BOS/EOS，在分片入口被跳过；这是政策差别。
    source.write_text('{"text":""}\n{"text":"A"}\n', encoding="utf-8")
    plain_with_empty = tmp_path / "with-empty.npy"
    prepare_data(source, plain_with_empty, ByteTokenizer())
    sharded_without_empty = prepare_sharded_data(source, tmp_path / "without-empty", ByteTokenizer())
    np.testing.assert_array_equal(np.load(plain_with_empty, allow_pickle=False), [1, 2, 1, 68, 2])
    manifest = json.loads(sharded_without_empty.read_text(encoding="utf-8"))
    assert manifest["stats"]["empty_documents"] == 1
    assert TokenDataset(sharded_without_empty, 259).size == 3
