"""分片边界、监督掩码和数据篡改检查。"""

import json

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
