"""分片边界、监督掩码和数据篡改检查。"""

import json

import numpy as np
import pytest

from ttt.data import BatchStream, ByteTokenizer
from ttt.dataset import TokenDataset, prepare_sharded_data


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
