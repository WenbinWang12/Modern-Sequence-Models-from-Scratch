"""检查教学 BPE 的手算结果及正式 tokenizer 的训练外字节覆盖。"""

import argparse
import hashlib
import importlib.util
import json
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module(relative_path):
    spec = importlib.util.spec_from_file_location(Path(relative_path).stem, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hand_counted_bpe_and_frozen_encoding():
    bpe = _load_module("examples/book/bpe_walkthrough.py")
    model = bpe.learn_bpe(bpe.WORD_COUNTS, 5)
    assert [step["pair"] for step in model["steps"]] == [
        ("c", "a"), ("ca", "t"), ("cat", "</w>"), ("ca", "p"), ("cap", "</w>"),
    ]
    assert [step["pair_count"] for step in model["steps"]] == [11, 7, 5, 4, 3]
    assert [step["positions_after"] for step in model["steps"]] == [36, 29, 24, 20, 17]
    assert model["initial_positions"] == 47
    assert len(model["vocabulary"]) == 11
    saved_model = json.dumps(model, sort_keys=True)
    catt = bpe.encode_word("catt", model)
    catcap = bpe.encode_word("catcap", model)
    assert catt["tokens"] == ("cat", "t", "</w>")
    assert catt["ids"] == [7, 5, 0]
    assert [step["rank"] for step in catt["trace"]] == [0, 1]
    assert catcap["tokens"] == ("cat", "cap</w>")
    assert [step["rank"] for step in catcap["trace"]] == [0, 1, 3, 4]
    assert json.dumps(model, sort_keys=True) == saved_model
    # 输入词频表的插入顺序不得改变同频裁决。
    reverse_model = bpe.learn_bpe(dict(reversed(list(bpe.WORD_COUNTS.items()))), 5)
    assert reverse_model == model


def test_overlapping_pairs_do_not_double_merge():
    bpe = _load_module("examples/book/bpe_walkthrough.py")
    model = bpe.learn_bpe({"aaa": 2}, 1)
    assert model["steps"][0]["pair_count"] == 4
    assert model["steps"][0]["after"]["aaa"] == ("aa", "a", "</w>")
    assert model["initial_positions"] - model["final_positions"] == 2
    with pytest.raises(ValueError, match="训练字母表"):
        bpe.encode_word("猫", model)


def test_teaching_report_refuses_overwrite(tmp_path, monkeypatch):
    bpe = _load_module("examples/book/bpe_walkthrough.py")
    report = tmp_path / "bpe.json"
    report.write_bytes(b"previous report\n")
    monkeypatch.setattr("sys.argv", ["bpe_walkthrough.py", "--output", str(report)])
    with pytest.raises(FileExistsError):
        bpe.main()
    assert report.read_bytes() == b"previous report\n"


@pytest.mark.parametrize("algorithm", ["bpe", "unigram"])
def test_training_retains_unseen_bytes_and_normalized_roundtrip(tmp_path, algorithm):
    pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    training = _load_module("scripts/train_tokenizer.py")
    source = tmp_path / "ascii.jsonl"
    source.write_text(json.dumps({"text": "cat cats cap caps " * 10}) + "\n", encoding="utf-8")
    args = argparse.Namespace(input=[source], output=tmp_path / algorithm, algorithm=algorithm,
                              vocab_size=280, min_frequency=2, max_piece_length=16, text_key="text", byte_level=True)
    output = training.train(args)
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(output, local_files_only=True)
    # 训练集只有 ASCII；覆盖中文、emoji、控制空白以及有损的 NFKC 变化。
    for text in ("Ａ① 猫🙂\n\té", "  unseen\tpunctuation!?\n", "汉字𠮷"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert tokenizer.unk_token_id not in ids
        assert tokenizer.decode(ids, clean_up_tokenization_spaces=False) == unicodedata.normalize("NFKC", text)
    from tokenizers import pre_tokenizers

    assert set(pre_tokenizers.ByteLevel.alphabet()).issubset(tokenizer.get_vocab())
    metadata = json.loads((output / "tokenizer_metadata.json").read_text(encoding="utf-8"))
    assert metadata["initial_alphabet_size"] == 256
    assert metadata["vocab_size_actual"] == len(tokenizer)
    assert metadata["tokenizer_sha256"] == hashlib.sha256((output / "tokenizer.json").read_bytes()).hexdigest()
    assert metadata["min_frequency"] == (2 if algorithm == "bpe" else None)
    assert metadata["max_piece_length"] == (16 if algorithm == "unigram" else None)
    assert len(set(metadata["special_token_ids"].values())) == 4
    assert metadata["inputs"][0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    expected_training_text = ("cat cats cap caps " * 10 + "\n").encode("utf-8")
    assert metadata["training_texts"][0]["sha256"] == hashlib.sha256(expected_training_text).hexdigest()
    previous_files = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    with pytest.raises(FileExistsError):
        training.train(args)
    assert {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()} == previous_files
    args.vocab_size = 259
    args.output = tmp_path / "too-small"
    with pytest.raises(ValueError, match="至少为 260"):
        training.train(args)
    assert not args.output.exists()


def test_training_rejects_source_changed_after_jsonl_conversion(tmp_path, monkeypatch):
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    training = _load_module("scripts/train_tokenizer.py")
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({"text": "cat cap cats caps"}) + "\n", encoding="utf-8")
    convert = training._write_training_text

    def convert_then_change_source(inputs, text_key, temporary):
        files = convert(inputs, text_key, temporary)
        source.write_bytes(source.read_bytes() + b"\n")
        return files

    monkeypatch.setattr(training, "_write_training_text", convert_then_change_source)
    args = argparse.Namespace(input=[source], output=tmp_path / "tokenizer", algorithm="bpe",
                              vocab_size=280, min_frequency=2, max_piece_length=16, text_key="text", byte_level=True)
    with pytest.raises(RuntimeError, match="输入语料在训练期间发生变化"):
        training.train(args)
    assert not args.output.exists()
