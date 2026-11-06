"""文本分词、数据准备与可恢复的采样器。"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .dataset import TokenDataset, prepare_sharded_data


class ByteTokenizer:
    vocab_size = 259
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, *, add_bos=True):
        tokens = [value + 3 for value in text.encode("utf-8")]
        return ([self.bos_token_id] if add_bos else []) + tokens

    def decode(self, tokens):
        return bytes(int(token) - 3 for token in tokens if 3 <= int(token) < 259).decode("utf-8", errors="replace")


class HuggingFaceTokenizer:
    def __init__(self, name):
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise ImportError("请安装 tokenizer 扩展：pip install -e '.[tokenizer]'。") from error
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=False)
        self.name = name
        self.vocab_size = len(self.tokenizer)
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    def encode(self, text, *, add_bos=True):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if add_bos and self.bos_token_id is not None:
            tokens = [self.bos_token_id] + tokens
        return tokens

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


def load_tokenizer(name):
    return ByteTokenizer() if name == "byte" else HuggingFaceTokenizer(name)


class BatchStream:
    def __init__(self, path, *, sequence_length, batch_size, vocab_size, seed=0):
        self.dataset = TokenDataset(path, vocab_size)
        if self.dataset.size < sequence_length + 1:
            raise ValueError("数据长度不足一个训练样本。")
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.fingerprint = self.dataset.fingerprint

    def next_batch(self):
        return self.next_batch_with_masks()[0]

    def next_batch_with_masks(self):
        starts = self.rng.integers(0, self.dataset.size - self.sequence_length, size=self.batch_size)
        return self.dataset.rows(starts, self.sequence_length + 1)

    def sample_rows(self, step, indices):
        starts = []
        for index in indices:
            rng = np.random.default_rng(np.random.SeedSequence((self.seed, int(step), int(index))))
            starts.append(int(rng.integers(0, self.dataset.size - self.sequence_length)))
        return self.dataset.rows(starts, self.sequence_length + 1)

    def state_dict(self):
        return {
            "rng": self.rng.bit_generator.state,
            "sha256": self.fingerprint,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }

    def load_state_dict(self, state):
        expected = (self.fingerprint, self.sequence_length, self.batch_size, self.seed)
        actual = (state["sha256"], state["sequence_length"], state["batch_size"], state.get("seed", self.seed))
        if actual != expected:
            raise ValueError("恢复训练的数据内容或采样配置与 checkpoint 不一致。")
        self.rng.bit_generator.state = state["rng"]


def prepare_data(input_path, output_path, tokenizer, *, text_key="text"):
    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open(encoding="utf-8") as source, tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as raw:
        documents = (
            (json.loads(line)[text_key] for line in source if line.strip())
            if input_path.suffix == ".jsonl"
            else (source.read(),)
        )
        for document in documents:
            if not isinstance(document, str):
                raise ValueError("文本字段必须是字符串。")
            tokens = tokenizer.encode(document)
            if tokenizer.eos_token_id is not None:
                tokens.append(tokenizer.eos_token_id)
            raw.write(np.asarray(tokens, dtype=np.int32).tobytes())
            count += len(tokens)
        if count < 2:
            raise ValueError("分词结果不足两个 token。")
        raw.seek(0)
        with output_path.open("wb") as output:
            np.lib.format.write_array_header_1_0(
                output,
                {"descr": np.dtype(np.int32).str, "fortran_order": False, "shape": (count,)},
            )
            shutil.copyfileobj(raw, output)
    return count


def main():
    parser = argparse.ArgumentParser(description="准备 TTT 训练数据")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default="byte")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--sharded", action="store_true")
    parser.add_argument("--supervised", action="store_true", help="JSONL 使用 prompt/response 字段")
    parser.add_argument("--shard-tokens", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.sharded or args.supervised:
        manifest = prepare_sharded_data(
            args.input,
            args.output,
            load_tokenizer(args.tokenizer),
            shard_tokens=args.shard_tokens,
            supervised=args.supervised,
        )
        print(json.dumps({"manifest": str(manifest)}, ensure_ascii=False))
        return
    count = prepare_data(args.input, args.output, load_tokenizer(args.tokenizer), text_key=args.text_key)
    print(json.dumps({"tokens": count, "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
