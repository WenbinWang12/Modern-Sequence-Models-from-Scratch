"""从文本或 JSONL 语料训练并保存 BPE/Unigram tokenizer。"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from importlib.metadata import version
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_training_text(inputs: list[Path], text_key: str, temp_dir: Path) -> list[Path]:
    training_files: list[Path] = []
    for index, input_path in enumerate(inputs):
        if input_path.suffix.lower() != ".jsonl":
            training_files.append(input_path)
            continue
        output_path = temp_dir / f"input-{index:04d}.txt"
        with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get(text_key) if isinstance(record, dict) else None
                if not isinstance(text, str):
                    raise ValueError(f"{input_path}:{line_number} 的 {text_key!r} 必须是字符串。")
                target.write(text.replace("\r\n", "\n").replace("\r", "\n"))
                target.write("\n")
        training_files.append(output_path)
    return training_files


def train(args: argparse.Namespace) -> Path:
    try:
        from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
        from transformers import PreTrainedTokenizerFast
    except ImportError as error:
        raise SystemExit("请先安装 tokenizer 依赖：uv sync --extra tokenizer，或 pip install transformers tokenizers。") from error

    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    if args.algorithm not in ("bpe", "unigram"):
        raise ValueError("algorithm 必须是 bpe 或 unigram。")
    # ByteLevel 的字符映射不等于覆盖保证；两种 trainer 都需要保留完整字节表。
    initial_alphabet = sorted(pre_tokenizers.ByteLevel.alphabet()) if args.byte_level else []
    minimum_vocab_size = len(initial_alphabet) + len(special_tokens)
    if args.vocab_size < minimum_vocab_size:
        raise ValueError(f"vocab_size 至少为 {minimum_vocab_size}，才能容纳基础字符和特殊 token。")
    if args.min_frequency < 1 or args.max_piece_length < 1:
        raise ValueError("min_frequency 和 max_piece_length 必须是正整数。")
    output_dir = args.output.resolve()
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，请为本次训练选择新目录：{output_dir}")

    if args.algorithm == "bpe":
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        trainer = trainers.BpeTrainer(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=special_tokens,
            initial_alphabet=initial_alphabet,
        )
    else:
        tokenizer = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(
            vocab_size=args.vocab_size,
            special_tokens=special_tokens,
            initial_alphabet=initial_alphabet,
            unk_token="<unk>",
            max_piece_length=args.max_piece_length,
        )

    tokenizer.normalizer = normalizers.NFKC()
    if args.byte_level:
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    input_paths = [path.resolve() for path in args.input]
    input_records = [{"path": str(path), "sha256": _sha256(path)} for path in input_paths]
    with tempfile.TemporaryDirectory(prefix="ttt-tokenizer-") as temp_name:
        training_files = _write_training_text(input_paths, args.text_key, Path(temp_name))
        training_text_records = [
            {"input": str(input_path), "sha256": _sha256(training_path)}
            for input_path, training_path in zip(input_paths, training_files)
        ]
        tokenizer.train([str(path) for path in training_files], trainer=trainer)
        for path, record in zip(training_files, training_text_records):
            if _sha256(path) != record["sha256"]:
                raise RuntimeError(f"tokenizer 训练文本在训练期间发生变化：{path}")
    for path, record in zip(input_paths, input_records):
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"输入语料在训练期间发生变化：{path}")

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        clean_up_tokenization_spaces=False,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    fast_tokenizer.save_pretrained(output_dir)

    metadata = {
        "algorithm": args.algorithm,
        "byte_level": args.byte_level,
        "vocab_size_requested": args.vocab_size,
        "vocab_size_actual": len(fast_tokenizer),
        "min_frequency": args.min_frequency if args.algorithm == "bpe" else None,
        "max_piece_length": args.max_piece_length if args.algorithm == "unigram" else None,
        "text_key": args.text_key,
        "normalizer": "NFKC",
        "pre_tokenizer": "ByteLevel" if args.byte_level else "Whitespace",
        "initial_alphabet_size": len(initial_alphabet),
        "decoder": "ByteLevel" if args.byte_level else None,
        "special_tokens": special_tokens,
        "special_token_ids": {token: fast_tokenizer.convert_tokens_to_ids(token) for token in special_tokens},
        "tokenizer_sha256": _sha256(output_dir / "tokenizer.json"),
        "library_versions": {name: version(name) for name in ("tokenizers", "transformers")},
        "inputs": input_records,
        "training_texts": training_text_records,
    }
    (output_dir / "tokenizer_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="训练并保存 BPE/Unigram tokenizer。")
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="纯文本或 JSONL 训练语料。")
    parser.add_argument("--output", type=Path, required=True, help="Hugging Face tokenizer 输出目录。")
    parser.add_argument("--algorithm", choices=("bpe", "unigram"), default="bpe")
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--min-frequency", type=int, default=2, help="仅 BPE 使用的相邻 pair 最小频次。")
    parser.add_argument("--max-piece-length", type=int, default=16, help="仅 Unigram 使用的候选片段长度上限。")
    parser.add_argument("--text-key", default="text", help="JSONL 文本字段名。")
    parser.add_argument("--byte-level", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    output_dir = train(args)
    print(json.dumps({"output": str(output_dir), "metadata": str(output_dir / "tokenizer_metadata.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
