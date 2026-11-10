"""从文本或 JSONL 语料训练并保存 BPE/Unigram tokenizer。"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
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
                text = record.get(text_key)
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

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

    if args.algorithm == "bpe":
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        trainer = trainers.BpeTrainer(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=special_tokens,
        )
    else:
        tokenizer = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(
            vocab_size=args.vocab_size,
            special_tokens=special_tokens,
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
    with tempfile.TemporaryDirectory(prefix="ttt-tokenizer-") as temp_name:
        training_files = _write_training_text(input_paths, args.text_key, Path(temp_name))
        tokenizer.train([str(path) for path in training_files], trainer=trainer)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    fast_tokenizer.save_pretrained(output_dir)

    metadata = {
        "algorithm": args.algorithm,
        "byte_level": args.byte_level,
        "vocab_size_requested": args.vocab_size,
        "vocab_size_actual": len(fast_tokenizer),
        "min_frequency": args.min_frequency,
        "max_piece_length": args.max_piece_length,
        "text_key": args.text_key,
        "special_tokens": special_tokens,
        "inputs": [{"path": str(path), "sha256": _sha256(path)} for path in input_paths],
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
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-piece-length", type=int, default=16)
    parser.add_argument("--text-key", default="text", help="JSONL 文本字段名。")
    parser.add_argument("--byte-level", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    output_dir = train(args)
    print(json.dumps({"output": str(output_dir), "metadata": str(output_dir / "tokenizer_metadata.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
