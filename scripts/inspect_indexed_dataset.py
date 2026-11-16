"""只读检查 Megatron 风格的 MMIDIDX 数据集。

这个脚本只读取索引中的少量记录和对应的 token，不会扫描或修改整个数据集。
它适合在远端机器上确认 .idx 与 .bin 的布局，尤其适合排查把二进制文件直接
用编辑器打开后看到乱码的情况。
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import BinaryIO

__all__ = ["inspect_indexed_dataset", "main"]


_MAGIC = b"MMIDIDX\x00\x00"
_HEADER_SIZE = 34

# Megatron indexed_dataset.py 使用的 dtype 编号。第二项是 struct 的格式字符，
# 第三项是每个元素占用的字节数。所有字段都按 little-endian 读取。
_DTYPES: dict[int, tuple[str, str, int]] = {
    1: ("uint8", "B", 1),
    2: ("int8", "b", 1),
    3: ("int16", "h", 2),
    4: ("int32", "i", 4),
    5: ("int64", "q", 8),
    6: ("float64", "d", 8),
    7: ("float32", "f", 4),
    8: ("uint16", "H", 2),
}


class IndexedDatasetError(ValueError):
    """表示索引或二进制数据的格式不一致。"""


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    """从当前位置读取指定字节数，不足时给出明确错误。"""

    data = handle.read(size)
    if len(data) != size:
        raise IndexedDatasetError(
            f"文件提前结束：需要 {size} 字节，实际读取 {len(data)} 字节"
        )
    return data


def _read_array(
    handle: BinaryIO,
    *,
    offset: int,
    count: int,
    format_code: str,
) -> list[int]:
    """从索引文件中读取一个连续的小数组。"""

    if count <= 0:
        return []
    item_size = struct.calcsize("<" + format_code)
    handle.seek(offset)
    raw = _read_exact(handle, item_size * count)
    return list(struct.unpack(f"<{count}{format_code}", raw))


def _read_tokens(
    handle: BinaryIO,
    *,
    offset: int,
    count: int,
    format_code: str,
) -> list[int | float]:
    """读取一段 token；调用方已经保证了范围落在 bin 文件内。"""

    if count <= 0:
        return []
    item_size = struct.calcsize("<" + format_code)
    handle.seek(offset)
    raw = _read_exact(handle, item_size * count)
    return list(struct.unpack(f"<{count}{format_code}", raw))


def _resolve_paths(
    prefix: Path | None,
    idx_path: Path | None,
    bin_path: Path | None,
) -> tuple[Path, Path]:
    """解析位置参数或显式路径，并避免把两种写法混用成歧义。"""

    if prefix is not None:
        if idx_path is not None or bin_path is not None:
            raise IndexedDatasetError("不能同时使用位置参数 prefix 和 --idx/--bin")
        return prefix.with_suffix(".idx"), prefix.with_suffix(".bin")
    if idx_path is None or bin_path is None:
        raise IndexedDatasetError("请提供 prefix，或同时提供 --idx 与 --bin")
    return idx_path, bin_path


def inspect_indexed_dataset(
    idx_path: Path,
    bin_path: Path,
    *,
    sequence_limit: int = 3,
    token_limit: int = 16,
    document_limit: int = 8,
    vocab_size: int | None = None,
) -> dict[str, object]:
    """读取并返回一个小型、可序列化的数据格式报告。

    sequence_limit、token_limit 和 document_limit 只控制报告大小。
    索引文件的大小由头部计数决定，脚本不会为了生成报告把整个 token 文件读入内存。
    """

    if sequence_limit < 0 or token_limit < 0 or document_limit < 0:
        raise IndexedDatasetError("三个 limit 都必须是非负整数")
    if vocab_size is not None and vocab_size <= 0:
        raise IndexedDatasetError("vocab_size 必须是正整数")
    idx_path = idx_path.expanduser().resolve()
    bin_path = bin_path.expanduser().resolve()
    if not idx_path.is_file():
        raise IndexedDatasetError(f"找不到索引文件：{idx_path}")
    if not bin_path.is_file():
        raise IndexedDatasetError(f"找不到二进制文件：{bin_path}")

    with idx_path.open("rb") as index_file:
        if _read_exact(index_file, len(_MAGIC)) != _MAGIC:
            raise IndexedDatasetError("索引文件不是 MMIDIDX 格式")
        version = struct.unpack("<Q", _read_exact(index_file, 8))[0]
        dtype_code = struct.unpack("<B", _read_exact(index_file, 1))[0]
        sequence_count = struct.unpack("<Q", _read_exact(index_file, 8))[0]
        document_count = struct.unpack("<Q", _read_exact(index_file, 8))[0]
        if dtype_code not in _DTYPES:
            raise IndexedDatasetError(f"不支持的 dtype 编号：{dtype_code}")
        dtype_name, format_code, item_size = _DTYPES[dtype_code]

        # 数组顺序与 Megatron 的 MMIDIDX 约定一致：
        # header -> sequence lengths(int32) -> byte pointers(int64) -> doc indices(int64)。
        lengths_offset = _HEADER_SIZE
        pointers_offset = lengths_offset + sequence_count * 4
        documents_offset = pointers_offset + sequence_count * 8
        expected_index_bytes = documents_offset + document_count * 8
        actual_index_bytes = idx_path.stat().st_size
        if actual_index_bytes < expected_index_bytes:
            raise IndexedDatasetError(
                "索引文件长度不足："
                f"需要至少 {expected_index_bytes} 字节，实际 {actual_index_bytes} 字节"
            )

        sampled_sequences = min(sequence_limit, sequence_count)
        lengths = _read_array(
            index_file,
            offset=lengths_offset,
            count=sampled_sequences,
            format_code="i",
        )
        pointers = _read_array(
            index_file,
            offset=pointers_offset,
            count=sampled_sequences,
            format_code="q",
        )
        last_sequence: dict[str, int] | None = None
        if sequence_count:
            last_index = sequence_count - 1
            last_length = _read_array(
                index_file,
                offset=lengths_offset + last_index * 4,
                count=1,
                format_code="i",
            )[0]
            last_pointer = _read_array(
                index_file,
                offset=pointers_offset + last_index * 8,
                count=1,
                format_code="q",
            )[0]
            last_sequence = {
                "index": last_index,
                "length": last_length,
                "offset_bytes": last_pointer,
            }

        sampled_documents = min(document_limit, document_count)
        first_documents = _read_array(
            index_file,
            offset=documents_offset,
            count=sampled_documents,
            format_code="q",
        )
        last_documents: list[int] = []
        if sampled_documents:
            last_documents = _read_array(
                index_file,
                offset=documents_offset + (document_count - sampled_documents) * 8,
                count=sampled_documents,
                format_code="q",
            )

    bin_bytes = bin_path.stat().st_size
    sequences: list[dict[str, object]] = []
    with bin_path.open("rb") as data_file:
        for index, (length, pointer) in enumerate(zip(lengths, pointers)):
            if length < 0 or pointer < 0:
                raise IndexedDatasetError(
                    f"第 {index} 条序列的 length/pointer 不能为负数：{length}, {pointer}"
                )
            if pointer % item_size:
                raise IndexedDatasetError(
                    f"第 {index} 条序列的字节 offset 未按 dtype 对齐：{pointer}"
                )
            end = pointer + length * item_size
            if end > bin_bytes:
                raise IndexedDatasetError(
                    f"第 {index} 条序列超出 bin 文件：[{pointer}, {end}) > {bin_bytes}"
                )
            first_count = min(token_limit, length)
            last_count = min(token_limit, length)
            first_tokens = _read_tokens(
                data_file,
                offset=pointer,
                count=first_count,
                format_code=format_code,
            )
            last_tokens = first_tokens
            if length > token_limit:
                last_tokens = _read_tokens(
                    data_file,
                    offset=end - last_count * item_size,
                    count=last_count,
                    format_code=format_code,
                )
            sequences.append(
                {
                    "index": index,
                    "length": length,
                    "offset_bytes": pointer,
                    "end_bytes": end,
                    "first_tokens": first_tokens,
                    "last_tokens": last_tokens,
                }
            )
        if last_sequence is not None:
            last_length = last_sequence["length"]
            last_pointer = last_sequence["offset_bytes"]
            if last_length < 0 or last_pointer < 0:
                raise IndexedDatasetError(
                    "最后一条序列的 length/pointer 不能为负数："
                    f"{last_length}, {last_pointer}"
                )
            if last_pointer % item_size:
                raise IndexedDatasetError(
                    "最后一条序列的字节 offset 未按 dtype 对齐："
                    f"{last_pointer}"
                )
            last_end = last_pointer + last_length * item_size
            if last_end > bin_bytes:
                raise IndexedDatasetError(
                    "最后一条序列超出 bin 文件："
                    f"[{last_pointer}, {last_end}) > {bin_bytes}"
                )
            last_sequence["end_bytes"] = last_end

    token_validation: dict[str, object] | None = None
    if vocab_size is not None:
        sampled_tokens = [
            token
            for sequence in sequences
            for token in sequence["first_tokens"] + sequence["last_tokens"]
        ]
        invalid = [
            token
            for token in sampled_tokens
            if not isinstance(token, int) or token < 0 or token >= vocab_size
        ]
        if invalid:
            raise IndexedDatasetError(
                "抽样 token 超出词表范围："
                f"前几个非法值为 {invalid[:8]}，vocab_size={vocab_size}"
            )
        token_validation = {
            "vocab_size": vocab_size,
            "sampled_token_count": len(sampled_tokens),
            "passed": True,
        }

    return {
        "magic": "MMIDIDX",
        "format": "MMIDIDX",
        "version": version,
        "dtype_code": dtype_code,
        "dtype": dtype_name,
        "item_size_bytes": item_size,
        "sequence_count": sequence_count,
        "document_count": document_count,
        "sampled_sequence_count": sampled_sequences,
        "sampled_document_count": sampled_documents,
        "sequence_lengths": lengths,
        "sequence_offsets": pointers,
        "index": {
            "path": str(idx_path),
            "bytes": actual_index_bytes,
            "expected_min_bytes": expected_index_bytes,
        },
        "bin": {
            "path": str(bin_path),
            "bytes": bin_bytes,
        },
        "sequences": sequences,
        "last_sequence": last_sequence,
        "token_validation": token_validation,
        "documents": {
            "first_indices": first_documents,
            "last_indices": last_documents,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    """创建命令行解析器。"""

    parser = argparse.ArgumentParser(
        description="只读检查 Megatron MMIDIDX 的 .idx/.bin 数据集"
    )
    parser.add_argument(
        "prefix",
        nargs="?",
        type=Path,
        help="数据前缀，例如传入 1 就读取 1.idx 和 1.bin",
    )
    parser.add_argument("--idx", type=Path, help="索引文件路径")
    parser.add_argument("--bin", dest="bin_path", type=Path, help="token 二进制文件路径")
    parser.add_argument(
        "--sequences",
        type=int,
        default=3,
        help="报告前多少条序列（默认：3）",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=16,
        help="每条序列报告首尾多少个 token（默认：16）",
    )
    parser.add_argument(
        "--documents",
        type=int,
        default=8,
        help="报告首尾多少个 document index（默认：8）",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        help="可选：校验抽样 token 是否位于 [0, vocab_size) 内",
    )
    parser.add_argument("--output", type=Path, help="可选的 JSON 报告路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行检查器并把 JSON 报告打印到标准输出。"""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        idx_path, bin_path = _resolve_paths(args.prefix, args.idx, args.bin_path)
        report = inspect_indexed_dataset(
            idx_path,
            bin_path,
            sequence_limit=args.sequences,
            token_limit=args.tokens,
            document_limit=args.documents,
            vocab_size=args.vocab_size,
        )
    except (OSError, IndexedDatasetError, struct.error) as exc:
        parser.exit(2, f"错误：{exc}\n")

    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
