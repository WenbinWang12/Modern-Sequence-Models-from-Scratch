"""有校验清单的分片 token 数据，支持监督标签掩码。"""

import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np

from .lifecycle import atomic_json


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TokenDataset:
    def __init__(self, path, vocab_size):
        path = Path(path).resolve()
        self.shards, self.masks = [], []
        if path.suffix == ".json":
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest["format_version"] != 1 or manifest["vocab_size"] != vocab_size:
                raise ValueError("数据清单版本或词表大小不匹配。")
            records = manifest["shards"]
            self.fingerprint = file_sha256(path)
        else:
            records = [{"tokens": path.name, "sha256": file_sha256(path)}]
            self.fingerprint = records[0]["sha256"]
        if not records:
            raise ValueError("数据清单不能为空。")
        for record in records:
            tokens_path = (path.parent / record["tokens"]).resolve()
            if tokens_path.parent != path.parent:
                raise ValueError("分片文件必须位于清单所在目录。")
            if file_sha256(tokens_path) != record["sha256"]:
                raise ValueError(f"分片校验失败：{tokens_path.name}")
            tokens = np.load(tokens_path, mmap_mode="r", allow_pickle=False)
            if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer) or not tokens.size:
                raise ValueError("token 分片必须为非空一维整数数组。")
            if tokens.min() < 0 or tokens.max() >= vocab_size:
                raise ValueError("训练数据中存在超出词表的 token。")
            if "count" in record and record["count"] != tokens.size:
                raise ValueError("分片 token 数量不匹配。")
            mask = None
            if "mask" in record:
                mask_path = (path.parent / record["mask"]).resolve()
                if mask_path.parent != path.parent or file_sha256(mask_path) != record["mask_sha256"]:
                    raise ValueError("标签掩码分片校验失败。")
                mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
                if mask.shape != tokens.shape or not np.isfinite(mask).all() or ((mask < 0) | (mask > 1)).any():
                    raise ValueError("标签掩码必须与 tokens 等长，且位于 [0, 1]。")
            self.shards.append(tokens)
            self.masks.append(mask)
        self.ends = np.cumsum([shard.size for shard in self.shards], dtype=np.int64)
        self.size = int(self.ends[-1])

    def rows(self, starts, length):
        indices = np.asarray(starts)[:, None] + np.arange(length)
        if (indices < 0).any() or (indices >= self.size).any():
            raise ValueError("数据切片越界。")
        shard_ids = np.searchsorted(self.ends, indices, side="right")
        result = np.empty(indices.shape, dtype=np.int32)
        masks = np.ones(indices.shape, dtype=np.float32)
        begin = 0
        for index, (shard, mask) in enumerate(zip(self.shards, self.masks)):
            selected = shard_ids == index
            offsets = indices[selected] - begin
            result[selected] = shard[offsets]
            if mask is not None:
                masks[selected] = mask[offsets]
            begin = self.ends[index]
        # 第 i 个输入的监督标签是第 i+1 个 token。
        return result, masks[:, 1:]


def prepare_sharded_data(input_path, output_dir, tokenizer, *, shard_tokens=1_000_000, supervised=False):
    if shard_tokens < 2:
        raise ValueError("每个分片至少包含两个 token。")
    input_path = Path(input_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("数据清单已存在；新版本应使用新目录。")
    input_before = file_sha256(input_path)
    consumed_digest = hashlib.sha256()
    version = uuid.uuid4().hex
    records, buffer, mask_buffer = [], [], []
    stats = {"documents": 0, "empty_documents": 0, "tokens": 0, "supervised_tokens": 0}

    def flush(count):
        nonlocal buffer, mask_buffer
        tokens, masks = buffer[:count], mask_buffer[:count]
        name = f"{version}-{len(records):05d}"
        token_path = output / (name + ".npy")
        mask_path = output / (name + ".mask.npy")
        for path, values, dtype in ((token_path, tokens, np.int32), (mask_path, masks, np.uint8)):
            with path.open("xb") as handle:
                np.save(handle, np.asarray(values, dtype=dtype), allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
        records.append(
            {
                "tokens": token_path.name,
                "mask": mask_path.name,
                "count": count,
                "sha256": file_sha256(token_path),
                "mask_sha256": file_sha256(mask_path),
            }
        )
        buffer, mask_buffer = buffer[count:], mask_buffer[count:]

    with input_path.open("rb") as source:
        for raw_line in source:
            # 按实际消费的原字节计算，包括空行与 CRLF，避免只记录处理结束时的文件。
            consumed_digest.update(raw_line)
            line = raw_line.decode("utf-8")
            if not line.strip():
                continue
            record = json.loads(line)
            if supervised:
                prompt, response = record["prompt"], record["response"]
                if not isinstance(prompt, str) or not isinstance(response, str):
                    raise ValueError("prompt 和 response 必须是字符串。")
                if not response.strip():
                    stats["empty_documents"] += 1
                    continue
                prefix = tokenizer.encode(prompt)
                suffix = tokenizer.encode(response, add_bos=False)
                tokens = prefix + suffix
                masks = [0] * len(prefix) + [1] * len(suffix)
            else:
                text = record["text"]
                if not isinstance(text, str):
                    raise ValueError("text 必须是字符串。")
                if not text.strip():
                    stats["empty_documents"] += 1
                    continue
                tokens = tokenizer.encode(text)
                masks = [1] * len(tokens)
                if tokens and tokenizer.bos_token_id is not None and tokens[0] == tokenizer.bos_token_id:
                    masks[0] = 0
            if tokenizer.eos_token_id is not None:
                tokens.append(tokenizer.eos_token_id)
                masks.append(1)
            stats["documents"] += 1
            stats["tokens"] += len(tokens)
            stats["supervised_tokens"] += sum(masks)
            buffer.extend(tokens)
            mask_buffer.extend(masks)
            while len(buffer) >= shard_tokens:
                flush(shard_tokens)
    if buffer:
        flush(len(buffer))
    if stats["tokens"] < 2:
        raise ValueError("数据至少需要两个 token。")
    input_consumed = consumed_digest.hexdigest()
    if input_before != input_consumed or file_sha256(input_path) != input_consumed:
        raise RuntimeError("输入语料在分片准备期间发生变化；未发布数据清单，请使用稳定输入重试。")
    manifest = {
        "format_version": 1,
        "version": version,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer": getattr(tokenizer, "name", "byte"),
        "supervised": supervised,
        "input_sha256": input_consumed,
        "shards": records,
        "stats": stats,
    }
    atomic_json(manifest_path, manifest)
    return manifest_path
