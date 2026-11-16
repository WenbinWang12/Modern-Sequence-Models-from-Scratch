#!/usr/bin/env bash
set -Eeuo pipefail

# 下载一个小型公开语言建模语料的完整文件，并额外生成可审阅的前 1 MiB 样本。
# URL 指向公开 GitHub 镜像；固定摘要用于防止 main 分支内容悄悄变化。

readonly SOURCE_URL="https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
readonly EXPECTED_SHA256="9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f"
readonly SAMPLE_BYTES=1048576

usage() {
  echo "用法：$0 [输出目录]" >&2
  echo "默认输出目录：data/external/wikitext-2" >&2
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -gt 1 ]]; then
  usage
  exit 2
fi

output_dir="${1:-data/external/wikitext-2}"
mkdir -p "$output_dir"
full_path="$output_dir/train.txt"
sample_path="$output_dir/train.prefix-1MiB-complete-lines.txt"
metadata_path="$output_dir/metadata.json"
temporary_path="$output_dir/.train.txt.part"

cleanup() {
  rm -f "$temporary_path"
}
trap cleanup EXIT

if [[ -e "$full_path" ]]; then
  echo "已存在：$full_path；将校验而不是覆盖它。"
else
  command -v curl >/dev/null || { echo "缺少 curl。" >&2; exit 1; }
  curl --fail --location --retry 3 --connect-timeout 20 --max-time 300 \
    --output "$temporary_path" "$SOURCE_URL"
  mv -- "$temporary_path" "$full_path"
fi

command -v sha256sum >/dev/null || { echo "缺少 sha256sum。" >&2; exit 1; }
actual_sha256="$(sha256sum "$full_path" | awk '{print $1}')"
if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
  echo "下载文件摘要不匹配。" >&2
  echo "期望：$EXPECTED_SHA256" >&2
  echo "实际：$actual_sha256" >&2
  exit 1
fi

# 先按字节截取，再删除最后一行，避免训练样本以半行文本结束。
head -c "$SAMPLE_BYTES" "$full_path" | head -n -1 > "$sample_path"

full_bytes="$(wc -c < "$full_path" | tr -d ' ')"
full_lines="$(wc -l < "$full_path" | tr -d ' ')"
sample_bytes="$(wc -c < "$sample_path" | tr -d ' ')"
sample_lines="$(wc -l < "$sample_path" | tr -d ' ')"
downloaded_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
sample_sha256="$(sha256sum "$sample_path" | awk '{print $1}')"

cat > "$metadata_path" <<EOF
{
  "dataset": "WikiText-2",
  "config": "wikitext-2-v1 mirror train split",
  "source_url": "$SOURCE_URL",
  "source_repository": "https://github.com/pytorch/examples",
  "dataset_card": "https://huggingface.co/datasets/Salesforce/wikitext",
  "license": "CC BY-SA 4.0 (verify upstream terms before redistribution)",
  "downloaded_at_utc": "$downloaded_at",
  "full_file": {
    "path": "train.txt",
    "bytes": $full_bytes,
    "lines": $full_lines,
    "sha256": "$actual_sha256"
  },
  "sample": {
    "path": "train.prefix-1MiB-complete-lines.txt",
    "bytes": $sample_bytes,
    "lines": $sample_lines,
    "sha256": "$sample_sha256",
    "selection": "first 1048576 bytes, then drop the incomplete final line"
  },
  "notes": [
    "The mirror file contains pretokenized WikiText-style text and <unk> markers; verify upstream semantics before calling it raw text.",
    "The sample is for pipeline verification, not a representative pretraining corpus."
  ]
}
EOF

echo "完整文件：$full_path"
echo "完整 SHA256：$actual_sha256"
echo "样本文件：$sample_path"
echo "样本 SHA256：$sample_sha256"
echo "元数据：$metadata_path"
