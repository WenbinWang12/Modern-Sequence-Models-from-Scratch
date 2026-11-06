#!/usr/bin/env bash
# 生成仓库示例 token 数据，便于第一次运行。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${TTT_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${TTT_PYTHON_FALLBACK:-python3}"
fi
OUTPUT="${1:-$ROOT/outputs/sample-tokens.npy}"
mkdir -p "$(dirname "$OUTPUT")"
cd "$ROOT"
exec "$PYTHON" -m ttt.data --input examples/sample.txt --output "$OUTPUT" --tokenizer byte
