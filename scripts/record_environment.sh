#!/usr/bin/env bash
# 记录设备、依赖、PBS 信息和代码摘要。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${TTT_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${TTT_PYTHON_FALLBACK:-python3}"
fi
OUTPUT="${1:-$ROOT/environment.json}"
if [[ "${1:-}" == "--output" ]]; then
  OUTPUT="${2:?--output 需要路径}"
fi
cd "$ROOT"
exec "$PYTHON" -m ttt.provenance --output "$OUTPUT"
