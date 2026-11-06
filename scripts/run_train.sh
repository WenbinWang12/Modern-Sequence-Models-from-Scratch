#!/usr/bin/env bash
# 统一设置训练运行时变量，并把参数原样交给 ttt.train。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${TTT_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${TTT_PYTHON_FALLBACK:-python3}"
fi
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-highest}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
cd "$ROOT"
exec "$PYTHON" -m ttt.train "$@"
