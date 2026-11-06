#!/usr/bin/env bash
# 在单机 GPU 上执行架构、因果性、梯度和流式一致性检查。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${TTT_PYTHON:-$ROOT/.venv/bin/python}"
OUTPUT="${1:-$ROOT/outputs/gpu-validation.json}"
mkdir -p "$(dirname "$OUTPUT")"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-highest}"
cd "$ROOT"
"$PYTHON" -m ttt.provenance --output "$(dirname "$OUTPUT")/environment.json"
exec "$PYTHON" -m ttt.validate --output "$OUTPUT" --require-gpu
