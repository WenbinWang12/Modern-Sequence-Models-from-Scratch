#!/usr/bin/env bash
# 在当前分配的多 GPU 设备上执行 mesh、检查点和恢复验证。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${TTT_PYTHON:-$ROOT/.venv/bin/python}"
OUTPUT="${1:-$ROOT/outputs/mesh-validation.json}"
CHECKPOINT_DIR="${2:-$ROOT/outputs/mesh-checkpoint}"
mkdir -p "$(dirname "$OUTPUT")" "$CHECKPOINT_DIR"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-highest}"
cd "$ROOT"
exec "$PYTHON" -m ttt.validate_mesh --require-gpu --output "$OUTPUT" --checkpoint-dir "$CHECKPOINT_DIR"
