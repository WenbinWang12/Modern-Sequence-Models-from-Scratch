#!/usr/bin/env bash
# GPU 训练入口。多进程 mesh 请通过 run_mesh_validate.sh 或集群启动器调用。
set -euo pipefail

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/run_train.sh" "$@"
