#!/usr/bin/env bash
# 在任意 Linux 节点准备可复现的 uv/Python/JAX 环境。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TOOLS_DIR="${TTT_TOOLS_DIR:-$ROOT/.tools}"
UV_VERSION="${TTT_UV_VERSION:-0.13.6}"
PYTHON_VERSION="${TTT_PYTHON_VERSION:-3.12.16}"
INSTALL_CUDA="${TTT_INSTALL_CUDA:-1}"

mkdir -p "$TOOLS_DIR"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64) UV_ARCH="x86_64" ;;
  aarch64|arm64) UV_ARCH="aarch64" ;;
  *) echo "不支持的 Linux 架构：$ARCH" >&2; exit 2 ;;
esac

UV="$TOOLS_DIR/uv"
if [[ ! -x "$UV" ]]; then
  archive="$TOOLS_DIR/uv.tar.gz"
  curl --fail --location --retry 3 \
    "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-${UV_ARCH}-unknown-linux-gnu.tar.gz" \
    --output "$archive"
  tar -xzf "$archive" -C "$TOOLS_DIR" --strip-components=1
  rm -f "$archive"
fi

export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$TOOLS_DIR/python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$TOOLS_DIR/cache}"
"$UV" python install "$PYTHON_VERSION"

extras=(dev checkpoint serve)
if [[ "$INSTALL_CUDA" == "1" ]]; then
  extras+=(cuda)
fi
cd "$ROOT"
sync_args=(--frozen --python "$PYTHON_VERSION")
for extra in "${extras[@]}"; do
  sync_args+=(--extra "$extra")
done
"$UV" sync "${sync_args[@]}"
TTT_PYTHON="$ROOT/.venv/bin/python" bash "$ROOT/scripts/record_environment.sh" \
  --output "$ROOT/environment.json"
printf 'ready\n' > "$ROOT/.environment-ready"
echo "环境已准备：Python $PYTHON_VERSION，CUDA extra=$INSTALL_CUDA"
