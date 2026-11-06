#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="${TTT_TOOLS_DIR:-$ROOT/.tools}"
mkdir -p "$TOOLS"
if [[ ! -x "$TOOLS/uv" ]]; then
  curl --fail --location --retry 3 \
    https://github.com/astral-sh/uv/releases/download/0.13.6/uv-x86_64-unknown-linux-gnu.tar.gz \
    --output "$TOOLS/uv.tar.gz"
  tar -xzf "$TOOLS/uv.tar.gz" -C "$TOOLS" --strip-components=1
fi
export UV_PYTHON_INSTALL_DIR="$TOOLS/python"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$TOOLS/cache}"
"$TOOLS/uv" python install 3.12.16
cd "$ROOT"
"$TOOLS/uv" sync --frozen --python 3.12.16 --extra dev --extra checkpoint --extra cuda --extra serve
printf 'ready\n' > "$ROOT/.environment-ready"
