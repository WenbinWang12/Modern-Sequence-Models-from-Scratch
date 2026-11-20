#!/usr/bin/env bash
# Run the first eight-GPU screening experiments.
#
# Each selected configuration is one distributed job that uses all visible
# GPUs on a node. The script runs configurations sequentially so that an
# eight-card machine is never split between independent cells.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
MODE="${1:-smoke}"
PYTHON="${TTT_PYTHON:-$ROOT/.venv/bin/python}"
DATA="${TTT_DATA:-$ROOT/configs/data/sweep/dolma3.json}"
OUTPUT_ROOT="${TTT_OUTPUT_ROOT:-$ROOT/outputs/experiments/first8}"
CONFIG_ROOT="${TTT_CONFIG_ROOT:-$OUTPUT_ROOT/configs}"
EXPECTED_GPUS="${TTT_EXPECTED_GPUS:-8}"
TOKENS="${TTT_TOKENS:-3000000000}"
BATCH_SIZE="${TTT_BATCH_SIZE:-64}"
SEQUENCE_LENGTH="${TTT_SEQUENCE_LENGTH:-4096}"
SMOKE_STEPS="${TTT_SMOKE_STEPS:-100}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-ttt-first8}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY

case "$MODE" in
  smoke|wave_a|wave_b|all) ;;
  *)
    echo "usage: bash scripts/run_first8_h100.sh {smoke|wave_a|wave_b|all}" >&2
    exit 2
    ;;
esac

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${TTT_PYTHON_FALLBACK:-python3}"
fi
if ! command -v "$PYTHON" >/dev/null 2>&1 && [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
  echo "Set TTT_PYTHON to the server virtualenv Python." >&2
  exit 1
fi

if [[ ! -e "$DATA" ]]; then
  echo "Dolma3 recipe/data path does not exist: $DATA" >&2
  echo "Build it first, for example:" >&2
  echo "  TTT_CORPUS_SCRATCH=/dev/shm/\$USER-ttt bash scripts/build_corpus.sh dolma3" >&2
  echo "Or set TTT_DATA to an existing mixture recipe or token manifest." >&2
  exit 1
fi

"$PYTHON" - "$DATA" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser().resolve()
if path.is_dir():
    raise SystemExit(f"TTT_DATA must be a manifest or mixture recipe, not a directory: {path}")
if path.suffix != ".json":
    raise SystemExit(0)
try:
    document = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"cannot read data JSON {path}: {error}")
if document.get("format") == "ttt_mixture_v1":
    missing = []
    for source in document.get("sources", []):
        source_path = (path.parent / source["data"]).resolve()
        if not source_path.is_file():
            missing.append(str(source_path))
    if missing:
        print("Missing Dolma3 source manifests:", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        print("Build the corpus first or set TTT_DATA to a verified recipe.", file=sys.stderr)
        raise SystemExit(1)
PY

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export JAX_DEFAULT_MATMUL_PRECISION="${JAX_DEFAULT_MATMUL_PRECISION:-highest}"
export PYTHONUNBUFFERED=1

mkdir -p "$CONFIG_ROOT" "$OUTPUT_ROOT"

echo "Checking visible accelerators with $PYTHON"
"$PYTHON" - "$EXPECTED_GPUS" <<'PY'
import sys

import jax

expected = int(sys.argv[1])
devices = jax.local_devices()
gpus = [device for device in devices if device.platform == "gpu"]
print(f"visible devices: {devices}")
print(f"visible GPUs: {len(gpus)} / expected {expected}")
if len(gpus) != expected:
    raise SystemExit(
        f"expected exactly {expected} visible GPUs for one distributed cell, found {len(gpus)}"
    )
PY

"$PYTHON" "$ROOT/scripts/generate_first8_experiments.py" \
  --output "$CONFIG_ROOT" \
  --tokens "$TOKENS" \
  --batch-size "$BATCH_SIZE" \
  --sequence-length "$SEQUENCE_LENGTH" \
  --smoke-steps "$SMOKE_STEPS" \
  --output-root "$OUTPUT_ROOT" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY"

mapfile -t RUNS < <("$PYTHON" - "$CONFIG_ROOT/manifest.json" "$MODE" <<'PY'
import json
import sys

manifest = json.loads(open(sys.argv[1], encoding="utf-8").read())
mode = sys.argv[2]
allowed = {"smoke"} if mode == "smoke" else {mode} if mode != "all" else {"smoke", "wave_a", "wave_b"}
for run in manifest["runs"]:
    if run["wave"] in allowed:
        print(run["name"])
PY
)

if [[ "${#RUNS[@]}" -eq 0 ]]; then
  echo "No runs selected for mode $MODE" >&2
  exit 1
fi

if [[ -n "$WANDB_API_KEY" ]]; then
  echo "WANDB_API_KEY is set. W&B integration is reserved for the callback layer."
else
  echo "WANDB_API_KEY is empty. Metrics will be kept in each output directory."
fi

for name in "${RUNS[@]}"; do
  config="$CONFIG_ROOT/$name.json"
  output="$OUTPUT_ROOT/$name"
  export WANDB_NAME="$name"
  export WANDB_RUN_NAME="$name"
  echo
  echo "=== $name ==="
  echo "config: $config"
  echo "output: $output"
  echo "using all $EXPECTED_GPUS visible GPUs in this job"
  bash "$ROOT/scripts/run_gpu_train.sh" \
    --config "$config" \
    --data "$DATA" \
    --output "$output" \
    --auto-resume
done

echo
echo "Completed ${#RUNS[@]} run(s). Names and factors are recorded in $CONFIG_ROOT/manifest.json."
