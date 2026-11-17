#!/usr/bin/env bash
# 在一张 GPU 上运行主模型的扩展 benchmark。每个输出文件都可独立续跑。
# 这些生成任务使用预训练基座和零样本提示，结果应与模型检查点、数据摘要和协议版本一起阅读。
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
  echo "用法：$0 <gpu-index> <run-name> [adapt|no-adapt] [limit]" >&2
  exit 2
fi

gpu=$1
run=$2
mode=${3:-no-adapt}
limit=${4:-}
python=${TTT_PYTHON:-.venv/bin/python}
data=${TTT_EXTENDED_DATA:-data/public}
checkpoint=${TTT_CHECKPOINT:-}
output_root=${TTT_EXTENDED_OUTPUT:-outputs/benchmarks-extended}

case "$mode" in
  adapt) adapt_args=() suffix=adapt ;;
  no-adapt) adapt_args=(--no-adapt) suffix=noadapt ;;
  *) echo "mode 必须是 adapt 或 no-adapt。" >&2; exit 2 ;;
esac

if [ -z "$checkpoint" ]; then
  checkpoint=$(find "outputs/$run" -maxdepth 1 -type d -name 'step_*.orbax' -print | sort | tail -n 1)
fi
[ -n "$checkpoint" ] && [ -d "$checkpoint" ] || { echo "找不到检查点：$run" >&2; exit 1; }

mkdir -p "$output_root"
export CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=.

run_json() {
  local name=$1
  shift
  local output="$output_root/${run}_${suffix}_${name}.json"
  [ -f "$output" ] && return 0
  "$python" "$@" --checkpoint "$checkpoint" --output "$output"
}

limit_args=()
[ -n "$limit" ] && limit_args=(--limit "$limit")

# ARC、PIQA 和 GPQA 的选择题脚本固定为 no-adapt continuation 协议。
run_json arc scripts/eval_arc_piqa_loglikelihood.py --task arc \
  --data "$data/arc/arc_challenge_test.jsonl" "${limit_args[@]}"
run_json piqa scripts/eval_arc_piqa_loglikelihood.py --task piqa \
  --data "$data/piqa/piqa_validation.jsonl" "${limit_args[@]}"
run_json gpqa scripts/eval_arc_piqa_loglikelihood.py --task gpqa \
  --data "$data/gpqa/gpqa_test.jsonl" "${limit_args[@]}"

run_json gsm8k scripts/eval_gsm8k_generate.py --data "$data/gsm8k/test.jsonl" \
  "${adapt_args[@]}" "${limit_args[@]}"
run_json humaneval scripts/eval_humaneval_generate.py --data "$data/humaneval/humaneval_test.jsonl" \
  "${adapt_args[@]}" --run-tests "${limit_args[@]}"
run_json bbh scripts/eval_bbh_loglikelihood.py --data "$data/bbh/sports_understanding.json" \
  "${limit_args[@]}"

# 这是 RULER/LongBench 的工程前置诊断，不冒充公开套件总分。
long_output="$output_root/${run}_${suffix}_synthetic_long_context.json"
if [ ! -f "$long_output" ]; then
  long_args=(--checkpoint "$checkpoint" --lengths 256 512 1024 2048 4096 8192 --samples 32 --output "$long_output")
  [ "$mode" = adapt ] && long_args+=(--adapt)
  "$python" scripts/eval_synthetic_long_context.py "${long_args[@]}"
fi

printf '%s\n' "完成：$output_root/${run}_${suffix}_*.json"
