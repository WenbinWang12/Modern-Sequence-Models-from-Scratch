#!/usr/bin/env bash
# 技术报告实验的单卡流水线：在一张 GPU 上依次执行训练、held-out 评测和 benchmark。
# 每一步都可重复执行，已完成的步骤自动跳过；两张卡各跑一个计划，互不通信。
#
# 用法：scripts/report_pipeline.sh <gpu-index> <plan>
#   plan=baseline  滑窗基线 -> 全注意力基线 -> 推理吞吐 -> 消融队列
#   plan=ttt       TTT-E2E -> 无更新对照 -> 朴素测试时更新基线 -> 消融队列
set -uo pipefail

gpu=$1
plan=$2
python=${TTT_PYTHON:-.venv/bin/python}
tasks="hellaswag piqa arc_easy arc_challenge winogrande lambada_openai sciq openbookqa wikitext blimp mmlu pg19"
mkdir -p logs outputs/benchmarks-v2 outputs/queue

log() { echo "$(date -Is) [gpu$gpu] $*"; }

complete() { grep -q '"status": "complete"' "outputs/$1/status.json" 2>/dev/null; }

wait_for() { # 等待在别处启动的训练完成并释放显存
  log "wait $1"
  until complete "$1"; do sleep 30; done
  while pgrep -f "ttt.train .*--output outputs/$1 " >/dev/null; do sleep 5; done
}

train() { # train <config-name> <run-name>
  complete "$2" && return 0
  if pgrep -f "ttt.train .*--output outputs/$2 " >/dev/null; then
    wait_for "$2" # 已经由别的进程启动：只等待，不重复训练
    return 0
  fi
  log "train $2"
  scripts/run_report_arm.sh "configs/report/$1.json" "$2" "$gpu" >>"logs/train_$2.log" 2>&1 || log "FAILED train $2"
}

evals() { # evals <stage> [arms...]
  local stage=$1
  shift
  log "evals $stage $*"
  if [ $# -gt 0 ]; then
    "$python" scripts/run_report_evals.py --stage "$stage" --gpu "$gpu" --arms "$@" >>logs/pipeline_evals.log 2>&1
  else
    "$python" scripts/run_report_evals.py --stage "$stage" --gpu "$gpu" >>logs/pipeline_evals.log 2>&1
  fi
}

bench() { # bench <output-name> <run-name> [extra eval_benchmarks.py args]
  local name=$1 run=$2 out checkpoint
  shift 2
  out=outputs/benchmarks-v2/$name.json
  [ -f "$out.done" ] && return 0
  checkpoint=$(ls -d "outputs/$run"/step_*.orbax | sort | tail -n 1)
  log "bench $name"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpu JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. \
    "$python" scripts/eval_benchmarks.py --checkpoint "$checkpoint" --tasks $tasks \
    --text-dir data/raw/pg19/test --output "$out" "$@" >>"logs/bench_$name.log" 2>&1 &&
    touch "$out.done" || log "FAILED bench $name"
}

make_iso_config() { # 用实测的单步时间决定“与 TTT 等训练时间”的滑窗基线步数
  until complete abl-ttt && complete abl-swa; do sleep 60; done
  local steps
  steps=$("$python" -c '
import json, statistics
def median_seconds(run):
    rows = [json.loads(line) for line in open(f"outputs/{run}/metrics.jsonl") if line.strip()]
    return statistics.median(row["seconds"] for row in rows[len(rows) // 5 :])
ratio = median_seconds("abl-ttt") / median_seconds("abl-swa")
print(int(round(5000 * ratio / 100) * 100))
')
  log "iso-compute sliding-window baseline: $steps steps"
  "$python" scripts/make_report_configs.py --iso-steps "$steps" >/dev/null
}

drain_queue() { # 两张卡共用一个任务列表，mkdir 的原子性保证每个任务只被领取一次
  local config run
  while read -r config run <&3; do
    [ -z "$config" ] && continue
    mkdir "outputs/queue/$run.claim" 2>/dev/null || continue
    if [ "$config" = "abl_swa_iso" ]; then make_iso_config; fi
    train "$config" "$run"
    log "evals ablations $run"
    "$python" scripts/run_report_evals.py --stage ablations --gpu "$gpu" --runs "$run" >>logs/pipeline_evals.log 2>&1
  done 3<configs/report/queue.txt
}

case "$plan" in
baseline)
  wait_for main-swa
  evals main swa
  evals long swa
  bench main_swa main-swa
  train main_full main-full
  evals curves full
  evals main full
  evals long full
  bench main_full main-full
  wait_for main-ttt
  # 三个模型的推理吞吐在同一张空闲的卡上连续测量。
  evals latency swa full ttt
  drain_queue
  ;;
ttt)
  wait_for main-ttt
  evals main ttt
  evals long ttt
  bench main_ttt main-ttt
  bench main_ttt_noadapt main-ttt --no-adapt
  train main_ck main-ck
  evals main ck
  evals long ck
  bench main_ck main-ck
  evals dynamic
  drain_queue
  ;;
*)
  echo "unknown plan: $plan" >&2
  exit 2
  ;;
esac
log "pipeline $plan finished"
