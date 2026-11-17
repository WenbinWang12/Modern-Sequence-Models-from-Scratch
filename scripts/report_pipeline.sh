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

# 同一计划只能有一个流水线。锁文件本身可以保留，flock 会在进程退出时自动释放，
# 因此进程被系统终止后仍能直接重新执行命令，不会留下需要人工清理的锁目录。
lock_file="outputs/.report_pipeline_${plan}.lock"
exec 9>"$lock_file"
if ! flock -n 9; then
  echo "report pipeline already running for plan=$plan" >&2
  exit 1
fi

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
  # 经监督脚本启动：进程因配额或外部中断退出时自动从检查点续跑。
  scripts/supervise_training.sh "configs/report/$1.json" "$2" "$gpu" || log "FAILED train $2"
}

prune() { # prune <arm>：曲线评测取完数值后删掉中间检查点，主目录有配额限制
  log "prune $1"
  "$python" scripts/prune_checkpoints.py --runs "outputs/main-$1" --keep 6000 --apply \
    >>logs/pipeline_evals.log 2>&1 || log "FAILED prune $1"
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

profile() { # profile <arm>：单会话的状态大小与逐 token 解码速度
  local out=outputs/evals/profile_$1.json checkpoint
  [ -f "$out" ] && return 0
  checkpoint=$(ls -d "outputs/main-$1"/step_*.orbax | sort | tail -n 1)
  log "profile $1"
  CUDA_VISIBLE_DEVICES=$gpu JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. \
    "$python" scripts/profile_report_sessions.py --checkpoint "$checkpoint" --data data/tokens/pg19-test \
    --context-lengths 2048 8192 32768 --output "$out" >>logs/pipeline_evals.log 2>&1 || log "FAILED profile $1"
}

niah() { # niah <arm> [extra args]：单针检索（含“针不在场”对照）
  local arm=$1 out checkpoint
  shift
  out=outputs/niah/main_$arm.json
  [ -f "$out" ] && return 0
  checkpoint=$(ls -d "outputs/main-${arm%%_*}"/step_*.orbax | sort | tail -n 1)
  log "niah $arm"
  CUDA_VISIBLE_DEVICES=$gpu JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. \
    "$python" scripts/eval_niah.py --checkpoint "$checkpoint" --output "$out" "$@" \
    >>logs/pipeline_evals.log 2>&1 || log "FAILED niah $arm"
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
  train main_swa main-swa
  evals curves swa
  prune swa
  evals main swa
  evals long swa
  bench main_swa main-swa
  train main_full main-full
  evals curves full
  prune full
  evals main full
  evals long full
  bench main_full main-full
  wait_for main-ttt
  # 三个模型的推理吞吐在同一张空闲的卡上连续测量。
  evals latency swa full ttt
  profile swa
  profile full
  profile ttt
  evals recall swa full
  niah swa
  niah full
  drain_queue
  ;;
ttt)
  train main_ttt main-ttt
  evals curves ttt
  prune ttt
  evals main ttt
  evals long ttt
  bench main_ttt main-ttt
  bench main_ttt_noadapt main-ttt --no-adapt
  train main_ck main-ck
  evals curves ck
  prune ck
  evals main ck
  evals long ck
  bench main_ck main-ck
  evals dynamic
  evals recall ttt ck
  niah ttt
  niah ttt_noadapt --no-adapt
  niah ck
  drain_queue
  ;;
*)
  echo "unknown plan: $plan" >&2
  exit 2
  ;;
esac
log "pipeline $plan finished"
