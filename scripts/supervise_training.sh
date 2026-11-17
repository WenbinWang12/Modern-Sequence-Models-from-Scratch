#!/usr/bin/env bash
# 监督一次训练：进程意外退出就从最近的检查点续跑，直到 status.json 变成 complete。
#
# 这台 GPU 机器是多人共享、没有调度器的，别的用户清理显卡时会连带杀掉我们的进程
# （2026-11-17 13:02 发生过一次，两个训练同时被 SIGKILL，日志里没有任何异常输出）。
# 训练本身支持 --auto-resume，恢复参数、优化器状态和数据采样位置，所以代价只是
# 上一个检查点之后的若干步。
#
# 用法：scripts/supervise_training.sh <config.json> <run-name> <gpu-index> [data-recipe]
set -uo pipefail

config=$1
name=$2
gpu=$3
data=${4:-configs/data/fineweb_pg19_50_50.json}
limit=${TTT_SUPERVISE_RETRIES:-50}

mkdir -p "outputs/$name" logs
events="outputs/$name/interruptions.jsonl"

complete() { grep -q '"status": "complete"' "outputs/$name/status.json" 2>/dev/null; }

step_now() {
  python3 -c "
import json, pathlib
path = pathlib.Path('outputs/$name/latest.json')
print(json.loads(path.read_text())['step'] if path.exists() else 0)
" 2>/dev/null || echo 0
}

for attempt in $(seq 0 "$limit"); do
  complete && break
  started=$(step_now)
  if [ "$attempt" -gt 0 ]; then
    # 把每次中断记录下来，训练稳定性一节据此说明，而不是假装没发生过。
    printf '{"attempt": %d, "resumed_from_step": %s, "time": "%s"}\n' \
      "$attempt" "$started" "$(date -Is)" >>"$events"
    echo "$(date -Is) [$name] restart #$attempt from step $started" >>"logs/supervise_$name.log"
  fi
  scripts/run_report_arm.sh "$config" "$name" "$gpu" "$data" >>"logs/train_$name.log" 2>&1
  code=$?
  complete && break
  echo "$(date -Is) [$name] exited code=$code at step $(step_now)" >>"logs/supervise_$name.log"
  sleep 20
done

if complete; then
  echo "$(date -Is) [$name] complete" >>"logs/supervise_$name.log"
else
  echo "$(date -Is) [$name] giving up after $limit restarts" >>"logs/supervise_$name.log"
  exit 1
fi
