#!/usr/bin/env bash
# 数据配比对照：同一个小模型、同样的步数和随机种子，只换数据配方。
# 每个配方训练完后逐领域评测 held-out loss，再跑零样本 benchmark。重复执行会跳过已完成的步骤。
#
# 用法：scripts/run_mixture_sweep.sh <gpu-index> <配方名>...
# 配方名对应 configs/data/sweep/<配方名>.json。换随机种子重复时，用 TTT_SWEEP_CONFIG 指向
# 另一个种子的配置，并用 TTT_SWEEP_TAG（例如 -s1）区分运行目录和结果文件。
set -uo pipefail

gpu=$1
shift
python=${TTT_PYTHON:-.venv/bin/python}
config=${TTT_SWEEP_CONFIG:-configs/report/abl_swa.json}
tag=${TTT_SWEEP_TAG:-}
tokens=${TTT_SWEEP_TOKENS:-data/tokens/v2}
tasks=${TTT_SWEEP_TASKS:-"hellaswag piqa arc_easy arc_challenge winogrande sciq openbookqa mmlu lambada_openai wikitext"}
results=outputs/mixture_sweep
mkdir -p logs "$results"

heldout=(books=data/tokens/pg19-validation)
for name in web_edu web wiki papers math_web math_problems code qa; do
  heldout+=("$name=$tokens/$name-heldout")
done

# 第二组 held-out 来自 Dolma 3 抽样语料（建好了才评测），用来比较不同语料训练出的模型。
dolma=()
for name in d3_web d3_pdfs d3_code d3_math d3_arxiv; do
  [ -e "data/tokens/dolma3/$name-heldout/manifest.json" ] && dolma+=("$name=data/tokens/dolma3/$name-heldout")
done

log() { echo "$(date -Is) [gpu$gpu] $*" >>logs/mixture_sweep.log; }

evaluate() { # evaluate <脚本> <输出文件> <日志> <参数...>
  local script=$1 out=$2 logfile=$3
  shift 3
  [ -f "$out.done" ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. \
    "$python" "$script" --output "$out" "$@" >>"$logfile" 2>&1 && touch "$out.done"
}

for mix in "$@"; do
  name=$mix$tag
  run=mix-$name
  log "train $name"
  if ! scripts/supervise_training.sh "$config" "$run" "$gpu" "configs/data/sweep/$mix.json"; then
    log "FAILED train $name"
    continue
  fi
  checkpoint=$(ls -d "outputs/$run"/step_*.orbax | sort | tail -n 1)
  log "evaluate $name"
  evaluate scripts/eval_domains.py "$results/${name}__domains.json" "logs/sweep_$name.log" \
    --checkpoint "$checkpoint" --window 2048 --data "${heldout[@]}" || log "FAILED domains $name"
  if [ "${#dolma[@]}" -gt 0 ]; then
    evaluate scripts/eval_domains.py "$results/${name}__dolma3.json" "logs/sweep_$name.log" \
      --checkpoint "$checkpoint" --window 2048 --data "${dolma[@]}" || log "FAILED dolma3 domains $name"
  fi
  # shellcheck disable=SC2086
  evaluate scripts/eval_benchmarks.py "$results/${name}__benchmarks.json" "logs/sweep_$name.log" \
    --checkpoint "$checkpoint" --tasks $tasks || log "FAILED benchmarks $name"
  log "done $name"
done
log "finished$tag: $*"
