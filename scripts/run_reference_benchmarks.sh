#!/usr/bin/env bash
# 用同一套零样本协议评测公开参考模型，并用 lm-evaluation-harness 交叉核对 GPT-2 的结果。
# 参考模型用 PyTorch 加载，放在独立环境 .venv-ref / .venv-lmeval 里，不影响训练环境。
#
# 用法：scripts/run_reference_benchmarks.sh <gpu-index>
set -uo pipefail

gpu=$1
tasks="hellaswag piqa arc_easy arc_challenge winogrande lambada_openai sciq openbookqa wikitext blimp mmlu pg19"
out=outputs/benchmarks-v2/reference
mkdir -p "$out" outputs/benchmarks-v2/lm-eval logs

[ -x .venv-ref/bin/python ] || {
  uv venv .venv-ref --python 3.12
  uv pip install --python .venv-ref/bin/python torch transformers numpy pyarrow
}

run() { # run <hub-id> [extra eval_benchmarks.py args]
  local name=$1 target
  shift
  target=$out/$(echo "$name" | tr '/' '_').json
  [ -f "$target.done" ] && return 0
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpu TOKENIZERS_PARALLELISM=false .venv-ref/bin/python scripts/eval_benchmarks.py \
    --hf-model "$name" --device cuda --tokens-per-batch 8192 --tasks $tasks \
    --text-dir data/raw/pg19/test --output "$target" "$@" && touch "$target.done"
}

run gpt2
run distilgpt2
run gpt2-medium
run EleutherAI/pythia-70m
run EleutherAI/pythia-160m
run EleutherAI/gpt-neo-125m
run facebook/opt-125m --always-prefix # OPT 的 tokenizer 默认在开头加 BOS，这里沿用它的用法
run HuggingFaceTB/SmolLM-135M
run HuggingFaceTB/SmolLM2-135M

# 交叉核对：参考实现在 CPU 上评测 GPT-2（124M）。
[ -x .venv-lmeval/bin/python ] || {
  uv venv .venv-lmeval --python 3.12
  uv pip install --python .venv-lmeval/bin/python "lm-eval[hf]" torch transformers
}
if ! ls outputs/benchmarks-v2/lm-eval/gpt2/*/results_*.json >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES='' TOKENIZERS_PARALLELISM=false .venv-lmeval/bin/python -m lm_eval --model hf \
    --model_args pretrained=gpt2 --device cpu --batch_size 32 \
    --tasks hellaswag,piqa,arc_easy,arc_challenge,winogrande,lambada_openai,wikitext,sciq,openbookqa \
    --output_path outputs/benchmarks-v2/lm-eval/gpt2
fi
