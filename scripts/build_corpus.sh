#!/usr/bin/env bash
# 一键构建一份语料：下载、建评测索引、构建、复查、汇总。在仓库根目录运行。
#
# 用法：scripts/build_corpus.sh <名称> [需要互相排除的其他语料名称...]
#   scripts/build_corpus.sh v2            自己挑选来源的多领域语料
#   scripts/build_corpus.sh dolma3 v2     Olmo 3 预训练配比的抽样，与 v2 的训练集和 held-out 互不重叠
#
# <名称> 对应 configs/data/corpus_<名称>.json。原始文件和 token 分片写到 TTT_CORPUS_SCRATCH
# （默认 /dev/shm/$USER-ttt，是内存盘，机器重启后需要重新运行）；data/tokens/<名称> 是指向分片目录
# 的符号链接，构建记录（来源清单、分片清单、复查和汇总结果）另存到 data/corpus_meta/<名称>。
# 输入文件版本固定，重复构建得到逐字节相同的分片和清单。已有构建结果时需要 TTT_CORPUS_FORCE=1
# 才会重建。原始文件在构建成功后删除，设置 TTT_CORPUS_KEEP_RAW=1 可以保留。
set -euo pipefail

name=$1
shift
python=${TTT_PYTHON:-.venv/bin/python}
scratch=${TTT_CORPUS_SCRATCH:-/dev/shm/$USER-ttt}
processes=${TTT_CORPUS_PROCESSES:-64}
spec=configs/data/corpus_$name.json
raw=$scratch/raw
evals=$scratch/evals/$name
tokens=$scratch/tokens/$name
meta=data/corpus_meta/$name
export PYTHONPATH=.

exclude=()
for other in "$@"; do
  [ -e "$scratch/tokens/$other/report.json" ] || { echo "要排除的语料还没有构建：$other" >&2; exit 1; }
  exclude+=(--exclude "$scratch/tokens/$other")
done
if [ -e "$tokens/report.json" ] && [ "${TTT_CORPUS_FORCE:-0}" != 1 ]; then
  echo "已有构建结果：$tokens（设置 TTT_CORPUS_FORCE=1 重建）" >&2
  exit 1
fi
rm -rf "$evals" "$tokens"
mkdir -p "$raw" "$tokens" data/tokens
if [ -L "data/tokens/$name" ] || [ ! -e "data/tokens/$name" ]; then
  ln -sfn "$tokens" "data/tokens/$name"
fi

"$python" scripts/build_corpus.py download --spec "$spec" --raw "$raw" --workers 12

# 评测索引：零样本 benchmark 的全部题目，加上 PG-19 的验证和测试书籍。
"$python" scripts/build_corpus.py index-evals \
  --jsonl "$PWD"/data/benchmarks/*.jsonl "$PWD/data/public/arc/arc_challenge_test.jsonl" \
  "$PWD/data/public/gpqa/gpqa_test.jsonl" "$PWD/data/public/gsm8k/test.jsonl" \
  "$PWD/data/public/humaneval/HumanEval.jsonl" "$PWD/data/public/piqa/valid.jsonl" \
  "$PWD/data/public/wikitext2/test.jsonl" "$PWD/data/public/wikitext2/valid.jsonl" \
  --json "$PWD/data/public/bbh/sports_understanding.json" \
  --text-dir "$PWD/data/raw/pg19/validation" "$PWD/data/raw/pg19/test" \
  --output "$evals"

"$python" scripts/build_corpus.py build --spec "$spec" --raw "$raw" --evals "$evals" --output "$tokens" \
  --processes "$processes" ${exclude[@]+"${exclude[@]}"}
"$python" scripts/build_corpus.py verify --spec "$spec" --evals "$evals" --output "$tokens" \
  --processes "$processes" --report "$tokens/verify.json" ${exclude[@]+"${exclude[@]}"}
"$python" scripts/build_corpus.py report --spec "$spec" --output "$tokens" --report "$tokens/report.json"

# 构建记录放进 data/（不在内存盘上），原始文件默认不保留。
sources=$("$python" -c "import json, sys; print(' '.join(s['name'] for s in json.load(open(sys.argv[1]))['sources']))" "$spec")
rm -rf "$meta"
mkdir -p "$meta/raw" "$meta/tokens" "$meta/evals"
for source in $sources; do
  cp "$raw/$source/source_manifest.json" "$meta/raw/$source.source_manifest.json"
done
for directory in "$tokens"/*/; do
  cp "$directory/manifest.json" "$meta/tokens/$(basename "$directory").manifest.json"
done
cp "$tokens/verify.json" "$tokens/report.json" "$meta/"
cp "$evals/index.json" "$meta/evals/"
if [ "${TTT_CORPUS_KEEP_RAW:-0}" != 1 ]; then
  for source in $sources; do
    rm -rf "${raw:?}/$source"
  done
fi
