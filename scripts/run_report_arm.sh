#!/usr/bin/env bash
# 在指定的单张 GPU 上运行一个技术报告实验臂；重复执行会从最新 checkpoint 续训。
# 用法：scripts/run_report_arm.sh <config.json> <run-name> <gpu-index> [data-recipe]
set -euo pipefail

config=$1
name=$2
gpu=$3
data=${4:-configs/data/fineweb_pg19_50_50.json}
python=${TTT_PYTHON:-.venv/bin/python}

mkdir -p "outputs/$name"
export CUDA_VISIBLE_DEVICES=$gpu
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}
export PYTHONPATH=.
exec "$python" -m ttt.train --config "$config" --data "$data" --output "outputs/$name" --auto-resume
