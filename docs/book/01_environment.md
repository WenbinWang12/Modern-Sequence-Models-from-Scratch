# 1. 环境与第一次训练

## 问题

从 scratch 训练最容易出错的地方不是模型公式，而是环境、数据、随机种子、输出目录和恢复方式没有固定下来。第一章只做一件事：在一台新机器上准备环境，生成示例数据，完成四步训练，并确认输出目录可恢复。

## 代码

核心入口位于：

- `scripts/bootstrap_linux.sh`：安装 uv、Python 3.12.16 和锁定依赖；
- `scripts/prepare_sample.sh`：生成示例 token 数据；
- `scripts/run_gpu_train.sh`：设置 GPU 运行时变量并调用 `ttt.train`；
- `configs/runs/tiny_gpu.json`：最小 GPU 配置；
- `ttt/train.py`：训练循环和检查点保存。

如果机器暂时没有 GPU，可以把 `TTT_INSTALL_CUDA=0`，然后使用 `scripts/run_train.sh`。正式 GPU 实验仍应先执行 `check_environment.py --require-gpu`。

## 实验

```bash
git clone https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch.git
cd Modern-Sequence-Models-from-Scratch
git checkout main

TTT_INSTALL_CUDA=1 bash scripts/bootstrap_linux.sh
python3 scripts/check_environment.py --require-gpu --min-gpus 1
bash scripts/prepare_sample.sh outputs/sample-tokens.npy

bash scripts/run_gpu_train.sh \
  --config configs/runs/tiny_gpu.json \
  --data outputs/sample-tokens.npy \
  --output outputs/tiny-gpu
```

训练输出至少包含 `experiment.json`、`environment.json`、`metrics.jsonl`、`status.json`、`latest.json` 和检查点。第一步通常包括 XLA 编译，不把它当成稳态吞吐。

## 证据

远端 CPU 已经用同一套通用入口完成示例数据准备、4 步训练和 `--auto-resume`。记录见 [通用 Linux 入口远端记录](../reports/remote-generic-run.md)。

## 下一步

下一章会把“代码能运行”变成“实验能复现”：固定 commit、环境摘要、数据 SHA256、配置和检查点路径。
