# 可复现实验流程

本页描述一台新的 Linux GPU 机器从 clone 到训练、恢复和验证的完整路径。命令默认在仓库根目录执行，所有输出都写入 `outputs/`，运行元数据保存在运行目录中。

## 获取代码和环境

```bash
git clone https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch.git
cd Modern-Sequence-Models-from-Scratch
git checkout main
git rev-parse HEAD
TTT_INSTALL_CUDA=1 bash scripts/bootstrap_linux.sh
bash scripts/record_environment.sh --output environment.json
python3 scripts/check_environment.py --require-gpu --min-gpus 1 --output outputs/environment-check.json
```

默认使用 Python 3.12.16、JAX 0.4.38 和 CUDA extra。机器需要安装 NVIDIA 驱动；CUDA 运行时由锁定的 JAX CUDA 依赖提供。CPU 调试使用 `TTT_INSTALL_CUDA=0`。

## 准备数据

```bash
bash scripts/prepare_sample.sh outputs/sample-tokens.npy
```

正式数据使用带 SHA256 清单的分片格式：

```bash
python -m ttt.data --input corpus.jsonl --output data/pretrain --sharded --tokenizer byte
```

数据文件、tokenizer 文件和许可信息不随仓库提交。训练恢复时会校验数据指纹、序列长度、batch size 和随机种子。

## 从随机初始化开始训练

单卡或单进程 GPU：

```bash
bash scripts/run_gpu_train.sh \
  --config configs/runs/tiny_gpu.json \
  --data outputs/sample-tokens.npy \
  --output outputs/tiny-gpu
```

训练目录应包含 `experiment.json`、`environment.json`、`metrics.jsonl`、`status.json`、`latest.json` 和检查点。第一步可能包含 XLA 编译时间，不能用来估计稳定吞吐。

## 恢复和改变训练阶段

自动从输出目录中的 `latest.json` 恢复：

```bash
bash scripts/run_gpu_train.sh \
  --config configs/runs/tiny_gpu.json \
  --data outputs/sample-tokens.npy \
  --output outputs/tiny-gpu \
  --auto-resume
```

从已有参数开始新的训练阶段：

```bash
bash scripts/run_gpu_train.sh \
  --config configs/runs/tiny_gpu.json \
  --data outputs/sample-tokens.npy \
  --output outputs/finetune-stage \
  --initialize-from outputs/tiny-gpu/run/step_00000004.orbax
```

`--resume` 要求实验配置和优化器状态完全一致；`--initialize-from` 会重新建立优化器，适合改变训练阶段。

## 多 GPU mesh

在一个已有多 GPU 分配中运行：

```bash
bash scripts/run_gpu_train.sh \
  --config configs/runs/mesh_tiny.json \
  --data outputs/sample-tokens.npy \
  --output outputs/mesh-tiny
```

`parallelism="mesh"` 和 `expert_parallel_size=2` 要求至少两张可见设备。多机运行还必须由调度器设置 `JAX_COORDINATOR_ADDRESS`、`JAX_PROCESS_COUNT` 和 `JAX_PROCESS_ID`，并把检查点目录放在所有进程可见的共享文件系统上。

## 验证

```bash
bash scripts/validate_gpu.sh outputs/gpu-validation.json
bash scripts/validate_mesh.sh outputs/mesh-validation.json outputs/mesh-checkpoint
```

验证报告会保存环境、代码摘要、编译时间、训练时间、梯度有限性、流式一致性、分片形状和检查点恢复结果。GPU 验证结果不代表语言质量、长上下文能力或正式吞吐。

## 复现记录

每次正式实验至少保存：Git commit、`environment.json`、`source_sha256`、完整实验配置、tokenizer 版本、数据清单及 SHA256、GPU 型号和驱动、训练日志、检查点路径、恢复点、验证报告、命令行和环境变量。

实验书网站使用 `gh-pages` 分支发布。修改 `docs/` 或 `mkdocs.yml` 后，Documentation workflow 会构建静态页面并更新该分支；GitHub 仓库设置应选择 `Settings → Pages → Deploy from a branch → gh-pages / root`。

官方权重、完整 RULER、32K/128K 长上下文和 RL/Agent 实验仍需要额外数据、权重、环境和评测定义。
