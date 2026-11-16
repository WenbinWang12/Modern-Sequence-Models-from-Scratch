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

如果需要一个可联网获取的公开小语料来验证下载和清单流程，可以运行：

```bash
bash scripts/download_wikitext2_sample.sh data/external/wikitext-2
```

脚本下载公开 GitHub 镜像中的 WikiText-2 train 文件，固定完整文件 SHA256，并生成前 1 MiB 且不含半行的样本和 `metadata.json`。它不会把样本伪装成代表性预训练语料；正式实验仍需单独完成许可、清洗、切分、tokenizer 和去重验收。

正式数据使用带 SHA256 清单的分片格式：

```bash
python -m ttt.data --input corpus.jsonl --output data/pretrain --sharded --tokenizer byte
```

数据文件、tokenizer 文件和许可信息不随仓库提交。训练恢复时会校验数据指纹、序列长度、batch size 和随机种子。

### 检查已有的 `.idx/.bin` 数据

公开语料或集群共享目录中常见的 Megatron indexed dataset 由同名的 `.idx` 和 `.bin` 两个文件组成。`.idx` 保存版本、整数类型、样本数量、每条样本的长度和字节偏移；`.bin` 只保存连续的 token ID。它不是文本文件，所以用 `vim` 打开出现乱码是正常的。先读取索引，再只读取少量 token，可以避免扫描数百 GB 的语料。

仓库提供只读检查器，不依赖 NumPy：

```bash
python3 scripts/inspect_indexed_dataset.py \
  --idx /path/to/1.idx --bin /path/to/1.bin \
  --sequences 3 --tokens 32
```

如果模型配置的词表大小已知，可以让检查器同时验证抽样 token：

```bash
python3 scripts/inspect_indexed_dataset.py /path/to/1 --vocab-size 131072
```

输出是 JSON，至少应核对 `magic`、`dtype`、`sequence_count`、`document_count`、`sequence_lengths`、`sequence_offsets` 和 `.bin` 文件大小。`sequence_lengths` 与 `sequence_offsets` 是抽样记录，抽样条数写在 `sampled_sequence_count`；`last_sequence` 单独记录并校验最后一条序列。提供 `--vocab-size` 后，报告还会记录抽样 token 的范围检查；它只覆盖抽样 token，不等于完整数据扫描。对于本书当前服务器上的样例，预期能看到 `magic=MMIDIDX`、`dtype=int32`、每条样本长度为 `4097`；这只说明文件布局可读，不说明 tokenizer 一定匹配。若要把报告保存下来，追加 `--output outputs/indexed-dataset-report.json`。正式训练前还应记录 tokenizer 文件摘要、词表大小、特殊 token ID、数据许可和训练/验证切分。

检查器不会修改输入文件，也不会把整个数据集载入内存。它只读取索引头、请求的索引记录和对应的少量 token；因此可以先在登录节点完成格式检查，再在 GPU 作业中使用相同路径启动训练。

如果这份检查报告对应即将启动的训练输入，可以把它绑定到运行记录。训练启动时会把报告路径、字节数和 SHA256 写入 `run_metadata.json`；环境变量指向不存在的文件时会直接失败：

```bash
python3 scripts/inspect_indexed_dataset.py /path/to/1 --output outputs/indexed-report.json
TTT_DATA_INSPECTION_REPORT=outputs/indexed-report.json \
  python -m ttt.train --config configs/tiny.json \
  --data /path/to/converted/manifest.json --output outputs/tiny
```

这里的 `converted/manifest.json` 表示已经把外部 indexed dataset 接入本仓库 loader 的适配清单；当前训练入口不会把原始 `.idx` 文件误当成 JSON manifest。检查报告的摘要证明“启动时登记了哪份报告”，不替代 tokenizer、词表、许可和采样协议的验收。

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
  --initialize-from outputs/tiny-gpu/step_00000004.orbax
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
