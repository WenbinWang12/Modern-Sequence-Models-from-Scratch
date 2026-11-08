# 1. 环境与第一次训练

## 先回答一个工程问题

从 scratch 训练最容易失控的地方，通常不是模型公式，而是运行环境没有被当成实验的一部分。Python 小版本、JAX 和 CUDA 版本、默认设备、随机种子、数据路径和输出目录，只要有一项随机器变化，后面的 loss 曲线就很难比较。

因此本章不追求训练一个“大”模型，而是建立一条最短的可信路径：在一台新机器上安装依赖，生成一份可追踪的样例 token，完成几步训练，保存 checkpoint，并且能够从同一个输出目录继续运行。后面的所有章节都复用这条路径。

## 环境的最小约定

仓库把环境分成三层：

| 层 | 固定内容 | 记录位置 |
|---|---|---|
| 系统层 | GPU 型号、驱动、CUDA、文件系统 | environment.json |
| Python 层 | Python、JAX、Optax、依赖版本 | pyproject.toml、锁文件 |
| 实验层 | commit、配置、数据摘要、随机种子 | experiment.json |

这三层分别回答“机器是什么”“代码依赖什么”“这次运行做了什么”。不要把 nvidia-smi 的输出或一段终端历史当成唯一记录；训练目录应该能脱离聊天记录独立说明自己。

## 代码入口

- scripts/bootstrap_linux.sh 安装 uv、Python 3.12 和锁定依赖；无 CUDA 机器可设置 TTT_INSTALL_CUDA=0。
- scripts/check_environment.py 检查 Python、JAX 后端、设备数和关键依赖。
- scripts/prepare_sample.sh 从 examples/sample.txt 生成 token 文件，并打印 token 数和摘要。
- scripts/run_train.sh 运行通用 CPU/单设备训练；scripts/run_gpu_train.sh 增加 GPU 运行时变量。
- ttt/train.py 负责配置解析、训练步、日志和 checkpoint；训练逻辑不写在 shell 脚本里。

这种分层很重要：shell 入口负责让人容易启动，Python 模块负责让机器容易复用。将来换成 PBS、Slurm 或容器时，只需要替换调度层，实验语义仍然由同一组 Python 入口定义。

## 第一次运行

下面的命令应在目标 Linux 机器或 GPU 作业中执行。main 只是示例；正式结果应固定到一个明确的 commit。

~~~bash
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
~~~

训练目录至少应包含 experiment.json、environment.json、metrics.jsonl、status.json、latest.json 和一个或多个 checkpoint。第一次 step 可能包含 XLA 编译，吞吐要在编译完成后再测量。若没有 GPU，可执行同一流程的 scripts/run_train.sh，但 CPU 结果只用于功能验证，不用于报告 GPU 吞吐。

## 如何判断这一步完成

本章的完成标准不是“命令没有报错”，而是以下事实都能从文件中核对：配置中的词表和序列长度被加载；loss 产生有限数值；checkpoint 可以被重新读取；输出目录中保存了 commit、环境和数据摘要；再次使用 --auto-resume 时不会从零开始。

远端 CPU 已经用同一套通用入口完成样例数据、4 步训练和 --auto-resume，记录见 [通用 Linux 入口远端记录](../reports/remote-generic-run.md)。这份记录的价值在于它把“仓库可启动”与“某一台机器偶然可启动”区分开来。

## 下一章要固定什么

环境能够运行以后，还必须回答“这次运行到底是哪一次”。下一章把 commit、配置、数据 SHA256、随机种子、恢复策略和输出目录组织成一个实验记录，使两个结果可以逐项比较。
