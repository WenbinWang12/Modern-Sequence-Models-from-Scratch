# 1　环境：第一次训练之前先固定什么

## 环境不是安装步骤的附录

从 scratch 项目经常出现一个奇怪现象：代码在作者机器上运行，换一台机器后不是依赖找不到，就是 loss 曲线完全不同。问题未必来自模型，可能来自 Python 小版本、默认 dtype、JAX 后端、CPU/GPU 设备发现、随机数、文件系统或输出目录。

所以环境是实验定义的一部分。我们希望别人拿到仓库后能回答：使用什么代码、什么依赖、什么设备、什么数据、什么命令，才能得到这个结果。只写“安装 JAX”远远不够，JAX 的 CPU 与 CUDA wheel、驱动兼容性和默认后端都会改变运行行为。

## 三层环境合同

把运行环境拆成三层，排错会更容易。

**系统层**包括操作系统、内核、GPU 型号、驱动、CUDA、磁盘和网络文件系统。它决定设备能否被发现、编译缓存放在哪里，以及多个进程能否访问同一目录。

**Python 层**包括 Python 版本、JAX、Optax、Orbax 和 tokenizer 扩展。它决定 API 和数值实现。`uv.lock` 固定了依赖版本，但不替代驱动与 CUDA 运行时检查。

**实验层**包括 Git commit、解析后的模型配置、tokenizer、数据摘要、batch、sequence length、seed 和输出路径。它决定这一次究竟训练了什么。

| 层 | 典型失败 | 应保存的证据 |
|---|---|---|
| 系统 | 找不到 GPU、OOM、通信失败 | device、driver、backend、文件系统 |
| Python | import 失败、API 变化 | Python 和包版本 |
| 实验 | loss 不一致、恢复错位 | commit、config、data hash、seed |

这三层不能互相替代。一个正确的环境不保证配置正确，一个正确的配置也不保证 GPU 有足够显存。

## 从仓库入口开始

`scripts/bootstrap_linux.sh` 在 Linux 节点下载 uv，安装指定 Python，并按照 extra 同步依赖。它默认包含开发、checkpoint 和 serving 依赖；GPU 作业通过 `TTT_INSTALL_CUDA=1` 添加 CUDA extra。

`scripts/check_environment.py` 使用 JAX 报告当前后端和设备数量。`--require-gpu` 只在至少有目标 GPU 数量时通过，它不验证模型能否在目标显存中编译；那是后面的功能和规模验证。

`scripts/record_environment.sh` 将运行时信息保存为 JSON。训练入口还会在输出目录写入 `environment.json`，这样报告不依赖终端日志。

在目标 Linux 环境中，第一次准备可以这样做：

~~~bash
TTT_INSTALL_CUDA=1 bash scripts/bootstrap_linux.sh
python3 scripts/check_environment.py \
  --require-gpu \
  --min-gpus 1 \
  --output outputs/environment-check.json
~~~

没有 GPU 的 CPU 节点使用 `TTT_INSTALL_CUDA=0`。本项目约定 CPU 只用于功能与数据路径验证，GPU 吞吐不能用 CPU 结果代替。

## 为什么先生成样例数据

直接拿正式语料启动，会把数据问题和模型问题混在一起。`scripts/prepare_sample.sh` 生成一个很小的一维 token 文件，让我们先检查：

- token 是否位于 vocab_size 范围；
- 数据长度是否能组成 sequence_length+1；
- BatchStream 是否能取窗口；
- loss mask 是否有有效位置；
- checkpoint 是否包含正确的数据指纹。

它不能证明语言能力，也不能证明清洗策略。它的作用类似电路板上的通断测试：先确认接口连接，再接入复杂输入。

## 第一次训练的四个阶段

一次训练的第一步不等于模型已经开始稳态计算。通常包含：

1. 进程启动与配置解析；
2. 参数初始化和随机 key 派生；
3. JAX tracing 与 XLA 编译；
4. 编译后的执行和指标写入。

因此 `metrics.jsonl` 中的第一条耗时可能很大。报告吞吐时要区分 compile time 与 steady-state tokens/s，否则一次编译时间会被平均到短实验里，得出没有意义的性能数字。

训练目录还应出现：

~~~text
experiment.json   # 完整输入和解析后的配置
environment.json  # Python、JAX、设备和源码摘要
metrics.jsonl     # 每步 loss、梯度、速度
status.json       # 成功、失败或中断状态
latest.json       # 最近检查点
step_*.npz        # tiny 配置的检查点
~~~

不同 checkpoint 格式的文件名可能不同，但身份和状态的含义不能改变。

## 运行失败时先看哪一层

如果 `check_environment.py` 找不到 GPU，先看后端和设备，而不是改模型。若设备存在但初始化 OOM，检查 dtype、预分配比例、batch、sequence length 和模型配置。若初始化通过而编译失败，保留 XLA 日志和配置，确认 shape 与 sharding。

若训练跑完但恢复结果不同，先对比 data fingerprint、采样器状态、optimizer state 和 global step。最后才检查随机浮点差异。把所有失败都归结为“随机性”会让真正的状态错误长期存在。

## 本章的完成标准

完成环境章需要一个可审计的目录，而不是一条绿色命令：

- 代码 commit 可以从输出中找到；
- 设备和依赖版本已经保存；
- 样例 token 有明确摘要；
- tiny 训练产生有限 loss；
- checkpoint 能被读取；
- auto-resume 识别已有状态；
- 首步编译没有被误报成稳态吞吐。

下一章把这些文件组织成实验身份。环境解决“在哪里跑”，可复现运行解决“这一次跑的是什么”。
