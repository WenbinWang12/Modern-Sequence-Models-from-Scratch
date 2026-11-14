# 2　环境：第一次训练之前先固定什么

## 本章导读：先证明机器能运行

本章不追求一次把大模型训练起来。我们先用 <code>tiny-contract-001</code> 做一件更小但更可靠的事：让机器报告自己是谁，生成一份短 token 文件，启动一次 tiny 训练，并留下可以解释的输出目录。这样，后面 loss 异常时，我们知道问题来自环境、数据还是模型。

读完本章，你应该能在目标 Linux 节点完成三项检查：看到 JAX 的设备列表，检查样例 token 的长度与词表范围，找到 tiny 训练的 <code>experiment.json</code>、<code>environment.json</code>、<code>metrics.jsonl</code> 和 checkpoint。CPU 只承担功能检查，GPU 才用于吞吐和规模结论。

## 预备知识与贯穿输入

你只需要会打开终端、运行 shell 命令，并理解“文件夹中的 JSON 是机器写出的记录”。本章使用 <code>examples/sample.txt</code>；它不是质量足够高的预训练语料，而是用于验证路径的最小输入。

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

`scripts/check_environment.py` 使用 JAX 统计设备，输出 <code>device_count</code>、<code>devices</code> 和 <code>gpu_count</code>。`--require-gpu` 只在至少有目标 GPU 数量时通过，它不验证模型能否在目标显存中编译；那是后面的功能和规模验证。

`scripts/record_environment.sh` 将运行时信息保存为 JSON。训练入口还会在输出目录写入 `environment.json`，这样报告不依赖终端日志。

在目标 Linux 环境中，第一次准备可以这样做：

~~~bash
TTT_INSTALL_CUDA=1 bash scripts/bootstrap_linux.sh
.venv/bin/python scripts/check_environment.py \
  --require-gpu \
  --min-gpus 1 \
  --output outputs/environment-check.json
~~~

成功时，JSON 应有 <code>python</code>、<code>packages.jax</code>、<code>device_count</code>、<code>devices</code>、<code>gpu_count</code> 和 <code>passed</code>。它没有独立的 <code>backend</code> 或 <code>dtype</code> 字段。若 CPU 节点被要求 <code>--require-gpu</code>，这是环境条件不满足。CPU 功能检查应使用 <code>TTT_INSTALL_CUDA=0</code> 安装，并去掉该参数。

这里明确调用 <code>.venv/bin/python</code>，因为 bootstrap 在子 shell 中创建环境，不会激活当前 shell。直接写 <code>python3</code> 可能检查的是系统 Python，随后却由训练脚本使用另一个解释器。

## 为什么先生成样例数据

直接拿正式语料启动，会把数据问题和模型问题混在一起。`scripts/prepare_sample.sh` 生成一个很小的一维 token 文件，让我们先检查：

- token 是否位于 vocab_size 范围；
- 数据长度是否能组成 sequence_length+1；
- BatchStream 是否能取窗口；
- loss mask 是否有有效位置；
- checkpoint 是否包含正确的数据指纹。

它不能证明语言能力，也不能证明清洗策略。它的作用类似电路板上的通断测试：先确认接口连接，再接入复杂输入。

样例输入的第一条检查结果应能回答三个问题：token 数是否至少为 <code>sequence_length + 1 = 17</code>，所有 ID 是否位于 <code>[0, 259)</code>，以及 shift 后是否仍有有效 loss mask。任一问题失败，都先修数据入口，不要修改模型。

## 第一次训练的四个阶段

一次训练的第一步不等于模型已经开始稳态计算。通常包含：

1. 进程启动与配置解析；
2. 参数初始化和随机 key 派生；
3. JAX tracing 与 XLA 编译；
4. 编译后的执行和指标写入。

因此 `metrics.jsonl` 中的第一条耗时可能很大。报告吞吐时要区分 compile time 与 steady-state tokens/s，否则一次编译时间会被平均到短实验里，得出没有意义的性能数字。

训练目录还应出现：

~~~text
experiment.json   # 解析后的模型、内循环、训练和 tokenizer 配置
environment.json  # Python、JAX、设备和源码摘要
run_metadata.json # 配置、输入路径、命令行、Git commit 和 manifest 摘要
metrics.jsonl     # 每步 loss、梯度、速度
status.json       # running、failed、complete 或 stopped
latest.json       # 最近检查点
step_*.npz        # tiny 配置的检查点
~~~

源码 commit 和样例文件 SHA256 需要另外保存；<code>experiment.json</code> 不会自动补入这两个值。<code>environment.json</code> 中的 <code>source_sha256</code> 是包源码摘要，不是 Git commit。下面的最小运行会显式记录它们。

## 运行失败时先看哪一层

如果 `check_environment.py` 找不到 GPU，先看后端和设备，而不是改模型。若设备存在但初始化 OOM，检查 dtype、预分配比例、batch、sequence length 和模型配置。若初始化通过而编译失败，保留 XLA 日志和配置，确认 shape 与 sharding。

若训练跑完但恢复结果不同，先对比 data fingerprint、采样器状态、optimizer state 和 global step。最后才检查随机浮点差异。把所有失败都归结为“随机性”会让真正的状态错误长期存在。

## CPU、GPU、JAX 和 mesh 各自负责什么

如果你第一次接触训练环境，可以先把它们分成四个角色。CPU 是负责启动程序、准备数据和调度工作的通用处理器；GPU 有大量适合矩阵乘法的并行计算单元，负责模型的大部分张量计算；JAX 是把 Python 函数转换成可自动求导、可编译和可并行执行计算图的库；mesh 是我们给多张 GPU 起的一张“设备坐标表”，告诉 JAX 哪个轴切分 batch、参数或专家。

例如，<code>hidden</code> 的形状是 <code>[B, L, d]</code>，其中 <code>B=2</code>、<code>L=4</code>、<code>d=8</code>。单卡运行时，整个数组可以放在一张 GPU 上。若建立一个有两张卡的 <code>data</code> mesh，常见做法是把 batch 轴切成每卡一行：每张卡看到 <code>[1, 4, 8]</code>，但全局结果仍然表示 <code>[2, 4, 8]</code>。这不是把模型复制两遍后再随意相加；每个 rank 必须按照同一份参数、数据分片和 collective 通信规则推进。

JAX 还会把第一次调用分成 tracing、编译和执行。可以把它想成先把菜谱翻译成设备指令，再执行。假设首步需 40 秒、后续每步 2 秒，将它们混合会低估稳态吞吐。这是时间计算示例，未声称本项目测得这些数。当前训练日志的 <code>seconds</code> 包含该步开销，不单独输出编译时间；<code>ttt.validate</code> 的 <code>compilation_seconds</code> 是验证函数的单独编译计时，也不能替代某次训练的编译测量。

## 本章的完成标准

完成环境章需要一个可审计的目录，而不是一条绿色命令：

- 代码 commit 可以从输出中找到；
- 设备和依赖版本已经保存；
- 样例 token 有明确摘要；
- tiny 训练产生有限 loss；
- checkpoint 能被读取；
- auto-resume 识别已有状态；
- 首步编译没有被误报成稳态吞吐。

## 本章的最小 checkpoint 与证据

在已安装依赖的目标节点上，按顺序执行：

~~~bash
bash scripts/prepare_sample.sh outputs/sample-tokens.npy
JAX_PLATFORMS=cpu bash scripts/run_train.sh \
  --config configs/book/tiny_control.json \
  --data outputs/sample-tokens.npy \
  --output outputs/tiny-contract-001-C
git rev-parse HEAD > outputs/tiny-contract-001-C/source-commit.txt
sha256sum outputs/sample-tokens.npy > outputs/tiny-contract-001-C/data-sha256.txt
~~~

首次执行时使用尚未训练过的输出目录；重复完整实验时换一个目录，继续原运行则按下一章使用 resume。GPU 节点应先完成设备检查，再将整段 <code>JAX_PLATFORMS=cpu bash scripts/run_train.sh</code> 替换为 <code>bash scripts/run_gpu_train.sh</code>。成功目录应有上述文件以及 <code>step_00000002.npz</code>、<code>step_00000004.npz</code>；<code>status.json</code> 的状态应为 <code>complete</code>、step 为 4。这些是预期输出，必须以自己的运行产物核对。

日志中的 <code>tokens_seen</code> 和 <code>tokens_per_second</code> 按 batch size × sequence length 计算，包含被 mask 的位置。当前样例是普通 NPY，目标 mask 默认全为 1；后面改成分片监督数据时，还要另记有效目标数。首步可能包含编译时间，不能直接当作稳态吞吐。

若命令失败，先检查样例路径、<code>configs/book/tiny_control.json</code> 和环境记录的 <code>devices</code>。初始化阶段失败可能尚未产生 <code>status.json</code>，应保留终端或调度日志；训练器不会自动生成 <code>error.json</code>。这个检查点只说明 tiny 路径能运行，不证明语言能力。下一章利用 step 2 的检查点检验恢复后的输入与参数是否与连续运行一致。

## 本章小结

- 系统层、Python 层和实验层必须分别记录；
- 样例 token 先验证词表范围、长度、shift 和 mask；
- tracing、编译和稳态执行是不同阶段；
- 预期输出和失败诊断与命令同样重要；
- 下一章将检查这些文件能否组成唯一的实验身份并支持恢复。
