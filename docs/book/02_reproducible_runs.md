# 3　可复现实验：把一次运行变成一个有身份的对象

## 本章导读：中断一次 tiny run，再证明它真的接上了

“能打开 checkpoint”不是恢复实验。我们要证明的是：连续运行和中断后恢复运行，在恢复点之后读取相同的输入、使用相同的 step 和 optimizer state，并在数值容忍范围内得到相同的下一步结果。本章使用上一章的 <code>tiny-contract-001-C</code>，把这个问题变成一条可以检查的路径。

读完本章，你应该能区分 <code>--resume</code> 与 <code>--initialize-from</code>，解释 RNG 为什么是实验状态，并从 <code>status.json</code>、<code>metrics.jsonl</code> 和 checkpoint 判断一次失败是否可恢复。

## 预备知识和结果表

假设 tiny run 在 step 2 保存 checkpoint，连续路径继续到 step 4，中断路径从 step 2 恢复后也继续到 step 4：

| 检查项 | 连续路径 | 恢复路径 | 成功判据 |
|---|---:|---:|---|
| 起点 | step 0 | step 2 | 恢复路径加载同一 checkpoint |
| 下一批索引 | 2→3 | 2→3 | <code>BatchStream</code> 状态一致 |
| optimizer step | 2→3 | 2→3 | 不回到 0 |
| valid token 数 | 16 | 16 | 与连续路径一致 |
| loss | \(L_3\) | \(L'_3\) | \(\lvert L_3-L'_3\rvert\leq\epsilon\) |

这张表说明“恢复成功”是一个状态比较，而不是文件存在检查。

## “同样的命令”并不代表同样的实验

两台机器执行同一条命令，可能读取不同的数据文件、使用不同默认值、从不同 checkpoint 开始，甚至在不同 step 记录第一条 loss。命令是入口，不是实验身份。

我们把一个运行写成：

$$
R=(S,C,I_{\mathrm{data}},E,K,Q_{\mathrm{budget}},O),
$$

其中 \(S\) 是源码 commit，\(C\) 是解析后的配置，\(I_{\mathrm{data}}\) 是数据、tokenizer 和 manifest 的指纹，\(E\) 是环境，\(K\) 是随机与优化状态，\(Q_{\mathrm{budget}}\) 是训练 token、序列长度、batch、步数和成本预算，\(O\) 是输出和证据。复现的目标不是让所有浮点数最后一位相同，而是让这些定义实验的对象一致，并解释不可避免的数值差异。

## resume 和 initialize-from 是两个动词

`--resume` 表示继续同一个优化过程。它必须恢复：

- 模型参数；
- AdamW 的一阶、二阶统计；
- 当前 global step；
- BatchStream 的 RNG state；
- 数据指纹、sequence length、batch size 和 seed；
- 任何影响下一步输入的采样状态。

如果只载入参数，优化器会从零开始，学习率 schedule 也可能回到 warmup。这样得到的是新实验，即使输出目录名字叫 resume。

`--initialize-from` 表示使用已有参数作为新实验的初始点。新实验应该有新的输出目录和 experiment identity，optimizer、step、数据顺序可以重新建立。用它从 dense baseline 初始化 TTT 是合理的；用它替代断点恢复会破坏训练曲线的解释。

## 一个最小的失败案例

假设训练在 global step 100 保存 checkpoint。这个 checkpoint 中的参数已经看过 100 个 batch，AdamW 也积累了对应的动量。若只复制 params 到新目录，再把 step 写成 100，下一步虽然能够运行，但它已经不是原实验的 continuation：

1. 优化器的一阶、二阶统计被清零，更新方向会突然改变；
2. 学习率 schedule 可能从 step 0 开始，或从 step 100 开始但使用了不匹配的 optimizer state；
3. BatchStream 的 RNG 没有恢复，下一批数据可能与原实验不同；
4. 数据 manifest 或 sequence length 变化后，step 仍然看起来连续，实际消耗的 token 已经不同。

恢复检查不能只问“文件能否打开”。至少要用同一个初始 checkpoint 做一次短实验：一条路径连续运行，另一条路径在中间保存并 resume，然后比较恢复点之后的下一批输入、step、有效 token 和 loss。低精度归约可能有小数差异，但输入顺序和状态字段不能悄悄变化。

相反，从 baseline 初始化 TTT 时，应明确产生新的身份：

~~~text
baseline: source=S1, data=D1, step=1000, optimizer=O_dense
TTT init: source=S2, data=D1, step=0, optimizer=O_ttt
~~~

这两行共享参数文件，但不是同一次运行；报告不能把第二行的 loss 曲线接在第一行后面。

## 当前 checkpoint 保存什么

`ttt/checkpoint.py` 将 Experiment、TrainState 和数据状态组合起来。实验对象包含模型和训练配置；TrainState 包含 params、opt_state 和 step；数据状态由 BatchStream 保存。恢复时先检查身份，再读取参数，避免在很晚才发现配置不兼容。

检查点不是只有模型权重。对 TTT 来说，快权重有两种不同角色：

- 训练时的 fast weights 是一个序列内部的临时状态，外循环更新静态模型；
- 服务或流式推理时的 fast weights 是 session 状态，需要随会话保存或回滚。

把 session snapshot 当成训练 checkpoint，会让优化器和数据采样状态缺失；把训练 checkpoint 当成会话 snapshot，又会携带不该暴露给请求的全局状态。两者需要不同的 schema。

## 运行目录应能自己解释自己

失败目录也要有固定字段，哪怕进程只完成了环境初始化：

| 字段 | 作用 |
|---|---|
| status.json | running、failed、completed、resumable 等生命周期状态 |
| error.json | 异常类型、阶段、最后一条错误和 traceback 摘要 |
| experiment.json | commit、解析后的配置、数据与 tokenizer 指纹 |
| environment.json | Python、JAX、设备、驱动、调度作业信息 |
| metrics.jsonl | 已写入的 step、loss、有效 token 和梯度统计 |
| checkpoint/ | 最近一次完整状态，以及是否允许 resume |

这样即使编译阶段失败，也能区分“环境不可用”“配置不合法”和“训练运行后发散”。缺少 status.json 的空目录不能证明作业从未启动，只能说明记录协议没有覆盖初始化失败。

一份可审计的输出目录至少能回答：

~~~text
这次用哪个 commit？
加载了哪个 config？
数据文件和 tokenizer 的摘要是什么？
从哪个 step 开始，消耗了多少有效 token？
机器、后端和 dtype 是什么？
最后一次保存在哪里？
失败发生在哪个阶段？
~~~

`experiment.json` 写输入，`environment.json` 写机器，`metrics.jsonl` 写过程，`status.json` 写生命周期，checkpoint 写状态。Markdown 报告是阅读入口，但不应成为唯一事实来源。

可以把运行信息组织成一个 manifest：

~~~json
{
  "source": {"commit": "<git-sha>"},
  "inputs": {
    "config": "<config-sha256>",
    "data": "<manifest-sha256>",
    "tokenizer": "byte-v1"
  },
  "budget": {"steps": 4, "sequence_length": 16},
  "state": {"start_step": 0, "resume": false}
}
~~~

这段结构是教学示意；仓库的实际字段以训练输出为准。重要的是把输入、预算和状态分开。

## 失败也应成为记录

作业可能在下载依赖、编译、设备初始化、数据读取、梯度更新或保存时失败。失败目录应该保留：

- 运行命令和调度作业 ID；
- 最后解析成功的配置；
- 环境摘要；
- 已写入的 metrics；
- 异常类型与阶段；
- 是否可以安全 resume。

“没有输出”通常是最难恢复的状态。即使进程在初始化阶段失败，也应由外层脚本创建目录并写出 status。

## 如何比较两个结果

比较模型质量时，优先固定有效 token 预算和数据 manifest。比较系统效率时，还要固定设备、精度、batch、序列长度和编译策略。比较流式状态时，固定初始 cache、位置和 session seed。

可以把差异分成三类：

**定义差异**：词表、模型结构、训练 token 不同。它们构成新实验，不能用“同配置”描述。

**环境差异**：GPU 型号、驱动或 XLA 版本不同。质量可能仍可比较，但性能必须分开。

**数值差异**：同一输入下并行归约、低精度或非确定性带来的小差异。需要给出容忍范围，而不是把所有差异都叫失败。

## 本章的实际复现路径

在远端环境中执行：

~~~bash
git rev-parse HEAD
bash scripts/record_environment.sh --output outputs/run/environment.json
bash scripts/run_train.sh \
  --config configs/tiny.json \
  --data outputs/sample-tokens.npy \
  --output outputs/tiny \
  --auto-resume
~~~

随后检查 step、metrics 和 latest checkpoint。对一个中断实验，先用 resume 继续；对一个新架构实验，复制输入记录并使用 initialize-from。书后面的每个架构消融都应遵循这个区分。

下一章进入配置。我们要把词表、隐藏维度、MLP、快权重和训练选项写成一份能被代码校验的架构合同。

## 本章的最小恢复 checkpoint

先生成样例数据并启动一个会保存中间状态的 tiny 运行：

~~~bash
bash scripts/prepare_sample.sh outputs/sample-tokens.npy
bash scripts/run_train.sh \
  --config configs/tiny.json \
  --data outputs/sample-tokens.npy \
  --output outputs/tiny-contract-001-C \
  --auto-resume
~~~

成功时，检查 <code>outputs/tiny-contract-001-C/status.json</code>、<code>metrics.jsonl</code>、<code>latest.json</code> 和 <code>step_*.npz</code>。把作业在 step 2 后中断，再用同一个命令重启；预期是从已有 step 继续，而不是重新写 step 0。若恢复后 loss 或下一批输入不同，先比较数据状态和 optimizer state，再比较低精度归约。

## 本章小结

- resume 恢复同一个优化过程，initialize-from 创建新的实验身份；
- RNG、采样位置、optimizer state 和 global step 都是恢复语义的一部分；
- 运行目录要能描述成功、失败和是否可恢复；
- 恢复实验比较的是状态和下一步行为，不只是最终 checkpoint 文件；
- 下一章把这些身份字段连接到具体的模型形状和词表维度。
