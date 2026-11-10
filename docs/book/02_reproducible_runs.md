# 3　可复现实验：把一次运行变成一个有身份的对象

## 本章导读：中断一次 tiny run，再证明它真的接上了

“能打开 checkpoint”不是恢复实验。我们要证明的是：连续运行和中断后恢复运行，在恢复点之后读取相同的输入、使用相同的 step 和 optimizer state，并在数值容忍范围内得到相同的下一步结果。本章使用上一章的 <code>tiny-contract-001-C</code>，把这个问题变成一条可以检查的路径。

读完本章，你应该能区分 <code>--resume</code> 与 <code>--initialize-from</code>，解释 RNG 为什么是实验状态，并从 <code>status.json</code>、<code>metrics.jsonl</code> 和 checkpoint 判断一次失败是否可恢复。

## 预备知识和结果表

假设 tiny run 在 step 2 保存 checkpoint，连续路径继续到 step 4，中断路径从 step 2 恢复后也继续到 step 4：

| 检查项 | 连续路径 | 恢复路径 | 成功判据 |
|---|---:|---:|---|
| 起点 | step 0 | step 2 | 恢复路径加载同一 checkpoint |
| 下一批数据 | RNG 的第三次抽样 | 恢复 RNG 后第一次抽样 | tokens 与 masks 逐元素一致 |
| optimizer step | 2→3 | 2→3 | 不回到 0 |
| 有效目标数 | 本章普通 NPY 样例为 16 | 由返回的 mask 求和 | 与连续路径一致，不从名义吞吐推断 |
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

`--initialize-from` 表示使用已有参数作为新实验的初始点。它重新创建 optimizer、step 和数据顺序，因此需要新的输出目录。当前入口只接受相同参数树、叶张量形状和 dtype，例如结构相同的 C 与 F 可以共享初始权重。没有 MLP′ 的 dense 模型不能用这一个参数直接变成新增 MLP′ 的模型；这需要单独的权重迁移规则，当前入口没有实现。用初始化替代恢复，会破坏训练曲线的解释。

## 一个最小的失败案例

假设训练在 global step 100 保存 checkpoint。这个 checkpoint 中的参数已经看过 100 个 batch，AdamW 也积累了对应的动量。若只复制 params 到新目录，再把 step 写成 100，下一步虽然能够运行，但它已经不是原实验的 continuation：

1. 优化器的一阶、二阶统计被清零，更新方向会突然改变；
2. 学习率 schedule 可能从 step 0 开始，或从 step 100 开始但使用了不匹配的 optimizer state；
3. BatchStream 的 RNG 没有恢复，下一批数据可能与原实验不同；
4. 数据 manifest 或 sequence length 变化后，step 仍然看起来连续，实际消耗的 token 已经不同。

恢复检查不能只问“文件能否打开”。至少要用同一个初始 checkpoint 做一次短实验：一条路径连续运行，另一条路径在中间保存并 resume，然后比较恢复点之后的下一批输入、step、有效 token 和 loss。低精度归约可能有小数差异，但输入顺序和状态字段不能悄悄变化。

相反，从同结构的 C 初始化 F 时，应明确产生新的身份：

~~~text
C:      source=S1, data=D1, step=1000, optimizer=O_C
F init: source=S1, data=D1, step=0, optimizer=O_F
~~~

这两行共享参数文件，但不是同一次运行；报告不能把第二行的 loss 曲线接在第一行后面。

## 当前 checkpoint 保存什么

`ttt/checkpoint.py` 将 Experiment、TrainState 和数据状态组合起来。实验对象包含模型和训练配置；TrainState 包含 params、opt_state 和 step；数据状态由 BatchStream 保存。恢复时先检查身份，再读取参数，避免在很晚才发现配置不兼容。

检查点不是只有模型权重。对 TTT 来说，快权重有两种不同角色：

- 训练时的 fast weights 是一个序列内部的临时状态，外循环更新静态模型；
- 服务或流式推理时的 fast weights 是 session 状态，需要随会话保存或回滚。

把 session snapshot 当成训练 checkpoint，会让优化器和数据采样状态缺失；把训练 checkpoint 当成会话 snapshot，又会携带不该暴露给请求的全局状态。两者需要不同的 schema。

## 运行目录应能自己解释自己

先区分训练器当前写出的文件与实验者需要补充的记录。当前训练路径在初始化成功后写入：

| 字段 | 作用 |
|---|---|
| status.json | running、failed、complete、stopped；失败时包含 error_type |
| experiment.json | 解析后的模型、内循环、训练和 tokenizer 配置 |
| environment.json | Python、JAX、设备、驱动、调度作业信息 |
| metrics.jsonl | step、loss、grad_norm、seconds、名义 tokens_seen/tokens_per_second 等 |
| step_*.npz、latest.json | tiny 检查点及最近一次保存的路径 |

训练器没有自动生成 <code>error.json</code> 或独立 <code>checkpoint/</code> 目录。源码 commit、文件 SHA256、完整异常栈和解释报告应另行留档；上章已保存前两项。参数初始化、数据打开和恢复检查可能在写 <code>status.json</code> 之前失败，缺少状态文件不能证明作业从未启动，应查看终端或调度日志。

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

这些问题是完整归档的目标，不是上述 JSON 已自动覆盖的字段。<code>experiment.json</code> 写配置，<code>environment.json</code> 写机器与源码摘要，checkpoint 的 <code>data_state</code> 保存数据指纹和 RNG 状态；<code>metrics.jsonl</code> 的 token 计数是名义输入位置数。Markdown 报告应引用相应原始证据。

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

“没有输出”通常是最难恢复的状态。完整生产启动器应在初始化失败时也留下异常记录；这是外层启动协议需要补充的能力，当前训练入口没有覆盖所有这类失败。

## 如何比较两个结果

比较模型质量时，优先固定有效 token 预算和数据 manifest。比较系统效率时，还要固定设备、精度、batch、序列长度和编译策略。比较流式状态时，固定初始 cache、位置和 session seed。

可以把差异分成三类：

**定义差异**：词表、模型结构、训练 token 不同。它们构成新实验，不能用“同配置”描述。

**环境差异**：GPU 型号、驱动或 XLA 版本不同。质量可能仍可比较，但性能必须分开。

**数值差异**：同一输入下并行归约、低精度或非确定性带来的小差异。需要给出容忍范围，而不是把所有差异都叫失败。

## 从一个确定的检查点验证恢复

四步 tiny 运行可能在人工按下中断键前已经完成。这里先完成连续路径，再从它保留的 step 2 检查点在另一个目录继续。检查点保存的是当时的完整状态，因此这能检验从保存点恢复的数值语义；它不检验突然断电、损坏文件或调度器信号处理。使用两个尚不存在的输出目录，在目标 Linux CPU 节点运行：

~~~bash
set -e
test ! -e outputs/recovery-continuous
test ! -e outputs/recovery-resumed
bash scripts/prepare_sample.sh outputs/sample-tokens.npy
JAX_PLATFORMS=cpu bash scripts/run_train.sh \
  --config configs/book/tiny_control.json \
  --data outputs/sample-tokens.npy \
  --output outputs/recovery-continuous
JAX_PLATFORMS=cpu bash scripts/run_train.sh \
  --config configs/book/tiny_control.json \
  --data outputs/sample-tokens.npy \
  --output outputs/recovery-resumed \
  --resume outputs/recovery-continuous/step_00000002.npz
~~~

连续路径应记录 step 1–4，恢复路径只记录 step 3–4。不要改配置中的总步数来制造中断：当前 resume 要求解析后的实验配置一致。<code>--auto-resume</code> 适合继续同一目录最新状态；对已经完成的 step 4 目录再次调用它，不会多训练一步。

随机数生成器可看成一台确定性机器：seed 选择起始状态，每抽取一批数据就推进一次内部状态。重新设同一个 seed 会从第一批重新开始；恢复内部状态才会接着抽第三批。下面从 seed 连续抽两次，与从 checkpoint 恢复后抽下一次做直接比较，再比较两条路径的 step 3 loss、最终参数和优化器：

~~~bash
JAX_PLATFORMS=cpu .venv/bin/python - <<'PY'
import json
from pathlib import Path
import jax
import numpy as np
from ttt.checkpoint import load_checkpoint
from ttt.data import BatchStream

continuous = Path("outputs/recovery-continuous")
resumed = Path("outputs/recovery-resumed")
middle = load_checkpoint(continuous / "step_00000002.npz", load_optimizer=True)
config = middle.experiment

def new_stream():
    return BatchStream(
        "outputs/sample-tokens.npy",
        sequence_length=config.training.sequence_length,
        batch_size=config.training.batch_size,
        vocab_size=config.model.vocab_size,
        seed=config.training.seed,
    )

reference, restored = new_stream(), new_stream()
for _ in range(int(middle.state.step)):
    reference.next_batch_with_masks()
restored.load_state_dict(middle.data_state)
expected = reference.next_batch_with_masks()
observed = restored.next_batch_with_masks()
for left, right in zip(expected, observed):
    np.testing.assert_array_equal(left, right)

def metrics(path):
    rows = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    return {row["step"]: row for row in rows}

full_log, resumed_log = metrics(continuous), metrics(resumed)
assert sorted(full_log) == [1, 2, 3, 4]
assert sorted(resumed_log) == [3, 4]
np.testing.assert_allclose(full_log[3]["loss"], resumed_log[3]["loss"], rtol=1e-5, atol=1e-6)
left = load_checkpoint(continuous / "step_00000004.npz", load_optimizer=True)
right = load_checkpoint(resumed / "step_00000004.npz", load_optimizer=True)
assert jax.tree.structure(left.state) == jax.tree.structure(right.state)
maximum = 0.0
for a, b in zip(jax.tree.leaves(left.state), jax.tree.leaves(right.state)):
    a, b = np.asarray(a), np.asarray(b)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    maximum = max(maximum, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
assert left.data_state == right.data_state
report = {"passed": True, "step": int(right.state.step),
          "next_batch_equal": True, "valid_targets": float(observed[1].sum()),
          "loss_step3": resumed_log[3]["loss"], "max_state_abs_diff": maximum,
          "rtol": 1e-5, "atol": 1e-6}
Path("outputs/recovery-check.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
~~~

只有断言全部完成才会写入 <code>outputs/recovery-check.json</code>。其中 <code>valid_targets</code> 来自实际返回的 mask，而非训练日志中的名义 token 计数。这是待读者在目标环境执行的验证步骤，没有附会一组尚未测得的 loss 数值。若数据比较失败，先看数据指纹与 RNG；若数据相同而状态不同，再检查 optimizer、step、dtype 和数值容差。时间和吞吐无需逐值相等，因为两次运行的编译与系统负载可能不同。

## 本章小结

- resume 恢复同一个优化过程，initialize-from 创建新的实验身份；
- RNG、采样位置、optimizer state 和 global step 都是恢复语义的一部分；
- 运行目录要能描述成功、失败和是否可恢复；
- 恢复实验比较的是状态和下一步行为，不只是最终 checkpoint 文件；
- 下一章把这些身份字段连接到具体的模型形状和词表维度。
