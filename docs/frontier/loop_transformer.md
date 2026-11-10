# Looped Transformer：重复计算，究竟重复了什么

**资料快照：2026-11-11。代码审阅基线：<code>021203e</code>；本轮正文修订见其后续提交。状态：三种共享参数的局部重复已实现；论文协议和质量比较未复现。**

## 为什么同一组参数值得算第二遍

修改文章时，第二遍阅读可能发现第一遍遗漏的问题。共享参数的深度循环也有类似的直觉：同一个变换再次处理已经改变的表示，可能提取第一次没有得到的信息。但这个比喻只解释“为什么值得研究”，没有证明第二遍一定有帮助。若变换丢失信息、不断放大误差，重复甚至会使结果变差。

普通深层 Transformer 为不同层保存不同参数。Looped Transformer 则在某段深度上重复使用同一组参数。它可能减少独立参数的数量，却不会免除重复计算的时间和训练时的激活存储。本专题的问题是：在明确的计算预算下，共享参数带来的有效深度是否值得？这需要把“参数少”“算得快”“质量好”分开测量。

## 来源与局部假设

本节固定引用 [Reasoning with Latent Thoughts: On the Power of Looped Transformers，v1，2025-02-24](https://arxiv.org/abs/2502.17416v1)。该工作研究循环 Transformer 的表达与计算能力。本文据此提出本仓库的实验问题，并不把下述局部循环实现称为论文架构或实验协议的完整复现。

设某个 token 位置的初始表示为 \(h^{[0]}\)，一个共享模块为 \(F_\theta\)。深度方向重复 \(r\) 次得到

$$
h^{[j+1]}=F_\theta(h^{[j]}),\qquad j=0,\ldots,r-1.
$$

这里 \(j\) 是重复次数，token 位置仍用 \(t\)；\(F_\theta\) 可以处理整段序列，因此式子省略了序列维度。每次使用同一份 \(\theta\)，所以这部分独立参数量不随 \(r\) 增加；但必须执行 \(r\) 次变换，训练还可能保存每次的中间值。参数共享与激活共享是两回事。

TTT 则沿序列推进更新快权重 \(\phi_c\)，其中 \(c\) 表示 chunk。Loop 改变深度计算，TTT 改变上下文到来后的权重状态。两者可以组合，但应先单独建立基线，否则组合收益无法归因。不了解梯度与状态的读者可先读[内外循环](../book/05_inner_outer_loop.md)。

## 代码已经实现什么

<code>ttt/models/ttt_e2e/modeling_ttt_e2e.py</code> 对下表的重复使用同一个参数对象。原先“先验证 <code>loop_mode</code>，再增加真正共享层”的表述不准确：共享参数已经存在；尚缺的是与指定论文一致的整体架构、训练对照和质量证据。

| 配置 | 已实现的计算 | 重复的范围 |
|---|---|---|
| <code>none</code> | 每层执行一次 | 无 |
| <code>early_ffn</code> | 同一层的静态 FFN 分支重复执行 | 选定层内的 FFN |
| <code>middle_attention</code> | 同一层的注意力分支重复执行 | 选定层内的注意力 |
| <code>middle_block</code> | 同一层的完整 block 重复执行 | 每个选定层分别重复 |

特别要区分“重复每一层”和“重复一段层”。选择第 0、1 层且重复两次时，当前 <code>middle_block</code> 的顺序是 \(0\to0\to1\to1\)，而整个两层模块循环两次是 \(0\to1\to0\to1\)。相同的层数和参数量不能消除这个计算顺序差异。当前模式不应直接替代任意 recurrent-depth 论文中的循环。

配置使用左闭右开区间 <code>[loop_start, loop_end)</code>，且循环区间必须落在静态前缀中。注意力重复和完整 block 重复为不同访问保存各自的缓存，避免把两个深度位置的历史混在一起。因此，共享参数也不意味着共享 KV 缓存；增加重复次数可能增加缓存和激活开销。

## 一个可审计的最小基线

先关闭快权重和内循环，只改变第一层的重复次数。以下命令应在已准备好环境的远端 GPU 节点、仓库根目录执行。本次编辑没有执行这些命令。样本数据只用于检查训练链路，不是独立验证集。

~~~bash
mkdir -p outputs/frontier-loop
git rev-parse HEAD > outputs/frontier-loop/source-commit.txt
bash scripts/prepare_sample.sh outputs/frontier-loop/sample.npy
.venv/bin/python - <<'PY'
import copy
import json
from pathlib import Path

root = Path("outputs/frontier-loop")
base = json.loads(Path("configs/tiny.json").read_text())
base["model"]["use_fast_mlp"] = False
base["model"]["ttt_suffix_length"] = 0
base["inner"]["enabled"] = False
for repeats in (1, 2, 4):
    config = copy.deepcopy(base)
    config["model"].update(
        loop_mode="none" if repeats == 1 else "middle_block",
        loop_start=0, loop_end=1, loop_repeats=repeats,
    )
    (root / f"r{repeats}.json").write_text(json.dumps(config, indent=2))
PY
for r in 1 2 4; do
  bash scripts/run_gpu_train.sh \
    --config "outputs/frontier-loop/r$r.json" \
    --data outputs/frontier-loop/sample.npy \
    --output "outputs/frontier-loop/run-r$r"
done
~~~

这三个模型都有两个独立 block，仅第一个 block 分别执行 1、2、4 次，因此完整 block 的访问次数为 2、3、5，而不是 2、4、8。<code>none</code> 组是无 TTT、无深度循环的功能基线。三组训练 token 数一致时，FLOPs 通常不一致；这条命令尚未完成计算预算匹配。

每个 <code>run-r*</code> 目录应包含 <code>experiment.json</code>、<code>environment.json</code>、<code>metrics.jsonl</code>、<code>status.json</code>、<code>latest.json</code> 与检查点。按当前 <code>configs/tiny.json</code> 的 4 步、NPZ 设置，最后一个检查点是 <code>step_00000004.npz</code>；修改步数或格式后应以 <code>latest.json</code> 为准。预期状态为 <code>complete</code>，不是预期某个 loss 数值。

另可执行 <code>bash scripts/validate_gpu.sh outputs/frontier-loop/functional.json</code> 检查三种 Loop 模式的梯度、因果性和流式一致性。该工具使用自己的 tiny 配置，不能把它的结果当作上面三个训练配置的逐项验证。任何检查都没有提供推理能力的任务分数。

## 公平比较需要控制什么

第一组正式实验保持数据划分、tokenizer、独立参数、初始化规则、训练 token、优化器与精度一致，记录重复计算的成本。这回答“给同样的模型多算几遍有什么变化”。第二组实验匹配总训练 FLOPs，允许调整步数并如实记录减少的训练 token；这回答“同样预算如何分配”。两组问题不同，应分别报告。

还应固定序列长度、batch、设备、缓存策略、残差与归一化位置，以及是否对中间重复提供额外监督。本仓库命令只使用最终语言模型损失，没有引入每次重复的额外监督。训练时的重复次数与评估时改变重复次数也属于不同实验，不能混在一张没有说明的结果表中。

质量指标至少包括独立验证集交叉熵和任务分数；资源指标包括独立参数量、训练 token、FLOPs、编译时间、预热后的吞吐、峰值显存与缓存大小。四步样本训练只能暴露执行错误，无法比较泛化。重复次数更多而 loss 更低，也必须连同增加的计算成本报告。

| 待判定问题 | 当前证据 | 决策 |
|---|---|---|
| <code>loop_mode</code> 是否真正共享参数？ | 同一参数对象被重复调用；代码可定位 | 已实现，执行仍需保存报告 |
| 是否复现指定论文的循环结构与效果？ | 当前是局部按层重复，缺完整协议 | 不宣称复现 |
| 增加重复是否改善固定预算下的质量？ | 尚无本协议的训练与验证结果 | 未判定 |
| TTT 与 Loop 是否互补？ | 可组合组件存在，四组对照未完成 | 计划实验 |

## 失败标准、已知缺口与下一次运行

若训练出现非有限值、不能产生完整检查点，或者流式输出与对应整段执行不一致，应先判为实现或配置失败。若在完整预算对照中没有稳定的质量提升，则该设定没有支持“重复更划算”的假设；不能通过只挑更长训练或更幸运 seed 的结果挽救结论。

目前还没有本文协议下的多 seed 质量报告、准确的跨配置 FLOPs 对齐，也没有整个多层 stack 的循环实现与论文复现记录。下一次运行先保存三组 tiny 原始产物，确认共享模式的参数数量与执行顺序，再选择真实训练与留出数据。TTT×Loop 四组比较在这之后开展，避免把两个尚未解释的变化叠在一起。
