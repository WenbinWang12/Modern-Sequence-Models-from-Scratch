# TTT-E2E 总体实验规划

当前数据管线已经具备可复现的多领域语料池。下一阶段把实验拆成方法、数据和规模三个轴，依次推进。每一轮只改变一个主要轴，保留固定 tokenizer、数据清单、训练预算和评测协议。

本文把 TPP 定义为训练 token 数除以总可训练参数量。小规模 smoke run 可以低于 TPP 20，正式语言模型比较以 TPP 20–40 作为起点，并另外记录更高 TPP 的过训练实验。

## 总体研究问题

1. 在相同 token 预算下，TTT-E2E 能否稳定超过滑窗注意力？
2. 收益来自测试时更新、额外的 MLP′ 参数，还是训练时的元学习？
3. 收益能否在更长上下文、不同模型规模和不同数据配方上重复？
4. 训练成本、推理吞吐、显存和会话状态大小是否支持实际使用？

已有的 [技术报告](technical_report.md) 已经完成了 FineWeb-Edu 与 PG-19 上的机制验证。新一轮实验使用 [Dolma3 数据池](data_corpus.md) 做独立的数据和规模验证，不把两套数据的结果混成一个结论。

## 实验阶段

### E0　数据与系统准入

这一阶段的目标是确认训练结果可以解释。

- 固定 tokenizer、词表、特殊 token、数据 revision、held-out 清单和随机种子。
- 检查训练集与 held-out、公开 benchmark 之间的重复，确认跨来源去重结果。
- 检查分片清单、SHA256、跨分片采样、断点恢复和流式读取一致性。
- 记录首步编译时间、稳定步耗时、tokens/s、峰值显存和数据读取耗时。

准入条件是 loss 能下降、恢复后下一步数值一致、验证集合无污染、数据加载不成为主要瓶颈。当前数据构建和定向测试已经覆盖了大部分条件。

### E1　Dolma3 上的机制主实验

第一轮使用约 125–140M 参数、4,096 上下文、窗口 512、更新块 256 的模型，训练约 3B tokens。这对应总参数量约 22–24 的 TPP，适合正式比较。五个模型共享初始化种子、数据顺序、优化器、学习率计划和检查点位置。

| 实验臂 | 作用 |
|---|---|
| Full attention | 质量参考，完整可见上下文 |
| Sliding-window attention | 常数窗口的基础对照 |
| TTT no-update | 区分额外 MLP′ 参数的作用 |
| TTT-E2E | 主要方法，训练和评测都执行测试时更新 |
| Loop+TTT | 共享循环深度与 TTT 快权重的融合模型 |

这一轮同时报告 fixed-token 结果和 fixed-wall-clock 结果。TTT 的训练成本更高，两种约束回答的是不同问题，分别报告。

### E1-L　Loop 和 Loop+TTT 的定义

Ouro 的核心是共享 Transformer stack 的多次 latent-space recurrence，公开基础模型使用 4 个 recurrent steps。Nanbeige4.2-3B 使用 22 层 decoder stack 的两次 full-stack recurrence。两者都把循环当作预训练架构，而不是推理时临时重复普通层。[Ouro paper](https://arxiv.org/abs/2510.25741) · [Nanbeige4.2 model card](https://huggingface.co/Nanbeige/Nanbeige4.2-3B-Base)

本项目分两步实现：

1. **Prefix Loop+TTT** 作为第一版。使用当前代码已有的 `middle_block` 循环，固定 `loop_repeats=2`，在前段共享 block 参数，最后 3 层保留 TTT fast MLP。每个 chunk 的 TTT 更新只在最后一个 loop pass 结束后执行一次，下一 chunk 使用更新后的快权重。
2. **Full-stack Loop+TTT** 作为第二版。整个 decoder stack 共享并重复两次，TTT fast MLP 挂在共享 stack 的最后若干层。一个 chunk 内所有 loop pass 使用同一组快权重，chunk loss 计算完成后只更新一次。这样保持“先预测、后更新”的因果关系，也避免循环次数把 TTT 更新次数重复放大。

第一版不加入动态 early-exit、entropy regularization、LoopSplit 或 mHC。固定深度结果稳定后，再单独研究 per-loop loss、退出门控和循环次数外推。

### E2　机制拆解和超参数

只在 E1 选出的 TTT 配置上进行，每个因素单独改变。优先级如下：

1. 测试时更新开关和未经元学习的 naive dynamic evaluation。
2. 一阶近似与完整二阶外循环。
3. 更新块长度 C 取 128 和 256。
4. 内循环学习率和梯度裁剪。
5. 后段 MLP′ 层数、注意力窗口和多 token 预测头。

机制消融可以使用 0.2–0.5B tokens，但只能作为实现和超参数诊断，不能作为正式质量结论。正式消融至少使用与 E1 同级别的 TPP，或明确标注为低 TPP 诊断。

### E3　数据配方选择

固定模型结构、训练预算和随机种子，只比较数据配方。首轮候选如下：

- `corpus_dolma3`，当前 174.48B GPT-2 token 多领域池。
- `corpus_v2`，现有替代多领域池。
- web-only 或 FineWeb/PG-19 对照，用于判断多领域数据带来的变化。
- 需要时加入提高数学或代码权重的候选配方。

每个配方都报告总体 held-out loss 和按来源拆分的 web、science、code、math、reference loss，同时记录 sampler 实际比例、重复率和有效覆盖量。数据配方只在代理模型上筛选，确定后再进入大规模训练。

### E4　上下文与记忆能力

固定已选模型和数据，只改变评测长度。至少覆盖 4K、8K、16K 和 32K，上限由显存和服务吞吐决定。评测包括：

- PG-19 按书统计的 loss 曲线。
- NIAH 或等价单针检索。
- 逐字回忆和跨段 recall。
- 关闭更新、只更新一次和正常分块更新的曲线。

需要同时记录快权重状态、KV cache、未完成更新块和每 token 推理成本，验证状态大小是否与上下文长度分离。

### E5　随机种子和模型规模

在 E1 的主结论出现后，再使用 3 个随机种子重复 Full、Sliding-window、TTT-E2E 和 Loop+TTT。随后按下表扩展模型规模。训练 token 是每个模型、每个实验臂的预算。

| 模型规模 | TPP 20 预算 | 推荐第一轮预算 | 174.48B 池的理论上限 |
|---:|---:|---:|---:|
| 125M | 2.5B | 3–5B | 约 1,395 TPP |
| 350M | 7B | 14B | 约 498 TPP |
| 760M | 15.2B | 30B | 约 229 TPP |
| 1.3B | 26B | 50–60B | 约 134 TPP |
| 3B | 60B | 100–120B | 约 58 TPP |

规模实验分别报告固定 token 和固定 FLOPs 结果。小规模、短预算的结果不能外推到 3B 或更大规模。

Loop 共享参数会让按参数计算的 TPP 显得偏高，因此每个 Loop 结果还要记录有效层访问次数、实测训练 FLOPs/token、wall-clock 和 unique trainable parameters。Loop 与非 Loop 模型分别做参数匹配和 FLOPs 匹配。

### E6　最终规模训练

只有在方法、配方和成本都通过前面阶段后，才进入 30B、60B 或更大 token 预算。174.48B 数据池可以支持 1.3B 级别的高 TPP 训练或 3B 级别的 TPP 50–60 训练。最终训练需要冻结：

- 数据 revision、来源比例和重复策略
- 模型配置、上下文长度和 TTT 更新规则
- 训练 token、FLOPs、学习率和 checkpoint 策略
- 独立 held-out、长上下文和 benchmark 清单

## 阶段准入门槛

| 门槛 | 进入条件 |
|---|---|
| G0 数据与系统 | 数据 digest、污染审计、恢复一致性和稳定吞吐全部通过 |
| G1 机制 | TTT-E2E 相对滑窗基线在 held-out 或长上下文上有稳定的配对收益，且无更新对照和 naive dynamic evaluation 不能解释主要收益 |
| G2 稳健性 | 3 个种子的方向一致，配对置信区间支持主结论 |
| G3 规模 | 模型规模或训练 token 增大后，主结论方向保持，并记录实际成本 |
| G4 经济性 | 同时给出 fixed-token 和 fixed-wall-clock 曲线，训练与推理开销完整可追溯 |

## 评测指标

### 主要指标

- 各来源 held-out 的 token NLL 和 bits-per-byte
- loss 随上下文长度的曲线
- TTT 相对 Sliding-window 和 Full attention 的配对差值

### 能力指标

使用现有评测入口覆盖 ARC、HellaSwag、PIQA、SciQ、MMLU、GSM8K、HumanEval 和 BBH。小模型结果主要用于相对比较，不能当作通用能力结论。

### 系统指标

- 训练 tokens/s、step time、编译时间和峰值显存
- 每个模型的实际 FLOPs 和 GPU-hours
- 预填充、更新和生成阶段的吞吐
- KV cache、快权重和会话状态大小

## 统计与记录规则

- 模型之间使用同一批评测窗口，按窗口或书做配对 bootstrap。
- 最终主结论至少重复 3 个种子，单次运行不能作为稳定性证据。
- 每个实验保存配置、数据清单摘要、随机种子、训练日志、checkpoint 指纹和原始评测输出。
- 公开模型只作为绝对水平参考，不作为 token 预算不匹配的公平对照。
- 报告必须区分训练 token、数据池容量、实际重复采样量和有效文档覆盖量。

## 推荐执行顺序

```text
E0 数据与系统准入
  → E1 Dolma3 四臂机制主实验
  → E2 TTT 机制拆解
  → E3 数据配方代理筛选
  → E4 长上下文评测
  → E5 多种子与规模
  → E6 最终规模训练
```

第一步应生成 E1 的五个 Dolma3 配置并运行约 3B token 的配对实验。Loop+TTT 先使用 Prefix Loop+TTT 版本，Full-stack Loop+TTT 在循环状态和 KV cache 通过独立正确性测试后再加入。若 TTT-E2E 或 Loop+TTT 在 held-out 和长上下文曲线上的优势不能重复，先停在 E2 诊断，不进入几十 B token 级训练。
