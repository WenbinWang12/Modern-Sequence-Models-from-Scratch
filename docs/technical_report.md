# TTT-E2E Technical Report

**Working draft — 2026-11-17**

本文档是 TTT-E2E 从数据、架构、训练到评测的技术报告草稿。采用“证据先于结论”的写法：

- **measured**：由保存的原始报告或远端命令直接测得；
- **inferred**：由配置或公式推导，尚未由完整实验确认；
- **not measured**：当前没有证据，不能写成模型能力结论。

当前公开结果是 CPU pilot，不能代表目标 GPU 规模，也不能声称击败任何开源或闭源模型。

## 摘要

我们实现了一个端到端 Test-Time Training（TTT-E2E）语言模型。模型把跨样本共享的慢参数 theta 与会话内更新的快权重 phi 分开：训练阶段用外循环优化 theta 和初始化，使模型学会如何从上下文适应；推理阶段只更新当前会话的 phi，不修改发布的慢参数。

本报告的第一条完整证据链是一个约 1.020M 参数的 CPU pilot：使用 WikiText-2 train split 的 5,120,000 个调度 token，完成 5,000 步训练、checkpoint、恢复、独立 validation/test NLL、profile 和功能验证。该实验验证工程闭环，不是 compute-optimal 预训练，也不是与大型模型的公平能力比较。

## 1. 研究问题与贡献边界

### 1.1 研究问题

我们要回答三个可区分的问题：

1. 在固定参数、数据、有效 token 和计算预算下，TTT 快权重是否降低独立 held-out NLL？
2. 在长上下文中，TTT 的会话状态是否比完整 KV 历史提供更好的内存/延迟折中？
3. 外循环学到的初始化是否使推理时适应稳定、可恢复且具有可重复收益？

“在某个 benchmark 上得分更高”只有在数据、tokenizer、prompt、适应开关、预算和统计协议都固定后，才可以归因于 TTT。

### 1.2 贡献声明模板

正式版本只允许从以下证据中写贡献：

| 声明 | 必须有的证据 | 当前状态 |
|---|---|---|
| TTT-E2E 训练闭环可运行 | 有限梯度、因果性、streaming 一致性、checkpoint 恢复 | **measured** |
| TTT 在固定预算下降低 NLL | plain/TTT 配对模型、多 seed、独立 test | **not measured** |
| TTT 提高长上下文能力 | 预注册的 RULER/LongBench/检索协议、adapt/no-adapt 对照 | **not measured** |
| TTT 降低推理成本 | 相同硬件、精度、上下文和 warm-up 后 latency/显存 | **not measured** |
| TTT 击败某个开源或闭源模型 | 同等协议和可复核原始结果 | **not measured** |

## 2. 模型规格

### 2.1 参数与状态

对第 c 个上下文块，慢参数记为 theta，快权重记为 phi_c。快权重不会跨独立 session 共享：

~~~text
phi_c,0 = g_theta(previous_state)
phi_c,i+1 = phi_c,i - eta * grad_phi loss_inner(x_c,i, y_c,i; theta, phi_c,i)
~~~

模型前向使用当前块开始时的 phi 预测，等目标已经观测后才更新到下一状态。这一时序是因果性的一部分，不能把“更新后权重”用于预测同一个尚未观测的目标。

### 2.2 当前 CPU pilot 配置（measured）

| 字段 | 值 |
|---|---:|
| 实际参数量 | 1,020,032 |
| 层数 | 6 |
| hidden size | 128 |
| intermediate size | 256 |
| attention heads / KV heads | 4 / 2 |
| layer types | full / sliding attention 混合 |
| context upper bound | 512 |
| training sequence length | 128 |
| tokenizer | byte，V=259 |
| parameter dtype / compute dtype | float32 / float32 |
| tie word embeddings | true |
| fast MLP initial state | 98,304 parameters |

完整配置见 [configs/runs/cpu_pilot_1m.json](../configs/runs/cpu_pilot_1m.json)。模型名中的规模档位不能替代实际参数计数；正式报告必须报告 embedding、slow weights、fast weights 和总参数。

## 3. TTT 训练算法

### 3.1 内循环目标

训练时，每个序列被当作一个会在推理阶段到来的上下文。内循环使用已观测的 next-token 对：

~~~text
loss_inner = sum_t mask_t * CE(model(x up to t; theta, phi_t), y_t) / sum_t mask_t
~~~

其中 mask_t 是监督 mask。文档边界、EOS、packing 和 chunk 顺序必须写入数据协议，不能由模型实现隐式决定。

### 3.2 外循环目标

外循环在内循环适应之后评价 next-token loss，并对慢参数、初始化和可学习规则求导。因为适应后的 phi 依赖 theta，外循环梯度一般需要穿过内循环。当前实现保留高阶梯度；正式模型必须报告是否截断、内循环步数、梯度裁剪和实际计算成本。

### 3.3 训练和推理的差别

| 阶段 | 更新对象 | 是否保留外循环梯度 | 状态范围 |
|---|---|---|---|
| 预训练 | theta，并在样本内模拟 phi | 是 | batch 内临时状态 |
| 推理 | 当前 session 的 phi | 否 | session 独立 |
| checkpoint 恢复 | theta、optimizer、sampler | 不适用 | 训练作业 |
| session 恢复 | phi、KV/Delta、未完成 chunk | 不适用 | 单一会话 |

只在推理时对现成模型做 SGD，不等于 TTT-E2E；那是没有外循环元学习的 post-hoc TTT 变体。

## 4. 数据与 tokenizer

### 4.1 数据对象

当前 pilot 使用公开 WikiText-2 train/validation/test 文件的公开镜像。下载、逐行 JSONL 转换、token manifest 和分片摘要均单独保存。

| split | 文档数 | token 数 | 用途 |
|---|---:|---:|---|
| train | 23,767 | 10,795,013 | 参数更新 |
| validation | 2,461 | 1,121,544 | 模型选择/诊断 |
| test | 2,891 | 1,256,406 | 最终一次性报告 |

数据卡和许可信息必须与文件 SHA256 一起归档。当前 pilot 的数据只适合管线与小模型验证；它不支持对大规模预训练能力的外推。

### 4.2 必须报告的处理信息

正式数据章节必须给出：

- 原始 URL、不可变 revision、许可和获取时间；
- 原始文件、清洗文件、JSONL、token 分片的字节数和 SHA256；
- 空行、非法 UTF-8、重复文档、过滤规则和保留率；
- 精确/近似去重方法及阈值；
- train/validation/test 的文档簇切分和污染检查；
- tokenizer 文件、revision、词表、特殊 token、归一化和 added tokens；
- packing、EOS、BOS、mask 和跨文档状态规则。

当前 pilot 使用 byte tokenizer，因此其 PPL 不能直接与 BPE 或 SentencePiece 模型比较。正式主实验必须增加固定 revision 的子词 tokenizer，并将 tokenizer 训练语料与预训练语料的关系写清楚。

## 5. 训练设置与系统

### 5.1 当前 pilot（measured）

| 项目 | 值 |
|---|---|
| 设备 | 双路 AMD EPYC 7742 主机上的单个 JAX CPU device |
| 可见内存 | 约 503 GiB |
| GPU | 无 |
| optimizer | AdamW，b1=0.9，b2=0.95，weight decay 0.1 |
| gradient clipping | global norm 1.0 |
| scheduler | warmup + cosine decay |
| batch / sequence | 8 / 128 |
| steps / scheduled tokens | 5,000 / 5,120,000 |
| precision | float32 |
| checkpoint | NPZ，每 500 步 |

Muon 不是当前结果的一部分。Muon 实验应单独作为 optimizer ablation，不能和 AdamW 结果混在同一主表中。

### 5.2 正式训练还要记录

正式 GPU run 必须保存 GPU 型号、数量、拓扑、驱动、JAX/XLA 版本、global batch、gradient accumulation、有效 token、FLOPs、wall-clock、吞吐、峰值显存、编译时间、重启次数、loss spike 和 checkpoint 恢复点。只写“用了八张卡”不足以复现成本。

## 6. 当前结果

### 6.1 训练曲线（measured）

CPU pilot 从早期 loss 约 4.65 降到最终 1.3237。最后 50 个记录点的线性趋势仍为负，但最后 10 个点的波动标准差约 0.049；因此应写成“基本进入平台但尚无完全收敛证据”，不能写成“已经收敛”。

训练最后一个记录点：

| step | tokens seen | loss/NLL | tokens/s |
|---:|---:|---:|---:|
| 5,000 | 5,120,000 | 1.3237 | 1,103.6 |

### 6.2 独立语言建模评测（measured）

评测从未用于训练或选 checkpoint，使用 256 个窗口、约 32.7k 个有效 token：

| split | adapt | NLL | PPL |
|---|---|---:|---:|
| validation | no | 1.2814 | 3.6016 |
| test | no | 1.2912 | 3.6370 |
| validation | yes | 1.2715 | 3.5661 |
| test | yes | 1.2779 | 3.5892 |

这里的 adapt=yes 是 TTT 会话内适应，不能和 adapt=no 混成同一个标准语言模型分数。由于 tokenizer 是 byte，PPL 只作本项目内部对照。

### 6.3 合成检索诊断（measured, not a public benchmark）

仓库自带的 256 条 secret-ID probe 在当前 pilot 上 exact match 为 0。该结果说明当前小模型和训练预算没有通过这项合成长上下文诊断；它不能解释为标准 benchmark 失败，也不能被替换成只展示成功样本的数字。正式长上下文结论需要 RULER、LongBench 或预注册的独立任务集。

### 6.4 推理系统指标（measured）

在 CPU、context length 128、decode 8 的 profile 中：

- prefill：约 0.111 s；
- decode：约 191.5 token/s；
- fast-weight state：393,216 bytes；
- KV cache：594,432 bytes；
- chunk cache：132,096 bytes。

这些是单进程 CPU 数字，不是 GPU 服务吞吐，也不代表大上下文或并发性能。

## 7. Scaling 与消融计划

每个主结论至少需要以下配对：

| 变量 | 对照 | 固定项 |
|---|---|---|
| TTT adaptation | plain/fast update off | 参数、数据、token、optimizer、seed |
| fast MLP | 原 FFN | 总参数或 FLOPs 预算 |
| inner steps | 0/1/2/4 | outer budget、chunk、数据 |
| optimizer | AdamW/Muon | 参数、batch、scheduler、seed |
| context | 128/512/4K/32K | tokenizer、模型、有效 token |
| tokenizer | byte/subword | 原始文本、模型预算、评测单位 |

正式 scaling 至少覆盖 10–30M pilot、125M 主模型和在资源允许时的 760M 扩展。每个规模报告实际参数、激活参数、有效 token、FLOPs、吞吐和 held-out 曲线。

### Muon 试验要求

Muon 只应作用于明确选择的二维矩阵参数；embedding、norm、bias、标量和快权重需要单独的 AdamW 或保持不更新的规则。必须报告：

1. Muon 的具体实现、Newton-Schulz/正交化步数和动量规则；
2. 参数分组、学习率、weight decay、梯度裁剪和 warmup；
3. 是否支持 TTT 外循环的高阶导；
4. 相同 token/FLOPs 下与 AdamW 的多 seed 结果；
5. loss spike、恢复和最终 test 的置信区间。

如果 Muon 的实现不能稳定穿过当前内循环高阶梯度，应该把它作为独立系统实验，而不是修改主结果的 optimizer。

## 8. Benchmark 协议

### 8.1 语言模型与理解

正式表格可包含 Pile/WikiText held-out、MMLU、C-Eval、CMMLU、HellaSwag、PIQA、ARC、OpenBookQA、BBH、DROP 和 NQ。每项都要写：版本、split、prompt、shot 数、答案归一化、有效样本、超时/截断率、重复次数和置信区间。

### 8.2 数学、代码和长上下文

适用时加入 GSM8K、MATH、HumanEval、MBPP、LiveCodeBench、LongBench 和 RULER。TTT 特有的 adapt/no-adapt 必须成对报告，并记录适应使用哪些 token、是否允许看到目标、更新步数和额外延迟。

### 8.3 污染与公平比较

训练原文、清洗文档、tokenizer 训练文本和 benchmark prompt/answer 要做 exact、NFC、长子串和近似匹配。未完成的检查写 not_checked，不能写“无污染”。比较模型时固定 tokenizer/revision、有效 token、参数量或 FLOPs、context、精度、prompt 和 seed；闭源模型只能作为外部参考，不能在不等价协议下声称击败。

## 9. 安全、治理与限制

报告必须说明数据许可、PII、隐私、毒性、偏见、越狱、危险能力、拒答行为和已知失败。还要给出模型权重、代码、数据清单、第三方许可证和使用限制。安全评估不应只在训练后挑选有利样本；测试集、提示模板和失败案例都要保留原始记录。

当前 CPU pilot 尚未完成正式安全评测，标记为 not_measured。

## 10. 复现与发布清单

一次可发布的 run 至少包含：

- Git commit；若使用 tarball，另存 upstream commit 和 archive SHA256；
- 完整 config、数据 manifest、tokenizer 摘要、环境报告和命令行；
- raw JSONL metrics、图表生成脚本和原始评测 JSON；
- checkpoint、optimizer、sampler、恢复点和失败日志；
- 参数计数、有效 token、FLOPs、wall-clock、吞吐、显存和编译时间；
- benchmark 版本、prompt、答案、评分器、样本数和置信区间；
- 数据许可、污染报告、安全评测和模型卡。

当前 CPU pilot 的远端产物位于 <code>outputs/cpu-pilot-1m-formal/</code>；正式发布前还需把验证 JSON、数据 manifest SHA、Muon 对照、多 seed、标准 benchmark 和安全报告纳入同一版本化结果目录。

## 11. 参考报告

本报告的组织方式参考公开技术报告中反复出现的证据类别：模型与训练细节、数据和预算、系统成本、消融、公开 benchmark、风险与复现材料。原始参考包括：

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)
- [GPT-4 Technical Report](https://cdn.openai.com/papers/gpt-4.pdf)
- [Claude 4 System Card](https://www-cdn.anthropic.com/6d8a8055020700718b0c49369f60816ba2a7c285/Claude%204%20System%20Card.pdf)
- [End-to-End Test-Time Training for Long Context](https://arxiv.org/abs/2512.23675)

StepFun、GLM 等报告也遵循类似的模型规格、数据/训练预算、系统效率、benchmark 和安全披露结构；在正式 TTT 报告中应加入对应版本的原始链接和可复核表格，而不是只引用二手榜单。

## 12. 与成熟技术报告的对齐矩阵

“技术报告”不是把论文摘要扩写，而是给出一条从输入、训练、系统到结论的证据链。公开报告的披露重点并不完全相同：DeepSeek-V3 报告同时给出了总参数、激活参数、训练 token、硬件小时、架构、基础设施、训练稳定性和大规模评测；Qwen2.5 报告把预训练规模、SFT、强化学习、模型家族和多类 benchmark 放在同一条产品证据链中；GPT-4 报告明确说明哪些架构、数据和训练细节没有披露；Claude 系统卡则把能力、安全、红队、代理风险和发布决定作为主要内容；TTT-E2E 报告重点给出内外循环、匹配参数/训练 token、长上下文 loss 和延迟曲线。

| 报告部分 | 大型模型报告中的常见证据 | TTT 报告必须给出的对应证据 | 当前状态 |
|---|---|---|---|
| 模型身份 | 版本、规模、激活参数、模型家族 | TTT 版本、slow/fast 参数、配置 hash、checkpoint hash | 部分 measured |
| 训练数据 | 来源、规模、混合、过滤、token budget | 每个 split 的 manifest、tokenizer、许可和污染报告 | 部分 measured |
| 训练系统 | GPU 型号、拓扑、精度、吞吐、稳定性 | inner/outer 计算、设备、通信、compile、loss spike、恢复 | CPU measured |
| 训练方法 | objective、optimizer、schedule、阶段 | inner objective、outer objective、chunk 时序、二阶导、Muon/AdamW 分组 | AdamW measured |
| Scaling | 不同规模/数据/compute 的曲线 | 参数、激活参数、有效 token、FLOPs、context、适应步数 | not measured |
| Benchmark | 标准任务、prompt、评分器、CI | PPL、理解/数学/代码/长上下文，adapt/no-adapt 成对结果 | 小型 PPL measured |
| 安全与治理 | system card、红队、发布等级 | TTT 适应风险、数据许可、PII、越狱、agentic failure | not measured |
| 可复现 | 代码、权重、配置、环境、日志 | tarball/commit、manifest、metrics、checkpoint、原始评测 JSON | 部分 measured |

### 12.1 透明披露也是结果

有的技术报告由于竞争或安全原因不披露模型规模、数据构造和硬件细节。这种“未披露”本身应被记录，而不是由读者用猜测补齐。本项目选择更严格的研究路线：无法公开的字段写 <code>not_disclosed</code>，尚未执行的检查写 <code>not_measured</code>，估算值写 <code>inferred</code>，实测值链接到原始 JSON。报告不会把估算写成事实。

### 12.2 TTT 特有的最低披露集

普通语言模型报告可以只写一次训练 loss；TTT 报告还必须回答：

1. fast weights 的初态来自哪里，是否依赖慢参数；
2. 内循环使用哪些目标，目标何时变成已观测信息；
3. 每个 chunk 的预测、更新、cache 写入顺序；
4. 外循环是否穿过 inner updates，是否截断高阶梯度；
5. 推理适应是否修改发布参数，session 是否相互隔离；
6. adapt/no-adapt 的额外 token、梯度、延迟和内存；
7. session checkpoint 能否在不同设备布局恢复；
8. 不适应的 plain baseline 是否拥有相同参数、数据和 FLOPs。

没有这些字段，TTT 的分数无法判断来自模型表示、测试时计算、标签泄漏还是评测协议差异。

## 13. 主实验设计

### 13.1 预注册的主问题

每次正式训练开始前，在 experiment.json 旁保存一份不可变的研究问题：

~~~json
{
  "question": "Does TTT reduce held-out NLL at matched parameter, token, and FLOP budgets?",
  "primary_metric": "test_nll_no_contamination",
  "secondary_metrics": [
    "test_ppl",
    "adapt_minus_no_adapt_nll",
    "prefill_latency",
    "decode_tokens_per_second",
    "fast_state_bytes"
  ],
  "primary_comparison": "plain_vs_ttt",
  "budget": {
    "parameters": "matched",
    "effective_tokens": "matched",
    "training_flops": "matched"
  },
  "seeds": [0, 1, 2]
}
~~~

研究问题一旦确定，不能在看完 test 结果后替换成更有利的问题。任何追加分析都要标记为 secondary 或 exploratory。

### 13.2 三组必要对照

正式主表至少包含：

1. **Plain baseline**：没有 fast-weight 更新的同架构模型；
2. **TTT no-adapt**：保留 TTT 参数但推理时关闭适应；
3. **TTT adapt**：执行预先声明的会话内更新。

这三组分别回答：结构本身是否有效、训练是否改变了慢参数表示、推理适应是否贡献额外收益。只比较 TTT adapt 与普通小模型，会把参数结构、训练策略和推理时额外计算混在一起。

### 13.3 成本匹配

记录每次运行的有效 token 数：

~~~text
effective_tokens = sum(batch_size * sequence_length * loss_mask)
~~~

还要记录内循环更新的次数、外循环反向的 FLOPs、编译时间和 wall-clock。TTT 的每个 token 可能触发额外的梯度计算，因此“训练步数相同”不等于“计算预算相同”。如果采用 matched-FLOPs 对照，应允许步数不同，但必须报告因此改变的 token 数。

## 14. 图表和表格设计

一份可读的技术报告不应把所有数字塞进一张 benchmark 表。建议固定以下图表：

| 图/表 | 横轴 | 纵轴或内容 | 目的 |
|---|---|---|---|
| Figure 1 | context length | held-out loss | 长上下文质量 |
| Figure 2 | context length | prefill/decode latency | 计算成本 |
| Figure 3 | training tokens | train/validation NLL | 是否过拟合、是否收敛 |
| Figure 4 | parameters/tokens | loss | scaling law |
| Figure 5 | inner steps | NLL/latency | 适应收益和成本 |
| Figure 6 | seed | test NLL/PPL | 方差和置信区间 |
| Table 1 | model variants | 参数、激活参数、context | 模型规格 |
| Table 2 | data splits | 文档、token、hash、license | 数据身份 |
| Table 3 | benchmark | score、N、CI、prompt version | 可比性 |
| Table 4 | systems | FLOPs、吞吐、显存、compile | 成本 |
| Table 5 | failure cases | 输入、预期、实际、根因 | 限制与诚实披露 |

图表脚本必须读取原始 JSONL，而不是手工复制最终数字。每个图旁边保存输入文件 hash、脚本 commit 和生成时间；如果图表重绘，旧图不能被静默覆盖。

## 15. 失败分析与反例

成熟报告不会只展示成功案例。TTT 报告至少需要保留：

- loss spike、NaN、梯度爆炸和恢复前后的 checkpoint；
- fast weights 发散、更新范数饱和或适应后 NLL 变差的样本；
- 长上下文中检索正确但生成格式错误的样本；
- 适应后优于 no-adapt 但在新域退化的样本；
- tokenizer 不匹配、mask 全零、文档边界错误和污染检测失败；
- Muon 与 AdamW 在高阶梯度、速度、稳定性上的失败对照。

每条失败记录包含输入版本、随机种子、命令、错误类型、是否重试、修复提交和是否进入最终统计。删除失败运行会破坏训练稳定性结论。

## 16. 发布门槛

TTT 模型达到“可发表/可发布”之前，必须同时满足：

### A. 数值正确性

- 反向梯度有限；
- 因果性和 future-token 修改测试通过；
- full sequence 与 prefill/decode 在容忍范围内一致；
- fast state、KV、pending chunk 的 checkpoint 恢复一致。

### B. 训练证据

- 至少一个 plain baseline 和一个 TTT variant；
- 固定 tokenizer、数据、有效 token 和 FLOPs；
- 至少三个 seed 或有预先声明的单 seed 限制；
- 独立 validation/test，不用 test 选超参；
- loss、吞吐、显存、compile、checkpoint 和失败日志齐全。

### C. 能力证据

- held-out NLL/PPL；
- 至少一个公开理解/数学/代码 benchmark；
- 至少一个长上下文 benchmark；
- adapt/no-adapt 和失败率；
- 污染检查和评分器版本。

### D. 治理证据

- 数据许可证和 PII 处理；
- 安全/偏见/越狱/危险能力评估；
- 模型卡、限制和可接受使用范围；
- 权重、代码、配置和复现说明。

当前 CPU pilot 只满足 A 的大部分、B 的工程子集和 C 的 held-out PPL 子集，因此应被称为 **engineering and algorithm pilot**，而不是完整发布模型。

## 17. 版本记录

| 版本 | 日期 | 变化 | 证据状态 |
|---|---|---|---|
| 0.1 | 2026-11-17 | CPU pilot、PPL、profile、技术报告草稿 | measured / not measured 混合，不能作能力发布 |
| 0.2 | 待定 | AdamW/Muon 配对、plain baseline、多 seed | planned |
| 1.0 | 待定 | GPU 规模、完整 benchmark、安全和复现包 | release gate |

## 18. H200 15M scaling（measured）

在 H200 GPU 5 和 GPU 7 上分别运行独立的 TTT 与 plain baseline，未使用分布式通信。两组实际参数量均为 15,449,856，训练 10,000 步、40,960,000 个 token，使用相同的 WikiText-2 train split、byte tokenizer、seed 和 sequence length 512。

| 组别 | GPU | train tokens/s | 峰值显存 | validation PPL | test PPL |
|---|---:|---:|---:|---:|---:|
| Plain baseline | 7 | 约 57.2k | 0.84 GiB | 2.7279 | 2.6865 |
| TTT no-adapt | 5 | 约 19.5k | 1.87 GiB | 2.7522 | 2.7082 |
| TTT adapt | 5 | 推理适应额外计算 | 同上 | 2.6642 | 2.6109 |

这组结果有两个同时成立的结论：TTT 的 no-adapt checkpoint 略差于 plain baseline；启用会话内适应后，test PPL 比 plain baseline 低约 0.0755。代价是训练吞吐约低 2.9 倍、状态和显存更大。由于这是单 seed、单数据集、byte tokenizer 的配对 pilot，不能写成普适架构优势；下一步需要 Muon/AdamW、多 seed、子词 tokenizer、更大数据和标准 benchmark。

### 18.1 公开数据混合与 benchmark 诊断（measured）

为避免把单一 WikiText-2 的结果误认为通用能力，我们又运行了一个固定配比的公开数据实验：WikiText-2 train 窗口 85%，Tiny Shakespeare 窗口 15%，`with_replacement`，seed 0，其他模型配置和 40.96M scheduled tokens 不变。80,000 个窗口的实测来源比例为 85.09375%/14.90625%。

| 模型 | valid PPL | test PPL | MMLU 5 科目诊断 |
|---|---:|---:|---:|
| plain baseline | 2.7555 | 2.6966 | 29.59% (392 valid) |
| TTT no-adapt | 2.7664 | 2.7105 | 26.53% (392 valid) |
| TTT adapt | 2.6765 | 2.6223 | not run |

PPL 是每个 split 128 个随机窗口、约 65k 有效 token；MMLU 是零样本答案字母 continuation log-likelihood，byte tokenizer、最大 512 token、TTT 适配关闭。该 MMLU 结果属于 diagnostic：模型只有约 15M 参数且没有 instruction tuning，且 professional_law 由于上下文上限跳过 95/100 题，不能和公开厂商榜单比较，也不能据此宣称 TTT 的通用能力下降或提升。正式 benchmark release 仍需 tokenizer 冻结、多 seed、污染审计和数学/代码/长上下文任务。
