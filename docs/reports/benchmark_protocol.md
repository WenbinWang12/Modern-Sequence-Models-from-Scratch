# Benchmark 测评协议与当前进度

!!! note "版本说明"
    本页同时保留 2026-11-17 的 15M 流程验证记录和 1.0 主配方的 H200 结果。两者的模型规模、tokenizer、数据和上下文长度不同，必须按小节中的协议分别解读。正式实验和结论见 [TTT-E2E 技术报告](../technical_report.md)。

技术报告中的 benchmark 不是一张可以随意挑选的榜单。每个分数都必须绑定模型检查点、数据版本、tokenizer、提示模板、shot 数、上下文上限、适配开关和有效样本数。本页把公开报告中常见的测评按能力分层，并记录本项目已经实际运行的协议。

## 0. 与 1.0 主报告的关系

仓库最近的实验流水线已经切换到技术报告 1.0 的主配方。主配方使用 GPT-2 BPE 词表（\(V=50{,}257\)）、FineWeb-Edu 与 PG-19 的 50/50 混合、4,096 token 上下文和约 1.3 亿参数的四组匹配模型。原始结果入口是 [`docs/assets/report/results.json`](../assets/report/results.json)，完整方法、数据版本和复现实验命令见 [TTT-E2E 技术报告](../technical_report.md)。

本页下面保留的 15M、byte-token 结果是更早的流程验证记录，不能替代 1.0 主实验。1.0 主实验的训练曲线、消融、NIAH、扩展 benchmark 原始 JSON 和资源失败记录均已写入 `docs/assets/report/results.json`。扩展 benchmark 的有效样本数、跳过数和协议版本以技术报告表 9a 为准。

## 1. 分层测评矩阵

| 层级 | 任务 | 主要回答的问题 | 当前状态 |
| --- | --- | --- | --- |
| 训练健康度 | held-out NLL、PPL、数据混合比例、重复率 | 模型是否学会了训练分布，训练是否稳定 | 已运行 |
| 基础知识与常识 | MMLU、ARC-E/ARC-C、OpenBookQA、PIQA、HellaSwag、WinoGrande | 小模型是否形成可迁移的知识和常识判断 | 1.0 已完成 ARC-Challenge、PIQA。MMLU、ARC-E、OpenBookQA 和 WinoGrande 尚未纳入主表 |
| 数学推理 | GSM8K、MATH、AIME/AMC | 能否进行多步符号推理 | 1.0 已完成完整 GSM8K。MATH、AIME/AMC 尚未运行 |
| 代码 | HumanEval、MBPP、LiveCodeBench | 代码生成、执行正确性和时间切分泛化 | 1.0 已完成完整 HumanEval。MBPP、LiveCodeBench 尚未运行 |
| 难知识与推理 | BBH、GPQA、TruthfulQA | 长链推理、专业知识和事实性 | 1.0 已完成 GPQA 与 `sports_understanding` BBH 诊断。BBH 全套和 TruthfulQA 尚未运行 |
| 长上下文 | RULER、LongBench、DROP | 随上下文长度增长时的检索和组合能力 | 1.0 已完成 NIAH 与 256–8192 synthetic preflight。正式 RULER、LongBench、DROP 尚未运行 |
| 中文能力 | CMMLU、C-Eval | 中文知识和考试型推理 | 双语 tokenizer 冻结后运行 |

可复现的扩展 runner 是 `scripts/run_extended_benchmarks.sh`，为指定的最终检查点运行 ARC-Challenge、PIQA、GPQA、GSM8K、HumanEval、`sports_understanding` BBH 诊断和合成键值长上下文前置检查。合成行用于验证 RULER/LongBench 接入，不会被当作正式套件分数。

这些层级不能互相替代。比如 PPL 较低不能推出 GSM8K 较高；MMLU 的零样本答案选择也不能替代代码执行评测。正式论文表格至少要覆盖知识/常识、数学或代码、以及长上下文中的三类，并同时给出 held-out NLL 作为训练分布指标。

## 2. 15M MMLU 诊断协议

仓库中的 `scripts/eval_mmlu_loglikelihood.py` 使用 MMLU test split 的 CSV 文件，对每道题构造；`scripts/eval_hellaswag_loglikelihood.py` 使用 HellaSwag validation 的四个 completion 做同类 continuation 评分。

### MMLU 协议

对每道题构造：

```text
question

A. choice_0
B. choice_1
C. choice_2
D. choice_3
Answer:
```

对四个候选答案分别计算 continuation log-likelihood：

\[
\hat y=\arg\max_{c\in\{A,B,C,D\}}
\log p_\theta(\text{space}+c\mid\text{prompt}).
\]

评测固定使用 byte tokenizer、最大 512 个 token、TTT 会话内更新关闭（`no-adapt`）。超过上下文上限或 CSV 格式不合法的样本计入 `skipped`，不会静默截断。输出 JSON 会保留每个科目的 `correct`、`samples`、`skipped` 和 `accuracy`。

这个协议只用于当前 15M、byte-token、非 instruction-tuned 模型的工程诊断，不能与使用子词 tokenizer、few-shot 提示或指令微调的厂商分数直接比较。正式能力结果需要在 tokenizer、模型规模、训练数据和提示模板冻结后，至少运行 3 个 seed，并报告 Wilson 或 bootstrap 置信区间。

远端 H200 上的运行命令如下（GPU5 和 GPU7 分别运行单卡，不建立跨卡通信）：

```bash
CUDA_VISIBLE_DEVICES=5 JAX_PLATFORMS=cuda PYTHONPATH=. \
  .venv/bin/python scripts/eval_mmlu_loglikelihood.py \
  --checkpoint outputs/h200-15m-ttt-mixture-85-15/step_00010000.orbax \
  --data-dir /home/wang.21020/mmlu-data/mmlu \
  --subjects abstract_algebra high_school_mathematics college_physics computer_security professional_law \
  --limit 100 --output outputs/benchmarks/mmlu_ttt_mixture_5x100.json
```

baseline 只替换 checkpoint 路径和 `CUDA_VISIBLE_DEVICES=7`。评测结果必须同时保存 checkpoint 的实验配置、数据集压缩摘要、脚本 commit 和下载文件 SHA256；否则只能算临时观察，不能进入论文主表。

在 H200 15M 混合数据配对实验中，五个 MMLU 科目各取最多 100 题：baseline 为 116/392（29.59%），TTT no-adapt 为 104/392（26.53%）。HellaSwag validation 前 100 题中，baseline 和 TTT no-adapt 均为 32/100（32%）。这些数字都是 byte-token、非 instruction-tuned 小模型的诊断结果，不是发布榜单分数。

长上下文目前提供 `scripts/eval_synthetic_long_context.py`，它在 256/512/1024/2048 等长度生成随机 key-value 文档，使用 greedy generation 读取目标值，分别报告 adapt/no-adapt 的 exact match。它是 RULER/LongBench 的工程前置检查，不是公开套件分数；正式运行仍需固定 RULER/LongBench revision、任务集合、上下文长度、答案归一化、污染审计和多 seed。

ARC 与 PIQA 使用 `scripts/eval_arc_piqa_loglikelihood.py`。ARC 输入为官方 JSONL 的 `question.stem`、`choices`、`answerKey` 字段，PIQA 输入为 `goal`、`sol1`、`sol2`、`label` 字段；脚本将答案文本作为 continuation 评分，并记录超过 512 token 的跳过样本。它们与 MMLU 一样先作为小规模诊断，正式结果仍需完整 split、多 seed 和污染审计。

H200 15M 混合数据配对的前 100 题诊断为：ARC-Challenge baseline 12/100、TTT no-adapt 11/100；PIQA baseline 和 TTT no-adapt 都是 57/100。它们均为 byte tokenizer、no-adapt、512-token 上限，不能解释成正式榜单差异。

GPQA 使用公开去门控镜像的 test 数据前 50 题；固定 512-token 上限后 baseline 为 9/32（28.13%），TTT no-adapt 为 5/32（15.63%），两组各跳过 18 题。由于原始 GPQA 对题目再发布和污染有严格要求，仓库只保存结果 JSON，不保存题目文本；这不是官方完整 GPQA leaderboard 分数。

BBH 先运行 `sports_understanding` 子任务前 50 题的零样本 greedy exact match；baseline 和 TTT no-adapt 都是 0/50。BBH 正式协议通常包含子任务专用 answer parser 和 few-shot prompt，本次结果只验证生成链路，不能当作 BBH 总分。

GSM8K 使用 greedy numeric exact match：test 前 100 题 baseline 和 TTT 都是 0/100。HumanEval 使用官方测试执行前 50 题，baseline 和 TTT 的 pass@1 都是 0/50。当前模型是 15M、byte-token、非 instruction-tuned base model，这两个生成任务结果只用于确认评测链路。

RULER/LongBench 的前置 synthetic key-value 诊断在 256、512、1024 请求长度各取 16 题：baseline、TTT no-adapt 和 TTT adapt 的 exact match 全部为 0。它不是公开 RULER/LongBench 分数；正式测评仍需固定套件 revision、任务集合、答案归一化、多 seed 和污染审计。

当前 15M 配置的 `max_position_embeddings` 为 2048，而官方 RULER 的标准 context bands 从 4K 开始；直接截断到 2048 会改变任务定义，所以本轮不冒充官方 RULER 分数。LongBench 同样需要按 task 固定原始输入、答案归一化和长度策略；在 tokenizer 与长上下文配置冻结前不静默截断，暂只保留 synthetic preflight。

## 3. 1.0 主配方扩展 benchmark

`scripts/run_extended_benchmarks.sh` 在 GPU5 和 GPU7 上分别以单卡运行了三个主条件。`main-full` 和 `main-ttt adapt` 使用完整公开 split；`main-ttt no-adapt` 与 `main-ck no-adapt` 使用前 100 题诊断，GPQA 因有效题目数固定为 65。ARC、PIQA、GPQA 是答案文本 continuation 的零样本 log-likelihood，GSM8K 和 HumanEval 是 greedy generation，HumanEval 同时执行官方测试。BBH 只运行 `sports_understanding` 子任务，仍是诊断而非 BBH 总分。

完整主模型和 TTT adapt 的结果已经进入技术报告表 9a。主模型 ARC 1158/1165、PIQA 1800/1819、GPQA 58/256、GSM8K 12/1319、HumanEval 0/164、BBH 0/250；TTT adapt 对应为 1158/1165、980/1819、63/256、10/1319、0/164、0/250。GPQA 的大量跳过来自固定 512-token 上限，不能把有效样本比例误读为题目准确率。

TTT no-adapt 的 100 题诊断为 ARC 24/100、PIQA 55/100、GPQA 17/65、GSM8K 2/100、HumanEval 0/100、BBH 0/100；checkpoint-only 条件对应为 20/100、51/100、15/65、3/100、0/100、0/100。它们的样本量较小，只用于检查配对条件和评测链路，不能和完整 split 的分数直接比较。

合成长上下文行在 256、512、1024、2048、4096、8192 请求长度各取 32 题，四个主条件全部为 0/32。它是 RULER/LongBench 接入的工程前置检查，不是公开套件分数。全注意力 NIAH 重试在 1K、2K、4K、8K、16K 完成并写入 `main_full_retry.json`，请求 32K 时因需要分配约 72.3 GiB 而触发 `RESOURCE_EXHAUSTED`；报告保留该失败记录，不将缺失的 32K 结果写成零分。

## 4. 污染与可比性检查

在正式评测前，训练语料、tokenizer 训练文本和 benchmark 的题目/答案需要做 exact match、Unicode NFC 归一化后的匹配、长子串匹配和近似匹配。每个 benchmark 还要固定 revision、split、prompt、shot 数、答案归一化、超时规则和重复次数。闭源模型只能作为外部参考，不能在不等价协议下写成“击败”。

## 5. Release gate

Release gate 的当前状态如下：

1. plain baseline 与 TTT 已在相同 token budget、数据配方和检查点步数下配对；
2. 主实验保留 seed 0，消融队列还运行了 seed 1 和 seed 2，并在 NLL、NIAH 和 benchmark 原始 JSON 中保留样本数与置信区间字段；
3. 已完成 MMLU 之外的 GSM8K 和 HumanEval，另有 ARC、PIQA、GPQA 与 BBH 诊断；
4. NIAH 和 synthetic preflight 已完成，但正式 RULER、LongBench、DROP 仍待固定 revision、任务集合和污染审计后运行；
5. 数据污染审计、配置与脚本摘要、数据哈希、检查点元数据和复现实验命令已随报告保存。

因此当前报告可以复现 1.0 主实验和扩展诊断，但在正式长上下文套件完成前，不应声称覆盖 RULER 或 LongBench 的官方分数。
