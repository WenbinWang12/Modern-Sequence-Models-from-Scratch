# Benchmark 测评协议与当前进度

!!! note "历史记录"
    本页是 2026-11-17 流程验证阶段的记录（小模型、字节级 tokenizer、WikiText-2），用于确认训练和评测链路可用，不代表模型能力。正式实验和结论见 [TTT-E2E 技术报告](../technical_report.md)。

技术报告中的 benchmark 不是一张可以随意挑选的榜单。每个分数都必须绑定模型检查点、数据版本、tokenizer、提示模板、shot 数、上下文上限、适配开关和有效样本数。本页把公开报告中常见的测评按能力分层，并记录本项目已经实际运行的协议。

## 1. 分层测评矩阵

| 层级 | 任务 | 主要回答的问题 | 当前状态 |
| --- | --- | --- | --- |
| 训练健康度 | held-out NLL、PPL、数据混合比例、重复率 | 模型是否学会了训练分布，训练是否稳定 | 已运行 |
| 基础知识与常识 | MMLU、ARC-E/ARC-C、OpenBookQA、PIQA、HellaSwag、WinoGrande | 小模型是否形成可迁移的知识和常识判断 | MMLU 诊断脚本已完成 |
| 数学推理 | GSM8K、MATH、AIME/AMC | 能否进行多步符号推理 | 待 tokenizer 和模型规模稳定后运行 |
| 代码 | HumanEval、MBPP、LiveCodeBench | 代码生成、执行正确性和时间切分泛化 | 待运行 |
| 难知识与推理 | BBH、GPQA、TruthfulQA | 长链推理、专业知识和事实性 | 待运行 |
| 长上下文 | RULER、LongBench、DROP | 随上下文长度增长时的检索和组合能力 | 已有合成检索诊断，正式套件待运行 |
| 中文能力 | CMMLU、C-Eval | 中文知识和考试型推理 | 双语 tokenizer 冻结后运行 |

这些层级不能互相替代。比如 PPL 较低不能推出 GSM8K 较高；MMLU 的零样本答案选择也不能替代代码执行评测。正式论文表格至少要覆盖知识/常识、数学或代码、以及长上下文中的三类，并同时给出 held-out NLL 作为训练分布指标。

## 2. 当前 MMLU 诊断协议

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

## 3. 污染与可比性检查

在正式评测前，训练语料、tokenizer 训练文本和 benchmark 的题目/答案需要做 exact match、Unicode NFC 归一化后的匹配、长子串匹配和近似匹配。每个 benchmark 还要固定 revision、split、prompt、shot 数、答案归一化、超时规则和重复次数。闭源模型只能作为外部参考，不能在不等价协议下写成“击败”。

## 4. Release gate

当前结果仍属于 pilot。进入技术报告主表前，需要补齐：

1. 与 plain baseline 的同 seed、同 token budget 配对结果；
2. 至少三个随机种子和置信区间；
3. MMLU 之外的一项数学或代码 benchmark；
4. 至少一项公开长上下文 benchmark；
5. 数据污染审计、完整版本信息和可复现实验命令。
