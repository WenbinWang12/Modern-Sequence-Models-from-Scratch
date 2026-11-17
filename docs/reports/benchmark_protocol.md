# Benchmark 测评协议与当前进度

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

## 3. 污染与可比性检查

在正式评测前，训练语料、tokenizer 训练文本和 benchmark 的题目/答案需要做 exact match、Unicode NFC 归一化后的匹配、长子串匹配和近似匹配。每个 benchmark 还要固定 revision、split、prompt、shot 数、答案归一化、超时规则和重复次数。闭源模型只能作为外部参考，不能在不等价协议下写成“击败”。

## 4. Release gate

当前结果仍属于 pilot。进入技术报告主表前，需要补齐：

1. 与 plain baseline 的同 seed、同 token budget 配对结果；
2. 至少三个随机种子和置信区间；
3. MMLU 之外的一项数学或代码 benchmark；
4. 至少一项公开长上下文 benchmark；
5. 数据污染审计、完整版本信息和可复现实验命令。
