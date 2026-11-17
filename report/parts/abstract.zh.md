我们从零实现了端到端测试时训练（TTT-E2E）：模型一边读上下文，一边用读过的内容对自己最后几层中的一组 MLP 做梯度下降，训练阶段则通过内循环求导，让模型学会“被这样更新”。在两张 H200 上，我们用完全相同的数据、batch 顺序、优化器和 token 预算训练了四个约 1.3 亿参数、上下文 4,096 的模型：全注意力、滑窗注意力（窗口 512）、与 TTT 结构和参数量相同但不做更新的对照，以及 TTT-E2E。训练语料是 FineWeb-Edu 与 PG-19 各半、共 {{ n(r.training.swa.tokens_seen) }} 个 GPT-2 BPE token。

{% if r.fig_context_4k.paired.ttt_minus_swa %}在 PG-19 test 上，TTT-E2E 的每 token 损失比滑窗注意力{{ "低" if r.fig_context_4k.paired.ttt_minus_swa.all.mean < 0 else "高" }} {{ f3(abs(r.fig_context_4k.paired.ttt_minus_swa.all.mean)) }} nat（95% 置信区间 {{ ci(r.fig_context_4k.paired.ttt_minus_swa.all.ci95) }}），比全注意力{{ "低" if r.fig_context_4k.paired.ttt_minus_full.all.mean < 0 else "高" }} {{ f3(abs(r.fig_context_4k.paired.ttt_minus_full.all.mean)) }} nat。差距随可用上下文的长度变化，详见第 4.3 节；把评测窗口延长到训练长度 8 倍（32,768 个 token）的结果见第 4.5 节。{% else %}主要数值将在四个模型全部训练完成后填入本摘要。{% endif %}

报告把“额外参数”、“测试时更新”和“训练时的元学习”三个因素分开度量：无更新对照回答第一项，同一检查点关闭更新回答第二项，对未经元训练的对照直接做测试时梯度更新（动态评估）回答第三项。成本方面，{% if r.training_cost.ttt %}TTT-E2E 的训练吞吐是滑窗注意力的 {{ f2(r.training_cost.ttt.tokens_per_second / r.training_cost.swa.tokens_per_second) }} 倍，{% endif %}而推理时的会话状态大小和每 token 成本与上下文长度无关。

我们同样度量方法的边界：逐字回忆探针和按 RULER 构造的单针检索，衡量被压进权重的历史在细节上保留了多少；十项公开零样本 benchmark 使用与 lm-evaluation-harness 逐项核对一致的协议（GPT-2 上的核对结果见第 4.7 节），并与一组训练量大一到三个数量级的公开模型放在同一张表里。数据审计覆盖采样配比、训练与评测集之间的重复，以及 benchmark 题目在训练语料中的逐字出现。本页的每个数字都由脚本从原始评测输出生成，原始输出随仓库发布。
