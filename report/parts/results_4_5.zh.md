表 3 和图 4 把训练长度 4K 外推到 32K。全注意力有两个参照：不截断的外推反映模型在更长输入上的直接注意力行为，只看最近 4K 的版本则把可见历史限制在训练时的范围。滑窗和 TTT-E2E 在 32K 中继续按各自的运行规则处理输入，因此表中的位置差异同时包含状态压缩、窗口截断和外推误差。

{% if r.fig_context_32k.paired.ttt_minus_swa %}在 32K 的 PG-19 窗口上，TTT-E2E 减去滑窗注意力的配对差为 {{ f3(r.fig_context_32k.paired.ttt_minus_swa.all.mean) }} nat/token，95% 置信区间为 {{ ci(r.fig_context_32k.paired.ttt_minus_swa.all.ci95) }}。表 3 的位置分段可以进一步判断这个总体差异是否主要出现在训练长度之后。{% endif %}

32K 结果是长度外推诊断。它不等价于在 32K 上训练过的模型，也不能替代 RULER 或 LongBench 的任务分数。结合 fast-weight norm、会话状态和不同位置区间阅读，可以判断损失变化来自状态漂移、局部窗口还是模型在训练长度之外的分布偏移。
