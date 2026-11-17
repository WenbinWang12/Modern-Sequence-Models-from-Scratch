表 3 和图 4 把评测从 4K 训练长度外推到 32K。全注意力只进行不截断的直接外推。滑窗注意力、无更新对照和 TTT-E2E 在 32K 输入中继续按各自的运行规则处理，因此位置差异同时包含状态压缩、窗口截断和外推误差。我们不把推理时强行限制为最近 4K 的可见范围称为匹配的全注意力对照，因为它会同时改变评测位置分布和可见历史。

{% if r.fig_context_32k.paired.ttt_minus_swa %}在 32K 的 PG-19 窗口上，TTT-E2E 减去滑窗注意力的配对差为 {{ f3(r.fig_context_32k.paired.ttt_minus_swa.all.mean) }} nat/token，95% 置信区间为 {{ ci(r.fig_context_32k.paired.ttt_minus_swa.all.ci95) }}。表 3 的位置分段可以进一步判断这个总体差异是否主要出现在训练长度之后。{% endif %}

32K 结果是长度外推诊断。它不等价于在 32K 上训练过的模型，也不能替代 RULER 或 LongBench 的任务分数。结合 fast-weight norm、会话状态和不同位置区间阅读，可以判断损失变化来自状态漂移、局部窗口还是模型在训练长度之外的分布偏移。
