表 1 把最终检查点放在同一套 held-out 协议下比较。PG-19 test 代表书籍文本，FineWeb-Edu held-out 代表网页文本；两者使用独立于训练的窗口。NLL 和 PPL 衡量模型在各自分布上的预测质量，bits/byte 把不同 token 化带来的单位差异转换到字节尺度。差值按同一批窗口配对计算，区间反映窗口之间的变化，而不是把两个独立均值相减。

{% if r.fig_context_4k.paired.ttt_minus_swa %}在 PG-19 test 上，TTT-E2E 减去滑窗注意力的配对 NLL 差为 {{ f3(r.fig_context_4k.paired.ttt_minus_swa.all.mean) }} nat/token，95% 置信区间为 {{ ci(r.fig_context_4k.paired.ttt_minus_swa.all.ci95) }}。这里的符号约定是处理组减对照组，因此负值表示 TTT-E2E 更低。{% endif %}

主比较应先看 TTT-E2E 与滑窗注意力，再看全注意力和无更新对照。滑窗模型保留固定的最近窗口，全注意力提供训练长度内的可见历史上限，无更新对照保留 TTT 的参数结构但关闭内循环。这个排列让表中的质量差异可以对应到可见上下文、额外参数和实际更新三个因素。FineWeb-Edu 与 PG-19 的绝对 NLL 不直接横向解释，配对差和各数据集内部的置信区间是主要证据。
