把评测窗口从 4,096 直接拉长到 32,768（训练长度的 8 倍），不做任何微调，也不改变模型。全注意力无法参与这项比较：它的位置编码只在 4,096 以内训练过，直接外推到 32K 会超出训练分布；我们单独测了它在 8,192 和 16,384 的表现，见 `results.json` 的 `full_extrapolation` 字段。

{% if r.fig_context_32k.paired.ttt_minus_swa %}在 32K 窗口内，TTT-E2E 相对滑窗注意力的优势扩大到 {{ f3(-r.fig_context_32k.paired.ttt_minus_swa.all.mean) }} nat（{{ ci(r.fig_context_32k.paired.ttt_minus_swa.all.ci95) }}），比 4K 窗口下的 {{ f3(-r.fig_context_4k.paired.ttt_minus_swa.all.mean) }} nat 更大。两个模型超过训练长度后损失都有所上升（图 4 左），但 TTT-E2E 上升得更慢。

这里有一个必须说清楚的区分：滑窗注意力和 TTT-E2E 都**没有长度外推问题**——它们的每一层都只看最近 512 个位置，位置编码在 32K 处遇到的相对距离与训练时完全相同。因此这张图衡量的不是"能否处理更长的输入"，而是"随着读入更多内容，快权重能否继续积累有用的信息"。TTT-E2E 的优势持续扩大，说明在 8 倍于训练长度的范围内，内循环仍在稳定工作而没有发散。{% endif %}

{% if r.fast_weights_32k.norm_by_chunk %}快权重的诊断支持这一点。在 32K 评测中，每个更新块后的快权重范数平稳变化，内循环梯度范数始终保持在裁剪阈值附近，{{ len(r.fast_weights_32k.norm_by_chunk) }} 个更新块里没有出现范数发散或塌缩。这是内循环在远超训练长度时仍然稳定的直接证据。{% endif %}
