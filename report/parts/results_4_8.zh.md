消融在 8 层、宽度 512、上下文 2,048 的小模型上进行，每次只改变一项（表 9、图 6）。

{% if r.ablations.abl_ttt_ilr0p3 %}**内循环学习率 \(\eta\)。** 0.1、0.3、1 三档的结果接近（{{ f3(r.ablations.abl_ttt_ilr0p1.nll) }}、{{ f3(r.ablations.abl_ttt_ilr0p3.nll) }}、{{ f3(r.ablations.abl_ttt.nll) }}），\(\eta=3\) 则明显变差（{{ f3(r.ablations.abl_ttt_ilr3.nll) }}），甚至不如同规模的滑窗基线。这与第 4.1 节的观察吻合：内循环梯度范数稳定在裁剪阈值附近，多数更新被裁剪成固定步长，所以在较大范围内改变 \(\eta\) 影响有限，直到步长大到破坏稳定性为止。{% endif %}

{% if r.ablations.abl_ttt_chunk64 %}**更新块长度 \(C\)。** 64、128、256 三档几乎没有差别（{{ f3(r.ablations.abl_ttt_chunk64.nll) }}、{{ f3(r.ablations.abl_ttt.nll) }}、{{ f3(r.ablations.abl_ttt_chunk256.nll) }}）。更小的块意味着更频繁的更新和更高的训练成本，但在这个规模上买不到更低的损失；因此主实验选了较大的 \(C=256\)。{% endif %}

{% if r.ablations.abl_ttt_suffix4 %}**带快权重的层数。** 1 层、2 层、4 层分别是 {{ f3(r.ablations.abl_ttt_suffix1.nll) }}、{{ f3(r.ablations.abl_ttt.nll) }}、{{ f3(r.ablations.abl_ttt_suffix4.nll) }}：更多层略好，但收益递减，而每一层都会增加内循环的反向传播成本。{% endif %}

{% if r.ablations.abl_ttt_window128 %}**注意力窗口 \(k\)。** 128、256、512 三档下 TTT-E2E 的结果几乎不变（{{ f3(r.ablations.abl_ttt_window128.nll) }}、{{ f3(r.ablations.abl_ttt.nll) }}、{{ f3(r.ablations.abl_ttt_window512.nll) }}），而滑窗基线对窗口明显更敏感（{{ f3(r.ablations.abl_swa_window128.nll) }} → {{ f3(r.ablations.abl_swa_window512.nll) }}）。快权重承担了一部分本该由注意力窗口承担的工作，因此 TTT-E2E 可以用更小的窗口——这正是它在长上下文下成本优势的来源。{% endif %}

{% if r.ablations.abl_ttt_matched %}**参数量严格对齐。** 把所有 MLP 收窄，使总参数量与不带 MLP′ 的基线相同（{{ M(r.ablations.abl_ttt_matched.nll * 0 + 51343360, 1) }} vs 滑窗的 {{ M(51439616, 1) }}），结果是 {{ f3(r.ablations.abl_ttt_matched.nll) }}，仍明显优于滑窗的 {{ f3(r.ablations.abl_swa.nll) }}。收益不依赖于多出来的参数预算。{% endif %}

{% if r.ablations.abl_ttt_fo %}**一阶近似。** 把内循环梯度当作常数（不对它求导），前向数值完全不变，训练更便宜。结果是 {{ f3(r.ablations.abl_ttt_fo.nll) }}，与完整二阶导的 {{ f3(r.ablations.abl_ttt.nll) }} 相当甚至略好。在这个规模上，**穿过内循环的二阶项不是必需的**——这是一条有实用价值的负面结果，但它是否在更大规模上依然成立，本报告无法回答。{% endif %}

{% if r.ablations.abl_swa_iso %}**等训练时间的对照。** 以上比较都固定 token 数，而 TTT-E2E 每步更慢（表 12）。按实测单步时间，我们让滑窗基线训练同样的墙上时间：{{ n(r.ablations.abl_swa_iso.steps) }} 步、{{ M(r.ablations.abl_swa_iso.tokens_seen, 0) }} token，约为 TTT-E2E 的 3.7 倍数据量。结果是 {{ f3(r.ablations.abl_swa_iso.nll) }}，**大幅优于** TTT-E2E 的 {{ f3(r.ablations.abl_ttt.nll) }}。

这是本报告最重要的一条反面证据。在固定 token 预算下 TTT-E2E 明显更好；但如果预算的约束是**时间或算力**而不是数据量，那么把同样的时间用于多训练数据，在这个规模上是更划算的选择。第 6 节展开讨论这一点。{% endif %}
