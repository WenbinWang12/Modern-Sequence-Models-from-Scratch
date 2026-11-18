TTT-E2E 与滑窗注意力的差别有三个来源：后段多了一个 MLP′（额外参数）、推理时执行梯度更新（测试时计算）、以及训练时就在模拟这种更新（元学习）。四个模型加上两项专门的评测，可以把三者分开。

{% if r.fig_context_4k.paired.ck_minus_swa %}**额外参数本身。** 无更新对照与 TTT-E2E 结构相同、参数量相同，训练和推理都不做内循环。它比滑窗注意力低 {{ f3(-r.fig_context_4k.paired.ck_minus_swa.all.mean) }} nat（{{ ci(r.fig_context_4k.paired.ck_minus_swa.all.ci95) }}）——多出来的 {{ M(r.training_cost.ttt.fast_parameters, 1) }} 个参数确实有用，但只解释了总差距 {{ f3(-r.fig_context_4k.paired.ttt_minus_swa.all.mean) }} nat 中的一小部分。

**测试时更新本身。** TTT-E2E 比无更新对照低 {{ f3(-r.fig_context_4k.paired.ttt_minus_ck.all.mean) }} nat（{{ ci(r.fig_context_4k.paired.ttt_minus_ck.all.ci95) }}）。两者参数量和结构完全相同，差别只有"推理时是否按内循环更新快权重"以及"训练时是否模拟了这一过程"。这是三项里最大的一项。{% endif %}

{% if r.dynamic.ck %}**没有元学习的测试时更新够不够？** 这是关键的一问：如果对任意一个普通模型直接做测试时梯度下降（动态评估）就能得到同样的收益，那么训练阶段的二阶导就是不必要的开销。我们对无更新对照直接执行同一套内循环，并扫描测试时学习率。结果是几乎没有改善：最好的一档是 {{ f4(min(r.dynamic.ck.lr0p01.validation, r.dynamic.ck.lr0p03.validation, r.dynamic.ck.lr0p1.validation, r.dynamic.ck.lr0p3.validation, r.dynamic.ck.lr1.validation)) }}，而不更新时是 {{ f4(r.dynamic.ck.off.validation) }}。同一组超参数作用在经过元训练的 TTT-E2E 上，则从 {{ f4(r.dynamic.ttt.off.validation) }} 降到 {{ f4(r.dynamic.ttt.lr1.validation) }}。

换句话说，**有用的不是梯度更新这个动作，而是模型事先被训练成"可被这样更新"**。这正是端到端训练二阶导的代价所换来的东西。{% endif %}

{% if r.fig_context_4k.paired.ttt_noadapt_minus_ck %}还有一个方向相反的观察值得记录：把训练好的 TTT-E2E **关闭更新**后评测，它比无更新对照差 {{ f3(r.fig_context_4k.paired.ttt_noadapt_minus_ck.all.mean) }} nat（{{ ci(r.fig_context_4k.paired.ttt_noadapt_minus_ck.all.ci95) }}）。用更新训练出来的慢参数会**依赖**这些更新：它们把一部分建模工作交给了内循环，拿掉内循环就留下缺口。这意味着 TTT-E2E 不是一个"可以随时关掉以省算力"的功能，而是一个必须成套使用的设计。{% endif %}
