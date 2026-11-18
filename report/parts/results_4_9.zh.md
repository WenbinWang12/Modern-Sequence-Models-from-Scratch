{% if r.seeds.arms.ttt %}**随机种子。** 表 10 给出三个种子（初始化和数据顺序同时改变）下的结果。各模型自身的标准差在 {{ f4(r.seeds.arms.swa.sd) }}–{{ f4(r.seeds.arms.ttt.sd) }} 之间，而 TTT-E2E 与滑窗注意力的差值是 {{ signed(r.seeds.paired.ttt_minus_swa.mean) }} ± {{ f4(r.seeds.paired.ttt_minus_swa.sd) }}，与全注意力的差值是 {{ signed(r.seeds.paired.ttt_minus_full.mean) }} ± {{ f4(r.seeds.paired.ttt_minus_full.sd) }}。差值的标准差与单个模型的标准差相当，而差值本身比它大一个数量级，因此结论不依赖于某一次幸运的初始化。

同一张表里还有一个值得注意的现象：滑窗注意力与全注意力的差只有 {{ signed(r.seeds.paired.swa_minus_full.mean) }} ± {{ f4(r.seeds.paired.swa_minus_full.sd) }}，**在种子噪声范围内无法区分**。在 2,048 的上下文和这个数据量下，把窗口从 256 扩大到全可见几乎没有帮助；而测试时更新有帮助。{% endif %}

{% if r.scaling.m %}**模型规模。** 表 11 在固定 1.64 亿 token 的预算下比较三种规模（6×384、8×512、12×768）。TTT-E2E 在三个规模上都优于两个基线，差距从最小规模的 {{ f3(r.scaling.xs.swa.nll - r.scaling.xs.ttt.nll) }} nat 到最大规模的 {{ f3(r.scaling.m.swa.nll - r.scaling.m.ttt.nll) }} nat，没有随规模增大而消失。

但三个点不足以拟合任何趋势，而且它们都在同一个很小的规模区间内、用同一个数据量。原论文在 1.25 亿到 30 亿参数上报告收益随规模变化的情况，本报告无法验证也无法反驳那一结论。{% endif %}
