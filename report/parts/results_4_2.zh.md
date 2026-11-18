表 1 是四个模型在两个 held-out 集上的最终结果。三组配对差都以全注意力为基准，在同一批窗口上计算。

{% if r.fig_context_4k.paired.ttt_minus_full %}在 PG-19 test 上，TTT-E2E 的每 token 损失比全注意力低 {{ f3(-r.fig_context_4k.paired.ttt_minus_full.all.mean) }} nat（95% 置信区间 {{ ci(r.fig_context_4k.paired.ttt_minus_full.all.ci95) }}），比滑窗注意力低 {{ f3(-r.fig_context_4k.paired.ttt_minus_swa.all.mean) }} nat。FineWeb-Edu held-out 上的方向一致，幅度略小（比全注意力低 {{ f3(-r.fig_context_4k_fineweb.paired.ttt_minus_full.all.mean) }} nat）。

这个 0.072 nat 的量级需要一个参照才有意义。同一张表里，**把注意力窗口从 512 扩大到完整的 4,096 个位置**（滑窗 → 全注意力）只带来 {{ f3(-r.fig_context_4k.paired.full_minus_swa.all.mean) }} nat 的改善。也就是说，在这个规模和这份数据上，测试时更新带来的收益是"让模型看见全部上下文"的两倍以上。{% endif %}

这是本报告的主结论，但它的适用范围很窄：单一模型规模、单一数据配方、4,096 的训练长度。第 4.9 节给出三个随机种子和三种模型规模下的重复结果，第 6 节说明哪些条件下不应期待同样的结论。
