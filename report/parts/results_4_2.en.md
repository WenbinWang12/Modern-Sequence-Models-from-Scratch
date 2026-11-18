Table 1 gives the final results of all four models on both held-out sets. The paired differences are measured against full attention on the same windows.

{% if r.fig_context_4k.paired.ttt_minus_full %}On PG-19 test, TTT-E2E's loss per token is {{ f3(-r.fig_context_4k.paired.ttt_minus_full.all.mean) }} nats below full attention (95% CI {{ ci(r.fig_context_4k.paired.ttt_minus_full.all.ci95) }}) and {{ f3(-r.fig_context_4k.paired.ttt_minus_swa.all.mean) }} nats below sliding-window attention. FineWeb-Edu held-out points the same way with a smaller margin ({{ f3(-r.fig_context_4k_fineweb.paired.ttt_minus_full.all.mean) }} nats below full attention).

The size of that 0.072 nat gap needs a yardstick. In the same table, **widening the attention window from 512 to all 4,096 positions** (sliding window → full attention) is worth only {{ f3(-r.fig_context_4k.paired.full_minus_swa.all.mean) }} nats. At this scale and on this data, test-time updates buy more than twice what letting the model see the whole context buys.{% endif %}

This is the report's main result, and its scope is narrow: one model size, one data recipe, a training length of 4,096. Section 4.9 repeats it across three seeds and three model sizes, and Section 6 states where the conclusion should not be expected to hold.
