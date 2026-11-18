{% if r.training_cost.ttt %}**训练。** 表 12 是单张 H200 上的实测吞吐。滑窗注意力最快（{{ K(r.training_cost.swa.tokens_per_second, 0) }} token/s），全注意力 {{ K(r.training_cost.full.tokens_per_second, 0) }}，TTT-E2E 最慢（{{ K(r.training_cost.ttt.tokens_per_second, 0) }}，即滑窗的 {{ f2(r.training_cost.ttt.tokens_per_second / r.training_cost.swa.tokens_per_second) }} 倍）。慢的原因有两部分：后段多一个 MLP′ 的前向，以及每个更新块一次穿过后段的反向传播，再加上外循环要对这条更新链求导。无更新对照跑到 {{ K(r.training_cost.ck.tokens_per_second, 0) }}，说明其中约一半的开销来自额外的 MLP′ 本身，另一半来自内循环。

第 4.8 节的等时间对照说明了这项成本的后果：同样的墙上时间，滑窗基线能多读 3.7 倍的数据，并因此取得更低的损失。{% endif %}

{% if r.sessions.ttt %}**推理。** 表 13 是单个会话实测的状态大小和解码速度。TTT-E2E 的会话状态由快权重（固定 {{ f1(r.sessions.ttt.rows[0].state_bytes.fast_weights / 1048576) }} MiB，与已读长度无关）和滑窗 KV 缓存（同样固定）组成；无论读入 2K 还是 32K 上下文，状态大小和逐 token 解码速度都不变。全注意力的 KV 缓存则随上下文线性增长。

这正是该方法的卖点所在：训练时多付出的代价，换来的是推理时与上下文长度无关的常数成本。在本报告的规模下这个权衡并不划算（模型小、上下文短，注意力本来就不贵），它要到更长的上下文和更大的服务量下才会显现价值——那超出了本报告能验证的范围。{% endif %}
