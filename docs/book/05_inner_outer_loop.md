# 16　内循环与外循环：两种时间尺度的优化

## 同一个 loss 里有两条更新路径

TTT 的难点不是写出一个 SGD，而是定义哪些梯度应该流向哪里。对一段序列，模型先用当前快权重预测，再计算局部 loss 和更新快权重；外循环再根据整个序列的预测结果更新静态模型。

~~~text
static parameters θ
        │
        ├── initialize fast weights φ0
        │
        └── prefix and suffix computation
                    │
       φ0 → inner loss → SGD → φ1 → inner loss → ...
                    │
                    └──────── outer loss → AdamW(θ)
~~~

$\phi$ 的更新是序列内状态转移，$\theta$ 的更新是训练步之间的参数转移。把它们都叫“模型参数更新”会掩盖最重要的算法差异。

## inner_step 做了什么

`inner_step` 接收 model、静态 params、当前 fast_weights、hidden、targets、mask、cache 和 position。它定义了一个 objective：

1. 用当前 fast weights 执行 suffix_hidden；
2. 将 features 投影到 logits，或使用分块交叉熵；
3. 依据 mask 计算 token loss；
4. 对 fast weights 求梯度；
5. clip 梯度并做 SGD；
6. 可选地计算更新后的 post loss。

更新前 loss 用来产生梯度，post loss 用来观察“更新是否让同一块更容易”。post loss 默认关闭，因为额外前向会增加成本；它也不应该被误认为 outer loss。

梯度裁剪使用所有快权重叶子的全局平方和。小量 1e-12 用于避免零梯度处的除零。clip norm 是内循环稳定性参数，不能与外循环的 clip norm 混成一个报告字段。

## sequence_loss 如何扫描一条序列

`sequence_loss` 首先检查一维 token 和长度，再将有效长度 pad 到 chunk size 的整数倍。prefix 一次计算所有位置，suffix 通过两个层次的 lax.scan 处理 chunk。

为什么要两层 scan？外层把 chunk 分组，内层扫描组内步骤；对 step 使用 jax.checkpoint，可以在反向时重算前向，减少保存的中间状态。它是显存-计算的工程取舍，不改变“按 chunk 更新 fast weights”的语义。

每个 chunk 的 carry 包含 current_fast 和 current_cache。输出包括 token losses、grad norms、router loads、fast weight norms、update norms 和 post losses。尾块位置通过 mask 评分，但更新使用 jnp.where 保留旧 fast weights，避免 padding 引入适应。

## 外循环的 weighted loss

训练 batch 内每个样本可能有不同有效 mask 数。`training.py` 对样本调用 sequence_loss，取得每个样本 loss 和有效 token count，再用：

$$
\mathcal{L}_{\mathrm{batch}}=
\frac{\sum_b n_b \mathcal{L}_b}
{\max(\sum_b n_b,1)}
$$

聚合。这样一个含 100 个有效 token 的样本不会和一个含 2 个有效 token 的样本拥有相同权重。

随后对 params 求 value_and_grad，必要时在 data axis 上 pmean，交给 Optax optimizer 更新。metrics 记录 outer loss、nll、grad norm、inner lr 和状态统计。MoE 还加入 auxiliary 与 z loss，路由统计不应被当成语言模型 nll。

## inner warmup 和外循环 schedule

当前训练配置有两个 schedule。外循环 learning rate 由 warmup 与 cosine decay 组合；内循环 learning rate 可以从 inner_lr_start 逐渐接近目标值。

这两个 warmup 解决的风险不同：

- outer warmup 控制静态参数初期的更新幅度；
- inner warmup 控制快权重在随机初始化附近是否过快改变。

若只记录总 loss，不记录 inner_lr，就无法解释模型在前几百步是否经历了不同的适应强度。对照实验也必须固定 schedule，否则“开启 TTT”的收益可能来自不同的有效学习率。

## 一阶与高阶梯度

在严格的 meta-learning 视角下，fast update 本身依赖 theta，outer gradient 还要通过 update 计算。当前路径使用 JAX 的可微函数和 checkpoint scan，因此默认可能保留这条依赖；但具体实验是否截断，需要由配置和实现明确。

可把两个版本分别命名：

- first-order：对 inner update stop_gradient；
- full meta-gradient：对 inner update 保留二阶路径。

这不是数值优化开关，而是算法定义。比较时要记录版本、显存、编译时间和最终质量。不能拿一阶版本的便宜成本与高阶版本的质量直接比较。

## C/F 对照怎样成立

C 与 F 应共享：

- 同一模型结构和初始化 seed；
- 同一 tokenizer、manifest 和样本顺序；
- 同一 sequence length、batch 和 token budget；
- 同一外循环 optimizer 和 schedule；
- 同一硬件与精度。

C 只关闭 inner update，F 打开它。若 C 删除了 MLP′ 参数而 F 包含它，这已经同时改变了参数量；需要额外的结构对照，不能称为纯 C/F。

## 用什么证据判断内循环有效

单步 fast weight norm 变化不能说明质量提升。至少看：

1. 更新前和更新后 loss；
2. prefix 未来 token 是否泄漏；
3. fast weight update norm 是否稳定；
4. outer gradient 是否有限；
5. resume 后下一步是否连续；
6. 同预算 C/F 的 held-out loss 与计算成本。

下一章把这些序列状态放进 attention cache、Gated Delta 和服务 session，讨论训练状态如何变成推理状态。
