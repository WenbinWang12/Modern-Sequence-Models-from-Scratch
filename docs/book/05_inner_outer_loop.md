# 17　内循环与外循环：两种时间尺度的优化

## 同一个训练过程里有两条更新路径

TTT 的难点不是写出一个 SGD，而是定义哪些梯度应该流向哪里。对一段序列，模型先用当前快权重预测，再计算局部 loss 和更新快权重；外循环再根据整个序列的预测结果更新静态模型。

静态参数 \(\theta\) 先初始化快权重 \(\phi_0\)，然后经过 prefix 和 suffix 计算。序列内的状态转移可以写成

~~~text
φ0 → inner loss → SGD → φ1 → next chunk
~~~

外循环则根据 outer loss 更新 \(\theta\)。\(\phi\) 是序列内的状态转移，\(\theta\) 是训练步之间的参数转移。把它们都叫“模型参数更新”会掩盖最重要的算法差异。

## 一个可以手算的 inner SGD

先把复杂网络缩成一个标量。假设某个快权重是 \(\phi\)，当前块的理想目标是 3，内循环损失定义为

\[
\ell_{\mathrm{inner}}(\phi)=\frac{1}{2}(\phi-3)^2.
\]

如果 \(\phi_0=1\)，则

\[
\frac{\partial\ell_{\mathrm{inner}}}{\partial\phi}
=\phi_0-3=-2.
\]

学习率 \(\eta_{\mathrm{inner}}=0.5\) 时，普通 SGD 得到

\[
\phi_1=\phi_0-\eta_{\mathrm{inner}}(-2)
=1-0.5(-2)=2.
\]

这表示当前块的观测让快权重从 1 向目标 3 移动。实际模型里 \(\phi\) 是许多矩阵叶子的 pytree，loss 是 token 交叉熵，但更新仍然遵循同一个公式。

代码中的 <code>sgd_update</code> 还会做全局梯度裁剪。若梯度范数是 2、clip norm 是 0.5，则缩放因子为 \(0.5/2=0.25\)，有效更新变成

\[
\phi_1=1-0.5(0.25)(-2)=1.25.
\]

因此 inner learning rate 和 clip norm 必须分别记录；只看更新后的 \(\phi\) 不能知道它是学习率小，还是梯度被裁剪了。

## inner_step 做了什么

<code>inner_step</code> 接收 model、静态 params、当前 fast_weights、hidden、targets、mask、cache 和 position。它定义一个 objective：

1. 用当前 fast weights 执行 suffix_hidden；
2. 将 features 投影到 logits，或使用分块交叉熵；
3. 依据 mask 计算 token loss；
4. 对 fast weights 求梯度；
5. clip 梯度并做 SGD；
6. 可选地计算更新后的 post loss。

更新前 loss 用来产生梯度，post loss 用来观察“更新是否让同一块更容易”。post loss 默认关闭，因为额外前向会增加成本；它也不应该被误认为 outer loss。

代码的主要张量接口可以先记成：

| 对象 | 典型形状 | 作用 |
|---|---|---|
| hidden | \([C,d]\) | 当前 chunk 的 prefix 表示 |
| targets | \([C]\) | 当前 chunk 的 next-token 标签 |
| mask | \([C]\) | 哪些标签计入 inner loss |
| fast_weights | pytree | MLP′ 或其他可适应状态 |
| cache | 结构化状态 | suffix 的 KV/Delta 状态 |

梯度裁剪使用所有快权重叶子的全局平方和。小量 \(10^{-12}\) 用于避免零梯度处的除零。这个 clip norm 是内循环稳定性参数，不能与外循环的 clip norm 混成一个报告字段。

## sequence_loss 如何扫描一条序列

<code>sequence_loss</code> 首先检查一维 token 和长度，再将有效长度 pad 到 chunk size 的整数倍。prefix 一次计算所有位置，suffix 通过两个层次的 <code>lax.scan</code> 处理。

假设有效预测长度为 8、chunk size 为 4，状态转移如下：

| chunk | 输入位置 | 使用的 fast state | 输出状态 |
|---|---|---|---|
| 0 | 0–3 | \(\phi_0\)、cache\(_0\) | \(\phi_1\)、cache\(_1\) |
| 1 | 4–7 | \(\phi_1\)、cache\(_1\) | \(\phi_2\)、cache\(_2\) |

下一块必须使用上一块的更新结果；如果每块都重新读取 \(\phi_0\)，模型就没有跨 chunk 适应。尾块不足一个完整 chunk 时，mask 仍然可以评分真实位置，但代码用 <code>jnp.where</code> 保留旧 fast weights，避免 padding 引入适应。

为什么要两层 scan？外层把 chunk 分组，内层扫描组内步骤；对 step 使用 <code>jax.checkpoint</code>，可以在反向时重算前向，减少保存的中间状态。这是显存—计算的工程取舍，不改变“按 chunk 更新 fast weights”的语义。

每个 carry 包含 current_fast 和 current_cache。输出包括 token losses、grad norms、router loads、fast weight norms、update norms 和 post losses。

## 外循环怎样聚合不同长度的样本

训练 batch 内每个样本可能有不同有效 mask 数。<code>training.py</code> 对样本调用 <code>sequence_loss</code>，取得每个样本 loss 和有效 token count，再用

\[
\mathcal{L}_{\mathrm{batch}}
=
\frac{\sum_b n_b\mathcal{L}_b}
{\max(\sum_b n_b,1)}
\]

聚合。举例来说，样本 A 有 100 个有效 token、平均 loss 为 2，样本 B 有 2 个有效 token、平均 loss 为 10，则正确聚合是

\[
\frac{100\times2+2\times10}{102}\approx2.16,
\]

而不是简单平均得到 6。按有效 token 加权，才与“所有有效位置的总损失除以有效位置数”一致。

随后对 params 求 value_and_grad，必要时在 data axis 上 pmean，交给 Optax optimizer 更新。metrics 记录 outer loss、nll、grad norm、inner lr 和状态统计。MoE 还加入 auxiliary 与 z loss，路由统计不应被当成语言模型 nll。

## inner warmup 和外循环 schedule

当前训练配置有两个 schedule。外循环 learning rate 由 warmup 与 cosine decay 组合；内循环 learning rate 可以从 inner_lr_start 逐渐接近目标值。

这两个 warmup 解决的风险不同：

- outer warmup 控制静态参数初期的更新幅度；
- inner warmup 控制快权重在随机初始化附近是否过快改变。

若只记录总 loss，不记录 inner_lr，就无法解释模型在前几百步是否经历了不同的适应强度。对照实验也必须固定 schedule，否则“开启 TTT”的收益可能来自不同的有效学习率。

## 一阶与高阶梯度：链式法则在哪里

设快更新为

\[
\phi_1(\theta)
=
\phi_0(\theta)
-\eta\nabla_\phi\ell_{\mathrm{inner}}(\theta,\phi_0(\theta)),
\]

外循环目标为 \(\ell_{\mathrm{outer}}(\theta,\phi_1(\theta))\)。链式法则给出

\[
\frac{d\ell_{\mathrm{outer}}}{d\theta}
=
\frac{\partial\ell_{\mathrm{outer}}}{\partial\theta}
+
\frac{\partial\ell_{\mathrm{outer}}}{\partial\phi_1}
\frac{d\phi_1}{d\theta}.
\]

最后一项包含 \(\phi_0\) 对 \(\theta\) 的依赖，以及 inner loss 梯度对 \(\theta\) 的依赖，因而会引入二阶或 Hessian-vector 形式。full meta-gradient 保留这条路径；first-order 版本对 inner update 使用 stop_gradient，把 \(d\phi_1/d\theta\) 近似去掉。

这不是一个只影响速度的实现开关，而是两个不同算法。比较时要同时记录版本、显存、编译时间和最终质量。不能拿一阶版本的便宜成本与高阶版本的质量直接比较。

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

一个合格的报告还应列出每个 chunk 的 fast norm、update norm、有效 mask 数和是否发生尾块保留。这样读者才能判断“模型学到了更好的更新规则”，还是仅仅发生了更大的状态漂移。

下一章把这些序列状态放进 attention cache、Gated Delta 和服务 session，讨论训练状态如何变成推理状态。
