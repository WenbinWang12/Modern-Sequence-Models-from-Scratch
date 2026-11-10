# 17　内循环与外循环：学会怎样适应

## 从一次预测区分两个时间尺度

上一章把一个 token 沿着前向路径送到输出头。现在设模型读到一段介绍新术语的文本：它刚刚预测过一个词，也已经看到了正确答案。我们希望这次经验能帮助它理解后面的内容，同时希望这种“如何利用经验”的能力能从许多训练文本中学出来。这是两个不同问题，因此需要两条更新路径。

本章沿用 tiny 配置的隐藏宽度 \(d=16\) 和 chunk 长度 \(C=4\)。读完后，你应能写出一个 chunk 使用哪个快状态、哪一个标签触发更新，以及外循环梯度如何经过这次更新。需要的背景是上一章的 prefix/suffix 和基线章的交叉熵；下面仍会展开链式法则，不把“自动求导”当成解释。

用 \(\theta_k\) 表示第 \(k\) 个外循环训练步的静态参数，用 \(\phi_c\) 表示当前序列第 \(c\) 个 chunk 开始时的快权重。这里 \(c\) 数 chunk，不数 token。一个序列开始时，代码从 \(\theta_k\) 中提取快权重初值 \(\phi_0\)；它不是重新抽取一套随机矩阵。

~~~text
固定 θk
  φ0 → chunk 0 的预测和 loss → φ1 → chunk 1 的预测和 loss → φ2
                   全序列 loss 对 θk 求导
                              ↓
                            θk+1
~~~

内循环改变当前序列的状态；外循环改变下一次训练时使用的静态参数和快权重初值。一次训练样本完成后的 \(\phi\) 不会直接成为下一条独立样本的起点；推理时是否继续使用它，则取决于是否继续同一个 session。

## 从 token 交叉熵推到快权重更新

先用一个刻意简化的二分类语言模型连接“预测”和“梯度”。词表只有 \(A,B\)，快参数为一个标量 \(\phi\)，输出 logits 为 \(\mathbf z=(0,\phi)\)。它没有真正的 attention 或 MLP，只用于看清导数。当前标签是 \(B\)，则

$$
p_\phi(B)=\frac{e^\phi}{1+e^\phi},\qquad
\ell(\phi)=-\log p_\phi(B)=\log(1+e^\phi)-\phi.
$$

对两项分别求导：

$$
\frac{d\ell}{d\phi}
=\frac{e^\phi}{1+e^\phi}-1
=p_\phi(B)-1.
$$

若 \(\phi_0=0\)，两个候选概率各为 \(0.5\)，损失为 \(\log2\approx0.6931\)，梯度为 \(-0.5\)。学习率为 \(1\) 的一步 SGD 得到 \(\phi_1=0-1(-0.5)=0.5\)。代回公式后，\(p_{\phi_1}(B)\approx0.6225\)，损失约为 \(0.4741\)。我们已经能手算一次“观测标签改变下一次预测”的过程。

若两个位置分别预测 \(B\) 和 \(A\)，并且这个极简模型在两个位置都输出同一个 \((0,\phi)\)，则在 \(\phi=0\) 时平均梯度为 \([(-0.5)+0.5]/2=0\)。没有更新并不一定意味着代码坏了，也可能是模型没有能区分上下文的表示。实际网络的 logits 依赖位置对应的 hidden，因此参数梯度是

$$
\nabla_\phi\mathcal L_{\mathrm{inner}}
=\frac{1}{\sum_t m_t}
\sum_t m_t
\left(\frac{\partial\mathbf z_t}{\partial\phi}\right)^{\mathsf T}
(\mathbf p_t-\operatorname{onehot}(y_t)),
\qquad \sum_t m_t>0.
$$

右边先计算每个词表 logit 的误差，再经由 logits 对快矩阵的 Jacobian 传回参数。前例中第一个 logit 的导数为 \(0\)，第二个为 \(1\)，所以只留下 \(p(B)-1\)。代码对整个 pytree 求导，包含多块矩阵；它没有让矩阵元素直接拟合 token ID 的数值。

Jacobian 可以看成一张导数表：第 \(v\) 行、第 \(i\) 列记录第 \(i\) 个参数略微变化时，第 \(v\) 个 logit 怎样变化。若把快矩阵展平成 \(n_\phi\) 个参数，这张表的形状为 \([V,n_\phi]\)；转置后乘 \([V]\) 的误差向量，得到 \([n_\phi]\) 的参数梯度。实际自动求导通常直接计算这个乘积，无需把整张表存下来。

## 裁剪改变的是步幅，不是标签

以平方误差 \(\ell(\phi)=\tfrac12(\phi-3)^2\) 为另一个教学例子，\(\phi_0=1\) 时梯度为 \(-2\)。学习率 \(0.5\) 的普通 SGD 把状态更新到 \(2\)。这个例子只解释更新算术，不是本模型实际采用的内循环目标。

若所有快权重叶子的梯度记为 \(g_j\)，代码先求全局范数

$$
\widetilde{\|g\|}_2=\sqrt{\sum_j\sum_i g_{j,i}^2+10^{-12}},
\quad
a=\min\left(1,\frac{\kappa}{\widetilde{\|g\|}_2}\right),
\quad
\phi'=\phi-\eta_{\mathrm{inner}}ag.
$$

平方根内的小量与代码一致，用于避免零梯度处高阶求导的数值问题。当 \(\kappa=0.5\) 时，上面的标量梯度被乘以约 \(0.25\)，所以更新后约为 \(1.25\)。clip norm 限制整个树的梯度长度，不是逐元素截断，也不是直接限制参数范数。报告必须同时记录学习率、裁剪阈值、梯度范数和更新范数；只看状态变化无法区分这些因素。

## 四个输入位置需要五个 token

长度为 \(C\) 的预测块需要 \(C\) 个标签。对输入 \(x_1,x_2,x_3,x_4\)，标签是 \(x_2,x_3,x_4,x_5\)。因此在线读取到 \(x_4\) 时，仍不知道最后一个标签 \(x_5\)；必须等它到来，才可以完成这块的更新。

| 阶段 | 输入的数学位置 | 预测时使用的状态 | 更新所需标签 | 后续状态 |
|---|---|---|---|---|
| chunk 0 | \(1,\ldots,4\) | \(\phi_0\) | \(x_2,\ldots,x_5\) | \(\phi_1\) |
| chunk 1 | \(5,\ldots,8\) | \(\phi_1\) | \(x_6,\ldots,x_9\) | \(\phi_2\) |
| 尾部不足 4 个预测位置 | 实际剩余位置 | 当前 \(\phi\) | 仅有的真实标签 | 评分，但保留旧 \(\phi\) |

训练时整条序列已经在内存里，仍必须维持“本块先预测，后续块才使用更新结果”的因果约束。<code>sequence_loss</code> 将尾部补齐以获得固定形状；它会计算真实位置的 loss，却用条件选择保留不完整块的旧快权重。尾块不是只要 mask 非零就更新。

## 公式对应哪些函数与数组

<code>ttt/inner_loop.py</code> 的 <code>inner_step</code> 执行 suffix、计算交叉熵、对 fast weights 求导并调用 <code>sgd_update</code>。输出 cache 来自更新前的预测；历史 KV 不会因为快权重更新而被重新编码。可选的 <code>track_post_loss</code> 对同一块再算一次更新后的 loss，它是适应诊断，不是新的 held-out 证据。

| 代码对象 | 单序列形状或类型 | 语义 |
|---|---|---|
| <code>hidden</code> | \([C,d]\) | prefix 的输出 |
| <code>targets</code> | \([C]\) | 下一 token 的整数 ID |
| <code>mask</code> | \([C]\) | 标签权重 |
| <code>fast_weights</code> | pytree | 要被内循环改变的数组 |
| <code>cache</code> | 每层状态的 tuple | suffix 读取的历史 |
| scan carry | \((\phi_c,\text{cache}_c)\) | 传给下一 chunk 的状态 |

<code>sequence_loss</code> 用两层 <code>lax.scan</code> 扫过 chunk，并用 <code>jax.checkpoint</code> 在反向时重算部分中间值。这里的 checkpoint 是计算重算，不是磁盘训练检查点；它节省反向保存的激活，但本身不切断梯度。

## 外循环：不同样本如何合成一个目标

若样本 \(b\) 的监督权重和为 \(n_b=\sum_t m_{b,t}\)，其平均损失为 \(\mathcal L_b\)，则全 batch 的加权平均是

$$
\mathcal L_{\mathrm{batch}}
=\frac{\sum_b n_b\mathcal L_b}{\sum_b n_b},
\qquad \sum_b n_b>0.
$$

对于二值 mask，\(n_b\) 就是有效标签数。100 个标签、平均 loss 为 2 的样本，与 2 个标签、平均 loss 为 10 的样本，合并后是 \(220/102\approx2.157\)，不是 \((2+10)/2=6\)。只有总权重为零时，代码才用安全分母 1 并返回零 NLL；分数权重的正分母小于 1 时也不能被截成 1。全零 mask 没有监督信息，零 NLL 不是有用训练或参数必然不变的证据。

<code>ttt/training.py</code> 对这个目标求静态参数梯度，再交给外循环 Optax optimizer。MoE 正则项会加入训练目标，但 <code>nll</code> 单独记录。内循环学习率还有独立的 warmup：从 <code>inner_lr_start</code> 线性走向目标学习率；外循环则使用 warmup 和 cosine schedule。两者时间表必须分别固定和记录。

## 外循环为什么包含二阶导数

设快权重初值为 \(\phi_0(\theta)\)，先对未触发裁剪的一步 SGD 展开依赖：

$$
\phi_1=\phi_0(\theta)-\eta g(\theta,\phi_0(\theta)),
\qquad g=\nabla_\phi\mathcal L_{\mathrm{inner}}.
$$

为了看清每条路径，令 \(J_0=d\phi_0/d\theta\)，并假设本次更新的 \(\eta\) 不依赖 \(\theta\)。则

$$
\frac{d\phi_1}{d\theta}
=J_0-\eta\left(
\frac{\partial g}{\partial\theta}
+\frac{\partial g}{\partial\phi}J_0
\right),
$$

$$
\frac{d\mathcal L_{\mathrm{outer}}}{d\theta}
=\frac{\partial\mathcal L_{\mathrm{outer}}}{\partial\theta}
+\frac{\partial\mathcal L_{\mathrm{outer}}}{\partial\phi_1}
\frac{d\phi_1}{d\theta}.
$$

\(g\) 已是一阶导数，再对它求导就出现二阶项。full meta-gradient 保留这些依赖。常见的一阶近似忽略 \(g\) 的导数，仍保留 \(J_0\)；若对整个 \(\phi_1\) 调用 <code>stop_gradient</code>，连初始化路径也会消失。这两种处理不等价，不能统称“一阶”。元学习中关于可适应初始化的原始背景见 [MAML](https://arxiv.org/abs/1703.03400v3)。

回到前面的二分类模型，可以把这三个结果手算出来。设 \(\phi_0=\theta=0\)，内循环标签为 \(B\)、学习率为 1，更新后仍有 \(\phi_1=0.5\)。因为 \(g(\phi)=p_\phi(B)-1\)，所以 \(dg/d\phi=p_\phi(B)(1-p_\phi(B))\)，在零点为 \(0.25\)。于是 \(d\phi_1/d\theta=1-0.25=0.75\)。

只为观察后续预测的导数，暂时把外目标选为下一个位置的标签 \(A\)，即 \(\mathcal L_{\rm future}=\log(1+e^{\phi_1})\)。它对 \(\phi_1\) 的导数是 \(0.6225\)。完整链式法则给出约 \(0.6225\times0.75=0.4669\)；忽略内梯度的导数会得到 \(0.6225\)；彻底断开 \(\phi_1\) 则得到 \(0\)。这是三个不同计算，不是同一个梯度的三种写法。真实序列目标还包括其他位置的 loss，这里单取后续项是为了隔离“经过更新”的路径。

实际更新若触发裁剪，应把上式的 \(g\) 换成 \(a(g)g\)，其导数也包含裁剪系数的依赖；不能直接把未裁剪推导当作全部实现。当前内循环没有提供一个可直接切换 FOMAML 的配置项；外循环沿可微更新求导。论文对照若要加入一阶近似，必须明确改了哪条导数路径，并另做数值检查，不能在报告中凭空写成已支持的选项。

## 远端最小验证与如何读报告

在已配置依赖的远端 GPU 节点、仓库根目录运行：

~~~bash
bash scripts/validate_gpu.sh outputs/ch17/validation.json
~~~

这会运行 <code>ttt.validate</code> 自带的 tiny case，不会读取你的正式训练语料，也不会启动 125M 训练。报告顶层应有 <code>passed</code>、<code>environment</code> 和 <code>cases</code>。各 case 的真实字段包括 <code>loss</code>、<code>finite_gradients</code>、<code>logits_max_error</code>、<code>compilation_seconds</code>、<code>training_seconds</code>、<code>state_bytes</code>、<code>inner_updates</code>。

成功要求 loss 和梯度有限、流式误差不超过脚本的 \(10^{-3}\) 门槛，并且改变未来 token 后的前缀 logits 通过脚本内的容差断言。因果性没有独立 JSON 布尔字段；只有断言未抛出异常后才会写 <code>passed=true</code>。GPU 未识别先检查后端；误差超限先检查 chunk 的标签边界和 cache；非有限梯度先检查学习率、裁剪和数值精度。异常报告使用 <code>passed=false</code>、<code>error_type</code>、<code>error</code>。

## 小结：状态变化还不是质量提升

本章建立了 token loss → 快权重梯度 → 后续 chunk 状态 → 外循环导数这条链。功能报告只能证明这些小配置完成了列出的检查；没有证明正式语料的泛化收益，没有验证所有长度，也没有执行 session snapshot 恢复。

质量对照 <code>C2/F2</code>（或 <code>C4/F4</code>）要固定架构、初始化、数据、预算和外循环日程，仅改变内循环是否更新；名称中的数字指配方的 full-attention 层数。删除 MLP′ 与关闭更新是两项不同消融。下一章将同一个“标签何时可用”的约束放到逐 token 推理中，检查缓存、未完成块和快权重能否一起推进。
