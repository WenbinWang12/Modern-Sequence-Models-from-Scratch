# 数学与深度学习工具箱

本章把全书会用到的最小数学与深度学习工具放在一起。目标是让读者能读懂张量形状、手算一个位置的概率和梯度，并解释一次参数更新；它不是一门完整的线性代数或微积分课程。

全书数学位置从 1 开始，代码数组索引从 0 开始。记 \(B\) 为 batch size，\(L\) 为序列长度，\(d\) 为 hidden size，\(V\) 为词表大小。

## 从标量到张量

标量只有一个数，例如学习率 \(\eta=0.1\)。向量是一列有顺序的数：

$$
\mathbf z=(2,1,0)\in\mathbb R^3.
$$

矩阵有两个轴，例如输出投影 \(W_{\mathrm{out}}\in\mathbb R^{d\times V}\)。若 hidden 是 \(h\in\mathbb R^{1\times d}\)，那么 \(hW_{\mathrm{out}}\in\mathbb R^{1\times V}\) 是一个位置对词表中所有 token 的 logits。

语言模型常见的张量形状如下：

| 对象 | 形状 | 含义 |
|---|---|---|
| token ID | \(B\times L\) | 每个位置一个整数坐标 |
| embedding | \(B\times L\times d\) | 每个 token 的向量 |
| hidden | \(B\times L\times d\) | 层之间传递的表示 |
| logits | \(B\times L\times V\) | 每个位置的词表分数 |
| loss mask | \(B\times(L-1)\) | 哪些目标进入 loss |

改变 \(B\) 主要改变并行样本数，改变 \(L\) 改变上下文和注意力开销，改变 \(d\) 改变表示宽度，改变 \(V\) 改变 embedding 与输出头的类别数。

## 形状如何沿模型流动

词表 embedding 矩阵为

$$
E\in\mathbb R^{V\times d}.
$$

输入 \(X\in\{0,\ldots,V-1\}^{B\times L}\) 用整数作为行坐标，得到

$$
E[X]\in\mathbb R^{B\times L\times d}.
$$

这不是把 token ID 当成有意义的实数；ID 只是词表中的索引。对每个位置应用 \(W\in\mathbb R^{d\times d'}\)，只改变最后一维：

$$
H\in\mathbb R^{B\times L\times d}
\quad\longrightarrow\quad
HW\in\mathbb R^{B\times L\times d'}.
$$

输出头使用 \(W_{\mathrm{out}}\in\mathbb R^{d\times V}\)，因此

$$
Z=HW_{\mathrm{out}}\in\mathbb R^{B\times L\times V}.
$$

这条形状关系直接连接到词表：改变 \(V\) 会改变输出头的最后一维，即使 \(d\) 和层数都不变。

### 注意力的形状

注意力把 hidden 投影成 query、key、value：

$$
Q=HW_Q,\qquad K=HW_K,\qquad V_h=HW_V.
$$

对一个头，投影矩阵可写成 \(W_Q,W_K,W_V\in\mathbb R^{d\times d_h}\)。多个头分别计算后在最后一维拼接，再通过输出投影回到 \(d\) 维。每个头的宽度为 \(d_h\) 时，位置间分数为

$$
S=\frac{QK^{\mathsf T}}{\sqrt{d_h}}\in\mathbb R^{B\times L\times L}.
$$

\(L\times L\) 表示“每个查询位置看每个键位置”，不是词表输出。因果 mask 禁止读取未来位置，softmax 在允许的位置上归一化，再对 value 加权。

## 导数、梯度和链式法则

导数是局部斜率。若 \(f(a)=a^2\)，则

$$
\frac{df}{da}=2a.
$$

对向量或矩阵的所有偏导组成梯度。最简单的梯度下降是

$$
\theta'=\theta-\eta\nabla_\theta\mathcal L.
$$

\(\theta\) 是跨训练 step 持久化的参数，\(\eta\) 是学习率。若 \(a=f(b)\)、\(b=g(c)\)，链式法则为

$$
\frac{da}{dc}=\frac{da}{db}\frac{db}{dc}.
$$

反向传播就是沿计算图反复使用链式法则。TTT 还会更新快权重 \(\phi\)，外循环可能继续对这次更新求导，因此会出现二阶路径。中心差分

$$
\frac{f(a+\varepsilon)-f(a-\varepsilon)}{2\varepsilon}
\approx\frac{df}{da}
$$

可以作为小例子的数值检查，但不是训练实现。

## 从 logits 到 softmax 和交叉熵

给定位置 \(t\) 的 logits \(z_{t,v}\)，softmax 是

$$
p_\theta(v\mid x_{\le t})
=\frac{\exp z_{t,v}}{\sum_{u=0}^{V-1}\exp z_{t,u}}.
$$

若真实下一个 token 是 \(y_t\)，负对数似然为

$$
\ell_t=-\log p_\theta(y_t\mid x_{\le t}).
$$

例如 \(z=(2,1,0)\) 且第二个类别正确时，概率约为 \(0.245\)，loss 约为 \(1.407\)。对 logits 求导：

$$
\frac{\partial\ell_t}{\partial z_{t,v}}
=p_{t,v}-\mathbf 1[v=y_t].
$$

正确类别的梯度为负，其他类别的梯度为正。共享参数会同时影响许多位置，所以真实网络不保证每个样本的正确 logit 单调增加。

若 \(m_t\in[0,1]\) 是 next-token loss mask，令 \(W=\sum_{t=1}^{L-1}m_t\)，则正权重窗口的平均 loss 为

$$
\mathcal L=\frac{\sum_{t=1}^{L-1}m_t\ell_t}{W}\qquad(W>0).
$$

\(W=0\) 时安全分母只负责避免除零，不代表存在监督信号。正的分数权重即使小于 1 也必须保留。

## 优化器与两个时间尺度

外循环可以写成

$$
\theta_{k+1}=\theta_k-\eta_{\mathrm{outer}}\nabla_\theta\mathcal L_k.
$$

AdamW 在这个基本方向上增加一阶、二阶移动统计和解耦权重衰减。TTT 的内循环则可能写成

$$
\phi_{c+1}
=\phi_c-\eta_{\mathrm{inner}}
\nabla_\phi\mathcal L_{\mathrm{inner},c}.
$$

\(k\) 表示长期训练 step，\(c\) 表示序列内的 chunk。内循环标签何时可用、尾块是否提交更新、外循环是否保留二阶路径，都是协议的一部分；两个学习率不能混为同一个参数。

## 数值稳定性和 dtype

softmax 通常先减去最大 logit：

$$
\operatorname{softmax}(z)_v
=\frac{\exp(z_v-\max_u z_u)}
{\sum_j\exp(z_j-\max_u z_u)}.
$$

这不会改变概率，却能降低指数溢出的风险。低精度节省显存，但会缩小数值范围；正式实验应分别记录参数、梯度、激活、快权重和归约使用的 dtype。遇到 NaN 或 Inf，应保留失败前的 loss、梯度、学习率、dtype 和 batch 信息。

## 阅读模型代码的顺序

1. 检查 \(X\) 的 ID 范围、\(B\) 和 \(L\)；
2. 检查 embedding 是否得到 \(B\times L\times d\)；
3. 检查 Q/K/V 和 causal mask 是否阻止未来信息；
4. 检查 residual、Norm、MLP 的输入输出宽度；
5. 检查 logits 是否为 \(B\times L\times V\)；
6. 检查 \(Y\) 的移位和 \(M\) 的监督位置；
7. 检查 loss、梯度和 optimizer state 谁在更新；
8. 若有 TTT，再追踪 \(\phi_c\) 在哪个 chunk 之后改变。

先读懂普通 dense 路径，再把 TTT 看成对其中一段状态和梯度协议的改变。这样才能判断一个新模块究竟改变了表示、上下文、状态还是优化目标。

## 小结

本章建立了全书最小的数学桥：shape 描述数据怎样流动，矩阵乘法描述表示怎样变换，softmax 和交叉熵定义预测目标，梯度和链式法则解释参数怎样改变，mask 定义哪些位置参与优化，dtype 说明数值边界。

下一章把这些概念变成实验契约：先写清楚要比较什么、哪些变量必须固定、什么证据足以支持结论。
