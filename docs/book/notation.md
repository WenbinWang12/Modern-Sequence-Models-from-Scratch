# 全书统一记号

这本书同时讨论数据、模型、优化和运行时状态。如果同一个字母在不同章节中反复改变含义，公式看起来会比实现更复杂。因此从这里开始，正文、公式和实验报告使用同一套记号；代码字段名会在首次出现时与数学记号对应起来。

数学推导默认使用从 1 开始的位置编号，代码索引默认从 0 开始。两者描述的是同一个序列：数学中的 $x_t$ 对应代码中的 <code>tokens[t - 1]</code>。

## 1. 索引和形状

| 对象 | 记号 | 约定 |
|---|---|---|
| batch 样本 | $b$ | $b\in\{1,\ldots,B\}$ |
| token 位置 | $t$ | $t\in\{1,\ldots,L\}$ |
| Transformer 层 | $\ell$ | $\ell\in\{0,\ldots,n_\ell-1\}$ |
| chunk | $c$ | 一个序列窗口可以拆成多个连续 chunk |
| 外循环训练步 | $k$ | 优化器更新一次静态参数的编号 |
| expert | $e$ | $e\in\{1,\ldots,E\}$ |
| batch 大小 | $B$ | 一个训练步中的样本数 |
| 序列长度 | $L$ | 一个窗口中的 token 数 |
| 词表大小 | $V$ | tokenizer 可以产生的 token ID 数量 |

一个 batch 的 token 张量写成 $X\in\{0,\ldots,V-1\}^{B\times L}$。<code>[B, L]</code>、<code>[B, L, d]</code> 这样的顺序表示 batch、序列、特征；除非特别说明，本书不交换这三个轴。

## 2. 输入、目标和 mask

- $x_{b,t}$：第 $b$ 个样本在位置 $t$ 的输入 token ID；
- $y_{b,t}$：对应的 next-token 目标；
- $m_{b,t}\in\{0,1\}$：该目标是否计入 loss；
- $X,Y,M$：分别由 $x,y,m$ 组成的 batch 张量。

标准 causal language modeling 使用 $y_{b,t}=x_{b,t+1}$。最后一个输入位置是否有目标，由窗口构造和 padding 规则决定，不能只凭张量长度猜测。

## 3. 模型和表示

| 对象 | 记号 | 形状或含义 |
|---|---|---|
| hidden/residual width | $d$ | 残差流和归一化层的特征维度 |
| FFN 中间宽度 | $d_{\mathrm{ff}}$ | MLP 或 SwiGLU 的中间维度 |
| 层数 | $n_\ell$ | Transformer block 的数量 |
| attention 头数 | $n_h$ | query heads 的数量 |
| 单头宽度 | $d_h$ | 通常 $d_h=d/n_h$ |
| embedding | $E$ | $E\in\mathbb{R}^{V\times d}$ |
| hidden state | $\mathbf{h}^{(\ell)}_{b,t}$ | 第 $\ell$ 层位置 $t$ 的 $d$ 维向量 |
| hidden tensor | $\mathbf{H}^{(\ell)}$ | 形状为 $[B,L,d]$ |
| logits | $\mathbf{z}_{b,t}$ | $\mathbf{z}_{b,t}\in\mathbb{R}^{V}$ |
| 概率分布 | $p_\theta$ | 模型对下一个 token 的预测分布 |

模型前向函数统一写作 $f_\theta$。当 TTT 或其他测试时适应机制出现时，使用 $f_{\theta,\phi_t}$：$\theta$ 是跨 batch 学习的静态参数，$\phi_t$ 是当前序列内可能变化的快权重。

## 4. 损失和优化

- $\ell^{\mathrm{tok}}_{b,t}$：一个有效 token 的负对数似然；
- $\mathcal{L}_{\mathrm{batch}}$：按有效 token 数加权的 batch loss；
- $\mathcal{L}_{\mathrm{inner}}$：用来更新快权重的内循环 loss；
- $\mathcal{L}_{\mathrm{outer}}$：用来更新静态参数的外循环 loss；
- $\mathcal{L}_{\mathrm{val}}$：验证集上的汇总 loss；
- $\eta_{\mathrm{outer}}$、$\eta_{\mathrm{inner}}$：外循环和内循环学习率。

静态参数的更新写成：

$$
\theta_{k+1}
=
\operatorname{Opt}_{\mathrm{outer}}
\left(\theta_k,\nabla_\theta\mathcal{L}_{\mathrm{outer}}\right).
$$

快权重的更新写成：

$$
\phi_{t+1}
=
\phi_t-\eta_{\mathrm{inner}}
\nabla_\phi\mathcal{L}_{\mathrm{inner}}.
$$

这两个下标表达不同时间尺度：$k$ 跨训练步递增，$t$ 在一个序列或 chunk 内递增。

## 5. 数据、预算和实验

| 对象 | 记号 | 含义 |
|---|---|---|
| 数据集或 manifest | $\mathcal{D}_{\mathrm{train}}$, $\mathcal{D}_{\mathrm{val}}$, $\mathcal{D}_{\mathrm{test}}$ | 具体可读取的数据对象 |
| 训练数据分布 | $D$ | 采样器实际让模型看到的分布 |
| 已消费 token 数 | $N_{\mathrm{tok}}$ | 训练预算中的 token 计数 |
| 参数量 | $N_{\mathrm{param}}$ | 模型可训练参数总数 |
| 计算量 | $F$ | FLOPs 或声明过的计算量估计 |
| 墙钟时间 | $T_{\mathrm{wall}}$ | 从启动到完成的实际时间 |

数据集 $\mathcal{D}$ 和数据分布 $D$ 不等价：前者是可以校验 hash、manifest 和 split 的输入对象，后者是采样器在训练中形成的概率分布。数据章节讨论的是这两个层次如何联系。

## 6. 状态边界

- $\mathbf{c}_t$：attention 的 KV cache；
- $\mathbf{r}_t$：线性注意力或其他递归模块的状态；
- $\phi_t$：TTT fast weights；
- $\mathbf{S}_k$：checkpoint 中与第 $k$ 步相关的可恢复训练状态，包括 optimizer、sampler 和随机数状态。

参数、优化器状态、数据采样状态和推理 session 状态不能统称为“模型状态”。每一章都会说明某个状态属于静态参数、训练过程，还是单个请求。

## 7. 书中的书写规则

1. 标量使用斜体小写或大写字母，向量使用粗体小写，矩阵使用粗体大写，集合或损失使用花体；
2. <code>d_model</code>、<code>num_heads</code>、<code>vocab_size</code> 等代码字段分别对应 $d$、$n_h$、$V$；
3. <code>input_ids</code>、<code>labels</code>、<code>loss_mask</code> 分别对应 $X$、$Y$、$M$；
4. <code>theta</code>、<code>phi</code> 出现在代码或日志中时，正文公式仍写作 $\theta$、$\phi$；
5. $L$ 只表示序列长度，$\mathcal{L}$ 只表示损失，$B$ 只表示 batch 大小；
6. scaling 章节使用 $N_{\mathrm{param}}$ 和 $N_{\mathrm{tok}}$，不再用含义不明确的 $N$ 和 $D$；
7. 公式中的位置从 1 开始，代码中的数组索引从 0 开始，转换关系必须明确写出。
