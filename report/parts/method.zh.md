## 2 方法

### 2.1 记号

记 token 序列为 \(x_1,\ldots,x_{L+1}\)，模型在位置 \(t\) 读入 \(x_{1:t}\) 并预测 \(x_{t+1}\)，\(L\) 是一个训练窗口里的预测位置数。模型有 \(n_\ell\) 个残差块，隐藏宽度为 \(d\)，MLP 中间宽度为 \(d_{\mathrm{ff}}\)，词表大小为 \(V\)。滑窗注意力只能看到最近 \(w\) 个位置。

参数分成两部分。**慢参数** \(\theta\) 在训练结束后固定，所有序列共用；**快权重** \(\phi\) 属于当前这一条序列，随着序列被读入而改变。序列按长度 \(C\) 切成更新块（chunk），第 \(c\) 块包含位置 \(\mathcal T_c=\{(c-1)C+1,\ldots,cC\}\)。\(\phi_{c-1}\) 表示预测第 \(c\) 块时使用的快权重，\(\phi_0\) 是它的初值。

### 2.2 模型结构

主干是一个 pre-norm 的 decoder-only Transformer。每个块依次做两件事：

$$
u=h+\operatorname{Attn}\bigl(\operatorname{RMSNorm}(h)\bigr),\qquad
h'=u+\operatorname{MLP}\bigl(\operatorname{RMSNorm}(u)\bigr).
$$

注意力使用旋转位置编码（RoPE，基数 \(5\times10^5\)）和 QK 归一化：query 和 key 在每个头内先做 RMSNorm 再做点积，分数除以 \(\sqrt{d_h}\)。MLP 是 SwiGLU，\(\operatorname{MLP}(v)=W_{\mathrm{down}}\bigl(\operatorname{SiLU}(W_{\mathrm{gate}}v)\odot W_{\mathrm{up}}v\bigr)\)，不带偏置。输入嵌入和输出投影共用同一个矩阵。

TTT-E2E 只改动最后 \(n_s\) 个块（下文称**后段**，其余 \(n_\ell-n_s\) 个块称**前段**）。后段的每个块在注意力和原有 MLP 之间多一个结构相同的 MLP′，它的三个矩阵就是快权重：

$$
u=h+\operatorname{Attn}(\operatorname{RMSNorm}(h)),\qquad
v=u+\operatorname{MLP}'_{\phi}(\operatorname{RMSNorm}(u)),\qquad
h'=v+\operatorname{MLP}(\operatorname{RMSNorm}(v)).
$$

原有的 MLP 仍然是慢参数，为预训练得到的知识保留一个不会被测试时更新改写的位置；这一点和只更新最后四分之一的块，都沿用原论文的设计[^ttte2e]。嵌入、归一化和注意力在测试时都不更新。

<figure markdown="span">
  ![TTT-E2E 的前向过程与两个循环](assets/report/fig_method.light.svg#only-light){ width="100%" }
  ![TTT-E2E 的前向过程与两个循环](assets/report/fig_method.dark.svg#only-dark){ width="100%" }
  <figcaption>图 1　TTT-E2E 的前向过程。前段对整条序列只算一次；后段逐块前向，每块先用当前快权重预测，再用这一块的损失把快权重更新一步。外循环把各块损失相加，并对整条更新链求导。</figcaption>
</figure>

### 2.3 内循环：先预测，后更新

前段不依赖快权重，对整条序列只计算一次，得到特征 \(g_t=f^{\mathrm{pre}}_\theta(x_{1:t})\)。后段逐块计算。读完第 \(c\) 块后，这一块的平均损失是

$$
\ell_c(\theta,\phi_{c-1})=\frac1C\sum_{t\in\mathcal T_c}-\log p\bigl(x_{t+1}\mid x_{1:t};\theta,\phi_{c-1}\bigr).
$$

注意预测第 \(c\) 块时用的是 \(\phi_{c-1}\)：它只见过前 \(c-1\) 块的目标。等第 \(c\) 块的目标全部成为已观测信息之后，才对快权重做一步梯度下降：

$$
g_c=\nabla_{\phi}\,\ell_c(\theta,\phi)\Big|_{\phi=\phi_{c-1}},\qquad
\hat g_c=\min\!\Bigl(1,\frac{\tau}{\lVert g_c\rVert_2}\Bigr)g_c,\qquad
\phi_c=\phi_{c-1}-\eta\,\hat g_c .
$$

\(\eta\) 是内循环学习率，\(\tau\) 是对全部快权重合在一起计算的梯度范数上限。更新后的 \(\phi_c\) 只用于第 \(c+1\) 块及之后的预测，所以任何一个 token 的损失都不会用到见过它自己的权重，评测时不存在标签泄漏。序列末尾不足一块的部分只参与评分，不触发更新。

后段的注意力带 KV 缓存：第 \(c\) 块可以看到窗口内前面各块的 key 和 value，它们是当时用旧的快权重算出来的，更新后不重算。

### 2.4 外循环：学习怎样被更新

训练目标是“按上述方式边读边更新之后”的平均损失：

$$
\mathcal L(\theta)=\frac{1}{n_c}\sum_{c=1}^{n_c}\ell_c\bigl(\theta,\phi_{c-1}(\theta)\bigr),\qquad \phi_0\subset\theta .
$$

\(\phi_0\) 是慢参数的一部分，也就是说快权重的初值是学出来的。因为 \(\phi_{c-1}\) 是 \(\theta\) 的函数，\(\mathcal L\) 对 \(\theta\) 的梯度除了普通的反向传播，还要沿着更新链回传：

$$
\frac{d\phi_c}{d\theta}=\frac{d\phi_{c-1}}{d\theta}-\eta\,\frac{d\hat g_c}{d\theta},
$$

其中 \(d\hat g_c/d\theta\) 含有 \(\ell_c\) 对快权重的二阶导。实现上就是对整个带内循环的前向函数再求一次导（梯度的梯度）。慢参数因此学到两件事：怎样预测下一个 token，以及怎样让后段在被梯度更新之后预测得更好。

消融实验里的**一阶近似**把 \(\hat g_c\) 当作常数，即令 \(d\hat g_c/d\theta=0\)。此时前向数值完全不变，训练更便宜，但外循环不再知道慢参数会怎样影响更新量。

### 2.5 推理

推理和训练时的内循环是同一个过程，只是不再有外循环。一个会话的状态由三部分组成：当前快权重 \(\phi\)（大小固定，与已读长度无关）、各层最近 \(w\) 个位置的 KV 缓存、以及尚未凑满一块的特征和目标。生成时每凑满 \(C\) 个 token 并观测到下一个 token，就执行一次更新。发布的慢参数不会被任何会话改动，不同会话之间也不共享快权重。

### 2.6 实现要点

实现基于 JAX，下面几点和结果的可信度直接相关。

- **外循环梯度。** 序列损失函数内部用 `lax.scan` 逐块前向并在每一步调用 `jax.value_and_grad` 得到 \(g_c\)，外层再对整个函数求导。块级扫描分成两层并加检查点，反传时只保存约 \(\sqrt{n_c}\) 份快权重快照，其余在反传时重算。
- **滑窗注意力。** 训练时前段对整条序列做滑窗注意力。我们把序列按 \(w\) 切块，每块只和自己以及前一块的 key 做稠密注意力，各块并行计算。在 float64 下它和逐位置掩码的参考实现最大相差 \(7\times10^{-7}\)（前向和对 query、key 的梯度都检查过），因此是同一个函数的另一种算法，不是近似。
- **交叉熵。** 词表投影和 log-softmax 按 token 分块计算；数值与一次性计算相同。
- **精度。** 参数为 float32；矩阵乘法用 bfloat16；归一化、softmax 和损失用 float32。
- **正确性测试。** 仓库的测试覆盖因果性（修改未来 token 不影响过去的输出）、外循环梯度与有限差分的一致性、流式推理与整段前向的一致性、检查点恢复，以及一阶近似“前向不变、梯度改变”这一性质；它们在 CPU 上全部通过。

### 2.7 复杂度

设每个 token 经过一个块的线性层成本为 \(c_{\mathrm{lin}}\)。全注意力每个 token 的注意力成本与已读长度成正比，KV 缓存也随长度线性增长。滑窗注意力和 TTT-E2E 的注意力成本都只与 \(w\) 有关。TTT-E2E 在此之上每个 token 多出两项：后段多一个 MLP′ 的前向，以及每块一次的后段反向传播（成本约为后段前向的两倍，再加输出层）。这两项都与已读长度无关，所以 TTT-E2E 的推理成本和状态大小对上下文长度是常数；代价是训练时要穿过内循环求二阶导。实测数字见第 4.10 节。
