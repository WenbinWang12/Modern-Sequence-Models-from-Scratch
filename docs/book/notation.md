# 全书统一记号

## 本章导读

后面的章节会同时讨论数据、张量、概率、优化器和运行时状态。如果同一个字母在不同章节中表示不同东西，读者会很难判断公式和代码是否在描述同一个对象。本章先用一条很短的中文句子走完一次语言模型输入，再固定全书会反复使用的记号。

读完本章，你应该能完成三件事：把文本写成 token ID；把 token 流变成输入 \(X\)、目标 \(Y\) 和损失掩码 \(M\)；看到一个 logits 向量时，知道它如何变成概率和 loss。数学位置从 1 开始，代码数组索引从 0 开始，这是全书最重要的阅读约定。

## 一个贯穿例子：猫坐下

为了手算，我们假设 tokenizer 把“猫坐下”以及句末标记分成四个 token，并给出一组**教学用** ID：

~~~text
文本 token：猫   坐   下   <EOS>
token ID：  17   42   31     2
数学位置：  1    2    3      4
代码索引：  0    1    2      3
~~~

真实 tokenizer 的 ID 不一定是 17、42、31、2；ID 只是词表中的整数坐标。模型不会看到“猫”这个汉字，而会看到整数序列 \(s=(17,42,31,2)\)。这个例子中的 \(s\) 表示 tokenizer 产生的连续 token 流，后面还要从它构造模型输入。

数学位置 \(t=1\) 对应代码里的 <code>tokens[0]</code>，一般关系是：

$$
\text{数学位置 }t
\quad\longleftrightarrow\quad
\text{代码索引 }t-1.
$$

如果代码打印了 <code>tokens[2]</code>，它表示数学位置 \(t=3\)，不是位置 2。我们保留这种差异，因为数学推导用从 1 开始的求和更自然，而 Python、NumPy 和 JAX 都用从 0 开始的数组索引。

## 从 token 流到 \(X,Y,M\)

因果语言模型的任务是：给定当前位置以前看到的 token，预测下一个 token。对上面的四个 token，取前三个作为输入：

$$
X=(x_1,x_2,x_3)=(17,42,31),
\qquad
Y=(y_1,y_2,y_3)=(42,31,2).
$$

逐项看，位置 1 输入“猫”并预测“坐”，位置 2 输入“猫、坐”并预测“下”，位置 3 输入“猫、坐、下”并预测 EOS。代码中的数组就是：

~~~python
input_ids = [17, 42, 31]
labels = [42, 31, 2]
loss_mask = [1, 1, 1]
~~~

这里的 <code>input_ids</code> 对应数学 \(X\)，<code>labels</code> 对应 \(Y\)，<code>loss_mask</code> 对应 \(M\)。如果把两篇文档直接拼接，第二篇文档的第一个 token 是否允许被第一篇文档的末尾预测，要由 packing 规则决定。例如：

$$
M=(1,1,0)
$$

在这个三位置例子中屏蔽的是 EOS，不能把它称为跨文档监督。真正的跨文档例子需要把第二篇文档的首个 token 加进来。设拼接流为 \(s=(17,42,31,2,55)\)，其中 55 是下一篇文档的首个 token，则 \(X=(17,42,31,2)\)、\(Y=(42,31,2,55)\)。使用 \(M=(1,1,1,0)\) 才会排除“EOS 预测 55”的那一项。loss mask 只改变监督位置；它不会阻止之后的位置通过注意力读取前一篇文档。文档隔离还需要 attention mask 或状态重置，这会在 packing 章节展开。

## 索引、形状和 batch

全书使用以下基本符号：

| 对象 | 记号 | 约定 |
|---|---|---|
| batch 样本 | \(b\) | \(b\in\{1,\ldots,B\}\) |
| token 位置 | \(t\) | \(t\in\{1,\ldots,L\}\) |
| Transformer 层 | \(\ell\) | \(\ell\in\{0,\ldots,n_\ell-1\}\) |
| chunk | \(c\) | 连续 token 窗口的一段 |
| 外循环训练步 | \(k\) | 静态参数被优化器更新一次 |
| expert | \(e\) | \(e\in\{0,\ldots,n_{\mathrm{exp}}-1\}\)，与代码专家 ID 一致 |
| batch 大小 | \(B\) | 一个训练步中的样本数 |
| 序列长度 | \(L\) | 一个窗口中的 token 数 |
| 词表大小 | \(V\) | tokenizer 可以产生的 token ID 数量 |
| 隐藏宽度 | \(d\) | 残差向量的分量数，配置 <code>hidden_size</code> |
| MLP 中间宽度 | \(d_{\mathrm{ff}}\) | 配置 <code>intermediate_size</code> |
| query / KV 头数 | \(n_h,n_{\mathrm{kv}}\) | 配置 <code>num_attention_heads</code>、<code>num_key_value_heads</code> |
| 每个头的宽度 | \(d_h\) | 配置 <code>head_dim</code>；默认 \(d/n_h\) |
| 专家总数 / 选择数 | \(n_{\mathrm{exp}},k_{\mathrm{sel}}\) | 不用 embedding 矩阵的 \(E\) 表示专家数 |
| 字节数 | \(N_{\mathrm{bytes}}\) | 与 batch 大小 \(B\) 区分 |

一个 batch 的输入张量写成：

$$
X\in\{0,\ldots,V-1\}^{B\times L}.
$$

例如 \(B=1,L=3\) 时，<code>input_ids</code> 的形状是 <code>[1, 3]</code>；模型把每个 ID 查成 \(d\) 维向量后，hidden 的形状变成 <code>[1, 3, d]</code>；最后词表投影产生 <code>[1, 3, V]</code> 的 logits。除非特别说明，张量轴顺序都是 batch、sequence、feature。

## embedding、hidden、logits 和概率

下面五个对象需要区分：

1. **token ID** 是词表中的整数坐标，例如 17；
2. **embedding** 是查表得到的 \(d\) 维向量；
3. **hidden state** 是经过模型层变换后的 \(d\) 维残差流；
4. **logits** 是对 \(V\) 个候选 token 的未归一化分数；
5. **概率** 是把 logits 经过 softmax 后得到的总和为 1 的分布。

设 embedding 矩阵为 \(E\in\mathbb{R}^{V\times d}\)。token 17 的向量是 \(E_{17,:}\)。某个位置经过模型后得到 hidden \(h\in\mathbb{R}^{d}\)，输出头矩阵为 \(W_{\mathrm{out}}\in\mathbb{R}^{d\times V}\)，则：

$$
z=hW_{\mathrm{out}}\in\mathbb{R}^{V}.
$$

第 \(v\) 个分量 \(z_v\) 只是分数，不是概率。softmax 把它变成：

$$
p_\theta(v\mid x_{1:t})
=
\frac{\exp z_v}{\sum_{u=0}^{V-1}\exp z_u}.
$$

分母把所有候选的指数分数加起来，所以 \(p_\theta(v\mid x_{1:t})\) 在所有 \(v\) 上相加等于 1。词表大小 \(V\) 因而同时决定输出头的最后一维和概率分布的类别数。

## 损失、梯度和两个时间尺度

如果目标 token 是 \(y_t\)，这个位置的负对数似然为：

$$
\ell_t^{\mathrm{tok}}(\theta)
=
-\log p_\theta(y_t\mid x_{1:t}).
$$

有 \(m_t\in\{0,1\}\) 时，一个窗口的平均 loss 是：

$$
\mathcal{L}
=
\frac{\sum_{t=1}^{L}m_t\ell_t^{\mathrm{tok}}}
{\max(\sum_{t=1}^{L}m_t,1)}.
$$

为什么梯度会出现 \(p-\text{one-hot}\)？设 logits 向量为 \(z\)，正确类别为 \(y\)：

$$
\ell=-\log\frac{\exp z_y}{\sum_v\exp z_v}
=-z_y+\log\sum_v\exp z_v.
$$

对第 \(j\) 个分量求导，第一项贡献 \(-\mathbf{1}[j=y]\)，第二项贡献 softmax 概率 \(p_j\)，因此：

$$
\frac{\partial\ell}{\partial z_j}
=p_j-\mathbf{1}[j=y].
$$

如果直接把这组 logits 当作独立变量更新，正确类别的负梯度会提高其分数，其他类别的正梯度会降低其分数。真实训练更新的是共享参数 \(\theta\)，还要通过链式法则计算 \(\nabla_\theta\mathcal L\)；有限步长、其他样本和优化器状态使一次参数更新不保证每个位置的正确概率都上升。这个局部推导会在 baseline 和 TTT 内循环中再次使用。

静态参数跨训练 step 更新：

$$
\theta_{k+1}
=
\operatorname{Opt}_{\mathrm{outer}}
\left(\theta_k,\nabla_\theta\mathcal{L}_{\mathrm{outer}}\right).
$$

TTT 的快权重按 chunk 更新。以下是不考虑梯度裁剪时的一步 SGD：

$$
\phi_{c+1}
=
\phi_c-\eta_{\mathrm{inner}}
\nabla_\phi\mathcal{L}_{\mathrm{inner}}.
$$

\(k\) 表示外循环训练步，\(c\) 表示当前序列内的 chunk 更新编号，\(t\) 仍表示 token 位置；三者不能混为同一个时间轴。

## 数据集、分布和状态 owner

数据集 \(\mathcal{D}\) 是可以检查文件、manifest 和 hash 的对象；分布 \(D\) 是采样器实际让模型看到的概率分布。例如两篇文档分别有 100 和 900 个 token：先以各 \(1/2\) 的概率选文档、再在其中选一个 token，短文档的每个 token 被抽到的概率是 \(1/200\)，长文档的是 \(1/1800\)。若在合并后的 1000 个 token 中均匀抽样，每个 token 的概率都是 \(1/1000\)。数据没变，抽样规则改变了模型看到的分布。这个例子只抽一个 token；窗口抽样还需考虑长度和边界。

状态也必须有 owner 和 lifetime：

| 状态 | 记号 | owner | 生命周期 |
|---|---|---|---|
| 静态参数 | \(\theta\) | model/checkpoint | 多个 batch |
| 优化器状态 | \(O_k\) | trainer/checkpoint | 训练 step |
| sampler/RNG | \(R_k\) | data stream/checkpoint | 训练恢复 |
| KV cache | \(\mathbf c_t\) | request session | 一次推理请求 |
| TTT fast weights | \(\phi_c\) | chunk/session | 由协议决定 |

“模型状态”不是一个足够精确的词。只恢复参数而不恢复 optimizer 和 sampler，可能加载成功，却不能称为同一个训练过程的 resume。

## 代码检查与证据

记号本身不需要 GPU。远端 CPU 节点上可以先生成 tiny token 文件：

~~~bash
bash scripts/prepare_sample.sh outputs/sample-tokens.npy
.venv/bin/python -m ttt.data --input examples/sample.jsonl --output outputs/sample-shards --tokenizer byte --sharded --shard-tokens 256
~~~

成功时应看到 token 文件、分片文件和 <code>manifest.json</code>；检查 <code>vocab_size</code>、shard token 数和输入摘要是否存在。这个检查只能证明数据对象和字段能生成，不能证明模型已经学会语言，也不能证明 GPU 训练吞吐。

## 小结与下一问

本章固定了位置、形状、\(X/Y/M\)、embedding/hidden/logits/probability、loss/gradient，以及 \(\theta\) 和 \(\phi\) 的状态边界。下一章用这些记号解释：为什么要先写实验契约，怎样判断一次 tiny run 是否真的可比较。

## 书写规则

标量使用斜体字母，向量使用粗体小写，矩阵使用粗体大写，集合或损失使用花体。实际配置字段 <code>hidden_size</code>、<code>num_attention_heads</code>、<code>vocab_size</code> 分别对应 \(d\)、\(n_h\)、\(V\)。本书的教学数组名 <code>input_ids</code>、<code>labels</code> 对应 \(X\)、\(Y\)；实际 <code>sequence_loss</code> 接收未 shift 的 <code>tokens</code>，在函数内部构造输入和目标。不能把教学数组名误认成 loader 的返回字典字段。数学变量和概率使用 LaTeX，反引号只用于代码名称。
