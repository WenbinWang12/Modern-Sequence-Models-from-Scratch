# 3　预训练数据：分布、质量与可复现性

本章使用[全书统一记号](notation.md)：数学位置从 1 开始，代码数组索引从 0 开始。

## 先定义两个核心概念

本章先回答两个容易被一句话带过的问题：什么是“训练数据分布”，什么是“预测损失”。只有定义清楚它们，后面的清洗、Tokenizer、packing 和 manifest 才能放在同一条技术链上。把 tokenizer 处理后的 token 序列记为 $x$，模型参数记为 $\theta$；后文所有公式都沿用这两个记号。

设原始数据空间中的一条记录为

$$
r_i=(u_i,m_i),
$$

其中 $u_i$ 是文本，$m_i$ 是来源、许可证、语言、时间和文档簇等元数据。用 $P_{\mathrm{raw}}$ 表示原始记录的分布；对于一组固定的原始文件，它可以是经验分布，例如

$$
P_{\mathrm{raw}}=\frac{1}{N}\sum_{i=1}^{N}\delta_{r_i},
$$

其中 $\delta$ 是集中在一条记录上的点质量。原始文件本身还不是最终的训练分布。我们需要经过来源选择、清洗、去重和切分，得到训练文档集合

$$
C_{\mathrm{train}}=\{r_1,\ldots,r_n\}.
$$

如果清洗、切分和来源混合构成变换 $C$，那么文档分布可以写成

$$
P_{\mathrm{doc}}=C_{\#}P_{\mathrm{raw}},
$$

这里的 $\#$ 表示把分布沿着变换推过去。对于固定的文档集合，如果采样器以权重 $w_i$ 选择文档，那么文档层面的经验分布是

$$
P_{\mathrm{doc}}(r_i)=
\frac{w_i}{\sum_{j=1}^{n}w_j}.
$$

均匀抽取文档、按文档长度抽取、按来源比例抽取，都会产生不同的 $P_{\mathrm{doc}}$。所以“训练集有多少文件”并不能单独说明模型实际看到了什么。

还要区分文档分布和训练位置分布。即使先均匀抽取文档，较长文档也可能在后续窗口中贡献更多 token；如果改为均匀抽取 token 起点，短文档、长文档和边界位置的权重又会不同。因此文档数、token 总数和真正参与 loss 的位置数不是同一个统计量。

Tokenizer $\tau$ 把一条文档的文本映射到 token 序列，并把文档分布进一步推到 token 序列分布：

$$
\tau(u_i)=z^{(i)}=(z^{(i)}_0,\ldots,z^{(i)}_{L_i}),
\qquad z^{(i)}_t\in\{0,\ldots,V-1\},
$$

其中 $V$ 是词表大小。采样器再按照 sequence length、packing、EOS 和文档边界规则，从这些 token 序列中构造训练窗口。记这个过程为 $\Pi$，它诱导出窗口分布：

$$
P_{\mathrm{tok}}=\tau_{\#}P_{\mathrm{doc}},
\qquad
P_{\mathrm{win}}=\Pi_{\#}P_{\mathrm{tok}}.
$$

$$
P_{\mathrm{win}}(X,Y,M)
=
\sum_i P_{\mathrm{doc}}(r_i)
\,P_{\Pi}(X,Y,M\mid \tau(u_i)).
$$

整条链可以压缩成：

$$
P_{\mathrm{raw}}
\xrightarrow{\,C\,}P_{\mathrm{doc}}
\xrightarrow{\,\tau\,}P_{\mathrm{tok}}
\xrightarrow{\,\Pi+\text{pack/shift/mask}\,}P_{\mathrm{win}}(X,Y,M)
\xrightarrow{\,p_\theta\,}\mathcal{L}.
$$

在实际项目中，$P_{\mathrm{win}}$ 才是更接近“训练数据分布”的对象。它不仅由原始文本决定，还由来源权重、清洗规则、去重规则、Tokenizer、packing、窗口长度和 mask 共同决定。

## 预测损失究竟是什么

设一个窗口中的连续 token 为

$$
z=(z_0,z_1,\ldots,z_L).
$$

因果语言模型把前面的 token 作为输入，把下一个 token 作为目标：

$$
X=(z_0,\ldots,z_{L-1}),
\qquad
Y=(z_1,\ldots,z_L).
$$

模型参数为 $\theta$ 时，第 $t$ 个位置的预测损失是观察到的目标 token 的负对数概率：

$$
\ell_t(\theta)=-\log p_\theta(y_t\mid x_{1:t}).
$$

如果模型在位置 $t$ 输出 logits 向量 $a_\theta(x_{1:t})$，那么

$$
p_\theta(v\mid x_{1:t})
=
\operatorname{softmax}(a_\theta(x_{1:t}))_v.
$$

如果 $M=(m_1,\ldots,m_L)$ 是 loss mask，$m_t=1$ 表示该位置参与训练，$m_t=0$ 表示忽略，那么一个窗口的平均预测损失为

$$
\ell(\theta;X,Y,M)
=
-\frac{\sum_{t=1}^{L}m_t\log p_\theta(y_t\mid x_{1:t})}
{\sum_{t=1}^{L}m_t}.
$$

模型真正优化的是这个损失在窗口分布上的期望：

$$
\mathcal{L}(\theta;P_{\mathrm{win}})
=
\mathbb{E}_{(X,Y,M)\sim P_{\mathrm{win}}}
\left[\ell(\theta;X,Y,M)\right].
$$

有限训练运行只能看到采样得到的 $N$ 个窗口。若按所有未被 mask 的 token 做平均，经验损失为：

$$
\widehat{\mathcal{L}}_N(\theta)
=
\frac{\sum_{k=1}^{N}\sum_{t=1}^{L_k}
m^{(k)}_t\ell^{(k)}_t(\theta)}
{\sum_{k=1}^{N}\sum_{t=1}^{L_k}m^{(k)}_t}.
$$

优化器通过 minibatch 梯度近似最小化这个经验目标。于是，“模型从数据中学习”可以更精确地表述为：模型参数被更新，使采样器产生的窗口上的平均 next-token 负对数似然下降。

验证集损失使用同一个定义，但把 $P_{\mathrm{win}}$ 换成独立的 $P_{\mathrm{val}}$。因此训练损失回答“模型是否适应了训练分布”，验证损失回答“模型在另一份声明过的分布上预测得如何”。如果验证集与训练集近重复，第二个问题就没有被可靠地测量。

这也解释了本章的核心判断：我们优化的不是一份抽象的“语言知识”，而是由数据处理和采样协议共同定义的预测目标。改变 $P_{\mathrm{doc}}$、$\tau$、$\Pi$ 或 $M$，即使模型结构和优化器不变，目标函数也会改变。

## 一份原始文档如何变成训练信号

数据管线不是把文件搬到 GPU，而是连续执行一组会改变数学对象的变换：

~~~mermaid
flowchart LR
    A[原始文件] --> B[文档记录与元数据]
    B --> C[清洗、去重、切分]
    C --> D[Tokenizer: text → token IDs]
    D --> E[Packing 与窗口采样]
    E --> F[input_ids / labels / loss_mask]
    F --> G[交叉熵与梯度]
    G --> H[参数更新]
~~~

用一组仅用于说明的 token ID 表示一条短文档：

| 对象 | 内容 |
|---|---|
| 连续 token | `[10, 11, 12, 2]` |
| `input_ids` | `[10, 11, 12]` |
| `labels` | `[11, 12, 2]` |
| `loss_mask` | `[1, 1, 1]` |

这一个窗口产生的损失是

$$
-\frac{1}{3}
\left[
\log p_\theta(11\mid 10)
+\log p_\theta(12\mid 10,11)
+\log p_\theta(2\mid 10,11,12)
\right].
$$

如果最后一个 EOS 位置被 mask 掉，分母和求和项都会改变；如果窗口跨越两篇文档，模型学习到的条件上下文也会改变；如果 tokenizer 把同一段文本拆成更多 token，预测位置数和成本也会改变。训练信号不是原始文本本身，而是经过这些规则构造出的 $(X,Y,M)$。

## 每一步改变了什么

**来源选择改变支持范围。** 网页、代码、书籍和论文拥有不同的词汇、长度和结构。模型只能在它实际被采样到的区域内获得训练信号。

**清洗改变样本集合。** 删除乱码、导航栏、截断页面和异常记录会改变哪些文档保留在 $C_{\mathrm{train}}$ 中。过滤过强又可能删除代码、数学、少数语言或长文档。

**去重改变样本权重。** 同一篇文档出现一百次，相当于提高它在 $P_{\mathrm{doc}}$ 中的概率。训练集和验证集近重复，还会让验证损失失去独立性。

**切分改变评估对象。** 去重和按文档簇切分决定验证窗口是否真正代表未见内容。评测集污染会让低损失或高分更接近记忆，而不是泛化。

**Tokenizer 改变样本空间和成本。** 同一段字符经过不同 tokenizer 会产生不同 token 序列。词表大小同时影响 embedding、LM head、序列长度、attention、KV cache 和吞吐。

**Packing 和 mask 改变条件概率。** 是否跨文档、是否保留 EOS、哪些位置参与 loss，都会改变 $P_{\mathrm{win}}$ 或窗口损失。不能把 packing 当成只影响存储的实现细节。

## 一个具体例子：文件大小不是训练分布

假设有三个来源：

| 来源 | 文档数 | 平均长度 | token 总量 |
|---|---:|---:|---:|
| 网页 | 900,000 | 1,000 | 900M |
| 代码 | 50,000 | 2,000 | 100M |
| 书籍 | 10,000 | 10,000 | 100M |

直接拼接会让网页约占 82% 的 token。均匀抽取文档则会提高书籍文档的采样概率。两种方案都可以合理，但它们定义的是不同的 $P_{\mathrm{doc}}$，因此也定义了不同的训练目标。

如果网页中 30% 是重复模板，清洗后网页有效 token 可能降到 630M；若不重新计算混合比例，代码和书籍的实际权重就会自动上升。每个阶段都必须报告自己的分母，不能只在最后写一个总 token 数。

当前仓库的 `BatchStream` 从一个逻辑 token 流中随机采样窗口，还没有实现按来源调度的 mixture sampler。后面的 mixture 章节会定义这种能力，但本章不把未来设计写成当前功能。

## 训练前的数据契约

进入长时间训练前，数据管线至少要回答五个问题：

1. **来源是否清楚？** 能否定位原始发布版本、获取时间和许可证？
2. **质量是否可解释？** 每条过滤规则是否有版本、阈值、统计和抽样审计？
3. **切分是否可信？** train、validation 和 test 是否按文档或项目边界隔离，并检查近重复和污染？
4. **表示是否一致？** tokenizer、特殊 token、packing、label shift 和 mask 是否与模型配置一致？
5. **版本是否可恢复？** manifest、分片 hash、采样器状态和 checkpoint 是否能确认下一批数据？

数据可以被程序读取，不代表它满足这些条件。每个条件都对应后续章节和一个可保存的证据对象。

## 数据对象和版本边界

**原始文件**是下载或采集得到的不可变输入，例如压缩包、HTML、JSONL 或代码仓库快照。

**文档记录**是抽取出的正文和元数据，支持清洗、去重、切分和文档边界管理。

**Token 分片**是 tokenizer 生成的整数序列和监督 mask，附带词表、tokenizer 版本、dtype、分片数量和 SHA256。

**训练窗口**是采样器从 token 流中取得的一段固定长度序列，进一步变成 `input_ids`、`labels`、`loss_mask` 和位置状态。

这四种对象不能共用一个含糊的“数据版本”。原始文件不变，清洗规则可以产生新的文档版本；文档不变，Tokenizer 可以产生新的 token 版本；分片不变，sequence length 和 packing 也可以产生新的窗口协议。

## 当前仓库已经实现什么

`ttt/data.py` 提供 byte tokenizer 和 JSONL 到 token 文件的转换。普通输入可以写成一维 NPY；sharded 路径会生成 token 分片、mask 分片和 manifest。

`ttt/dataset.py` 的 `TokenDataset` 会检查 manifest 版本、`vocab_size`、文件路径、SHA256、token 范围、mask shape 和 mask 数值，再用 memory mapping 读取分片。`BatchStream` 使用 NumPy Generator 随机选择窗口，并保存 RNG state、数据指纹、sequence length、batch size 和 seed。

~~~bash
.venv/bin/python -m ttt.data \
  --input examples/book/clean_corpus.jsonl \
  --output outputs/book-data \
  --sharded \
  --tokenizer byte \
  --shard-tokens 256
~~~

`shard-tokens` 是存储分片大小，不是模型一次处理的 sequence length。当前窗口可以跨越 EOS 和文档边界，模型没有通用的“遇到 EOS 就重置 attention”接口，因此应把现有 loader 描述为连续 token 流 packing，而不是文档隔离的 packing。

当前 manifest 也没有完整的 document-offset sidecar，所以任意 token 到原始文档的反查仍是大规模数据管线需要补齐的能力。把已实现行为和设计要求分开，是本书数据章节的基本写法。

## 后续章节如何推进

- `11_data_acquisition.md`：来源版本、许可证、下载和溯源；
- `12_data_cleaning.md`：规范化、格式、语言、长度和质量过滤；
- `13_deduplication.md`：exact/near dedup 以及重复对采样权重的影响；
- `14_split_and_contamination.md`：训练、验证、测试边界和污染检查；
- `15_tokenizer.md`：Tokenizer、词表和参数维度；
- `16_packing_and_masks.md`：packing、label shift、文档边界和 loss mask；
- `17_dataset_and_manifest.md`：分片、manifest、mmap 和可恢复采样；
- `18_data_mixture.md`：来源混合和确定性采样；
- `19_data_preflight.md`：训练前的硬门槛；
- `20_baseline_model.md`：用 dense Causal LM 检查这些输入是否真的支持学习。

从下一章开始，“数据”不再是一个笼统的输入名词。我们会逐步追踪每个对象如何产生、如何改变训练分布，以及如何留下能够被下一次运行复核的证据。
