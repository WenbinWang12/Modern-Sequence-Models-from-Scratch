# 16. Packing、标签与 mask

## 从文档到训练样本

训练循环通常需要固定形状的 batch，例如 batch × sequence length。原始文档长度不一致，因此要决定如何截断、拼接、填充和构造 next-token labels。Packing 不只是节省 padding，它会改变文档边界和 loss 的有效位置。

一个样本可以表示为：

~~~text
input_ids: [x0, x1, x2, ..., xL-1]
labels:    [x1, x2, x3, ..., xL]
loss_mask: [ 1,  1,  1, ...,  1]
~~~

若序列中包含 padding 或文档分隔符，loss_mask 必须明确哪些位置参与优化。不能用 token id 是否为零来猜 mask，因为零可能是合法词表项。

## 两种 packing 语义

**独立文档 packing** 在每个文档之间加入 EOS，并在边界处屏蔽跨文档的监督。它更容易解释，但可能浪费尾部空间。

**连续流 packing** 把多个文档拼成一条 token 流，只使用一个 causal mask。它提高利用率，却需要明确 EOS、文档来源和评测切分，否则相邻文档可能形成不自然的上下文。

无论选择哪一种，都要把 sequence length、pad token、EOS、截断规则和 mask 版本写进 manifest。

## 代码入口

数据读取和分片位于 ttt/data.py、ttt/dataset.py；训练 loss 位于 ttt/loss.py。数据 loader 应在输出 batch 时同时返回 input_ids、labels、loss_mask、position_ids 和样本来源摘要，避免训练模块自行猜测边界。

## 形状和边界检查

每个 batch 应满足：

- input_ids 和 labels 的 shape 相同；
- token id 在 [0, vocab_size)；
- mask 是有限的 0/1 或明确的权重；
- 有效 token 数大于零；
- 最后一个 token 是否有 label 由 next-token 约定决定；
- padding 和 EOS 的处理在训练与评估中一致。

packing 错误通常不会立刻报 shape error，却会改变 loss。最小验证应打印一条样本的原文、token、labels、mask 和文档边界，确认人能读懂它。

## 下一章

packing 解决一个 batch 怎么长出来，dataset 和 manifest 解决训练如何按顺序、按版本读取这些 batch。下一章把分片、索引、采样状态和 checkpoint 连接起来。
