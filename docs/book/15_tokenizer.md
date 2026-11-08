# 15. Tokenizer 与词表

## Tokenizer 是模型接口

Tokenizer 把可变长度文本映射成有限的 token ID 序列。它决定序列长度、字符覆盖、未知片段的处理、特殊 token 的语义，也直接决定 embedding 和分类头的词表维度。

词表不是越大越好。更大的词表可能减少常见文本的 token 数，但会增加 embedding、lm_head 和输出 softmax 的参数与带宽；更小的词表减少参数，却可能让序列变长。选择应同时考虑数据语言、代码比例、目标上下文和硬件预算。

## 需要固定的内容

Tokenizer 版本至少包含：

- 训练语料的来源和 manifest；
- 算法、词表大小和规范化规则；
- 特殊 token 及其 ID；
- 字节 fallback 或 unknown 策略；
- 训练代码 commit 和输出文件摘要。

特殊 token 的 ID 一旦进入 checkpoint 和数据 shard，就不能在后续实验中随意重排。新增 token 应生成新的 tokenizer 版本，并重新生成受影响的 token 数据。

## 词表与模型形状

~~~text
token id ∈ [0, vocab_size)
embedding: [vocab_size, hidden_size]
hidden:    [batch, length, hidden_size]
logits:    [batch, length, vocab_size]
~~~

如果使用 tied embeddings，输出头共享 embedding 的转置；如果不共享，输出头仍然必须映射到相同的 vocab_size。MLP 的中间维度可以变化，但它的输出必须回到 hidden_size，才能接入残差和分类头。

## 统计 tokenizer 的实际效果

不要只记录词表大小，还要记录不同来源的平均 tokens/byte、长度分位数、特殊 token 频率、未知或 fallback 比例和最长样本的截断比例。中文、英文、代码和数学符号的 token 效率可能差异很大，混合语料尤其要分组统计。

## 下一章

token 序列还不能直接喂给训练循环。下一章讨论如何把文档拼成定长序列、构造 labels 和 loss mask，并说明跨文档边界如何避免错误监督。
