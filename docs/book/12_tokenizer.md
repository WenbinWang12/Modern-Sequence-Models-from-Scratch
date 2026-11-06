# 12. Tokenizer 与词表

## 问题

Tokenizer 决定文本如何映射到 token ID；`vocab_size` 决定 embedding 和输出分类头的词表维度。词表是模型结构的一部分。

```text
embed_tokens: [vocab_size, hidden_size]
lm_head:      [hidden_size, vocab_size]
```

使用 tied embeddings 时，lm_head 共享 embedding 矩阵的转置；不共享时，分类头仍然必须输出 `vocab_size` 个 logits。

## 需要固定的内容

- tokenizer 文件和版本；
- vocab size；
- BOS、EOS、PAD、UNK 和 ignore token ID；
- 文档边界和 EOS 插入策略；
- Unicode 处理策略；
- tokenizer 训练语料和随机种子。

## 实验

先使用仓库的 byte tokenizer 跑通流程，再切换到目标 tokenizer。切换 tokenizer 后，旧 checkpoint 不能直接复用 embedding 和 lm_head，必须重新初始化或提供显式转换规则。

## 练习

统计同一批文本在 byte tokenizer 和目标 tokenizer 下的 token 数分布，比较平均长度、长尾和训练计算量。
