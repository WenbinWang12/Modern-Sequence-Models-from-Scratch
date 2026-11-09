# 8　Tokenizer：从文本到固定维度的词表

[English](en/15_tokenizer.md)

## Tokenizer 不只是预处理脚本

模型接收的是整数序列，不是 Unicode 字符。Tokenizer 决定一段文字被切成多少个位置、哪些字符串拥有独立 ID、未知字符如何处理、EOS 如何表示。它因此同时影响上下文容量、embedding 参数、输出头参数、数据吞吐和评估公平性。

例如相同的 4,000 字符文本，如果 tokenizer A 产生 1,000 token、tokenizer B 产生 2,000 token，那么在相同的 sequence length 下，两个模型看到的字符上下文并不相同；在相同字符数据下，B 还会产生更多训练位置。比较模型时必须说明是按 token、字符还是文档对齐预算。

## 词表大小怎样进入参数量

设词表为 `V`，隐藏维度为 `d`。输入 embedding 是 `V × d`，参数量为 `Vd`。如果不共享输出权重，`lm_head` 还需要一个 `d × V` 的矩阵；如果 tied embedding，则使用 embedding 的转置。

以 `V=259`、`d=16` 的 tiny 配置为例，embedding 只有 `4,144` 个参数；不共享时输出头再增加同样数量。若把词表扩大到 32,000，单个 embedding 就有 512,000 个参数；到 128,000 则是 2,048,000 个。对 768 维模型，词表从 32K 到 128K 会使 embedding 从约 2,460 万增加到约 9,830 万参数。

这解释了一个常见疑问：选定 tokenizer 后，MLP 后的分类头是否固定？MLP 的输出接口必须是 `d`，因为它要回到 residual stream；随后分类头把 `d` 映射到 `V`。因此最后 logits 的形状固定为 `[..., V]`。可以添加额外投影或改变 tied 方式，但那已经是新的结构和新的参数预算。

## Byte tokenizer 是什么

当前仓库的 `ByteTokenizer` 把 UTF-8 字节值 0–255 映射到 ID 3–258，并保留 0、1、2 给 pad、BOS、EOS。因此：

~~~text
byte 0       → token 3
byte 255     → token 258
BOS          → 1
EOS          → 2
PAD          → 0
vocab_size   = 259
~~~

它的优点是任何 UTF-8 文本都能编码，不需要先下载词表，也没有未知字符。缺点是一个汉字通常占三个 UTF-8 字节，一个常见英文词也可能拆成多个位置，序列效率较低。

这个 tokenizer 适合验证接口和恢复逻辑，不适合作为正式中英混合基模的最终选择。正式 tokenizer 要在代表性语料上测量 tokens/byte、长度分布和特殊 token 使用，不能只比较词表大小。

## 训练一个子词 tokenizer 要固定什么

常见子词算法会从训练文本统计片段，再选出有限词表。无论使用 BPE、Unigram 还是其他算法，都要固定：

- tokenizer 训练语料的 manifest；
- Unicode 规范化和预分词规则；
- vocab size、min frequency 和 dropout；
- special token 的字符串、ID 和添加位置；
- byte fallback 或 unknown 处理；
- 训练代码、版本和输出文件摘要。

特殊 token 不是普通词。若训练时自动在开头加入 BOS，数据脚本又手动加入一次，模型会看到重复标记；若 EOS 在文档间丢失，连续 packing 会让边界变得不可见。每个特殊 token 的语义都要由数据和模型共同遵守。

## 分词效率要按来源统计

设原始文本的 UTF-8 字节数为 `B`，token 数为 `T`，可用 `T/B` 衡量编码密度。它不是质量分数，但能说明 sequence length 的实际覆盖。

至少统计：

| 统计 | 为什么需要 |
|---|---|
| tokens/byte | 估计上下文和数据成本 |
| tokens/document | 查看长短文分布 |
| fallback/unknown 比例 | 发现覆盖问题 |
| special token 频率 | 检查模板和边界 |
| 截断比例 | 判断 max length 是否过小 |
| 按语言/领域分组 | 防止总体均值掩盖偏差 |

一个 tokenizer 可能让英文更紧凑，却让中文或代码显著变长。总体均值不能替代分组统计。

## 当前代码和替换边界

`ttt.data` 提供 byte tokenizer，也可通过 tokenizer extra 调用 Hugging Face tokenizer。`HuggingFaceTokenizer` 使用 `AutoTokenizer.from_pretrained`，但仓库不会自动把远端 tokenizer 文件提交进 Git；正式实验应把名称、revision 和本地缓存摘要记录下来。

更换 tokenizer 后必须同步配置中的 vocab_size 与特殊 ID，并重新生成所有 token 分片。旧分片中的整数仍然合法，不代表它们在新词表中具有相同含义。checkpoint 也不能只改一个 JSON 字段继续加载。

下一章从 token 流进入固定窗口，处理 label shift、EOS 边界和 loss mask。
