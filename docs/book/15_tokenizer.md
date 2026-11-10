# 9　Tokenizer：从文本到固定维度的词表

## Tokenizer 不只是预处理脚本

模型接收的是整数序列，不是 Unicode 字符。Tokenizer 决定一段文字被切成多少个位置、哪些字符串拥有独立 ID、未知字符如何处理、EOS 如何表示。它因此同时影响上下文容量、embedding 参数、输出头参数、数据吞吐和评估公平性。

例如相同的 4,000 字符文本，如果 tokenizer A 产生 1,000 token、tokenizer B 产生 2,000 token，那么在相同的 sequence length 下，两个模型看到的字符上下文并不相同；在相同字符数据下，B 还会产生更多训练位置。比较模型时必须说明是按 token、字符还是文档对齐预算。

一段文本在进入模型前至少经过两步：先由 tokenizer 按固定规则切分，再把每个片段映射成词表中的整数 ID。记文本为 $u$，tokenizer 为 $\tau$，则可以写成

$$
\tau(u)=s=(s_1,\ldots,s_L),
\qquad s_t\in\{0,\ldots,V-1\}.
$$

这里的 token 不一定是完整单词，也可能是字节、字符或子词；ID 只是索引，数值大小规律不表示语义相似度。Tokenizer 因此同时定义了模型看到的序列长度、可表示的字符串集合和特殊 token 的边界语义。

## Tokenizer 和词表是什么关系

**词表**是 tokenizer 产生的一个组成部分。它记录哪些 token 可以使用，以及每个 token 对应哪个整数 ID。例如，词表可能包含：

~~~text
"low"  → 1523
"er"   → 4987
"<eos>" → 2
~~~

**Tokenizer** 是更完整的系统，至少包括：

- normalizer：如何处理 Unicode、大小写和空格；
- pre-tokenizer：如何确定初始切分边界；
- vocabulary：token 到 ID 的映射；
- segmentation rules：例如 BPE merge rules 或 Unigram 概率；
- decoder：如何从 token 恢复文本；
- special-token 处理规则。

因此，词表不是先独立训练好、再交给 tokenizer 使用的对象。训练 BPE 时，词表和 merge rules 通常一起学习；训练 Unigram 时，词表和子词概率一起产生。只有 Byte tokenizer 这类固定映射，才可以不经过统计训练，直接预先规定词表。

还要区分 tokenizer 训练和语言模型训练：

~~~text
文本语料
→ tokenizer 训练
→ vocabulary + 切分规则
→ 固定 tokenizer
→ token IDs
→ Transformer 训练 embedding 和模型参数
~~~

Transformer 训练的是 embedding、attention、MLP 等神经网络参数；它不会在预训练过程中自动改变词表。更换词表或切分规则，就等于改变了模型输入空间，通常需要重新生成 token 分片并重新初始化或转换 embedding。

## 词表大小怎样进入参数量

设词表为 \(V\)，隐藏维度为 \(d\)。输入 embedding 是 \(V\times d\)，参数量为 \(Vd\)。如果不共享输出权重，`lm_head` 还需要一个 \(d\times V\) 的矩阵；如果 tied embedding，则使用 embedding 的转置。

这里的 **tied embedding** 指输入和输出共享参数。输入时，token ID 通过

$$
E\in\mathbb{R}^{V\times d}
$$

查表得到隐藏向量；输出时，隐藏状态 \(h\in\mathbb{R}^{d}\) 直接使用同一个矩阵的转置计算词表 logits：

$$
z=hE^{\mathsf{T}}\in\mathbb{R}^{V}.
$$

如果不共享，则需要独立的输出矩阵 \(W_{\mathrm{out}}\in\mathbb{R}^{d\times V}\)，并计算 \(z=hW_{\mathrm{out}}\)。因此 tied embedding 少保存一个约 \(Vd\) 参数的矩阵，也让输入和输出使用同一个 token 表示空间；代价是输入表示和输出分类不能完全独立学习。

以 \(V=259\)、\(d=16\) 的 tiny 配置为例，embedding 只有 4,144 个参数；不共享时输出头再增加同样数量。若把词表扩大到 32,000，单个 embedding 就有 512,000 个参数；到 128,000 则是 2,048,000 个。对 768 维模型，词表从 32K 到 128K 会使 embedding 从约 2,460 万增加到约 9,830 万参数。

这解释了一个常见疑问：选定 tokenizer 后，MLP 后的分类头是否固定？MLP 的输出接口必须是 \(d\)，因为它要回到 residual stream；随后分类头把 \(d\) 映射到 \(V\)。因此最后 logits 的形状固定为 \([\ldots,V]\)。可以添加额外投影或改变 tied 方式，但那已经是新的结构和新的参数预算。

## 子词 tokenizer 是怎样训练出来的

Tokenizer 的训练不是训练 Transformer，也不是让模型通过反向传播学习参数。它通常是在一份代表性文本上统计片段，最后得到一套固定的词表、切分规则和 token ID。模型预训练开始以后，Tokenizer 通常保持不变。

训练前必须先固定：

- tokenizer 训练语料的 manifest 和版本；
- Unicode 规范化、空格处理和预分词规则；
- 目标词表大小 \(V\)；
- BPE、WordPiece 或 Unigram 等算法；
- BOS、EOS、PAD、UNK 等特殊 token 及其 ID；
- byte fallback、unknown 和截断策略。

以 BPE 为例，训练过程可以理解为：

1. 选择基础符号。它们可以是字符，也可以是 UTF-8 byte；
2. 统计训练语料中相邻符号对的频率；
3. 把最高频的符号对合并成一个新 token；
4. 用新的符号重新统计相邻组合；
5. 重复合并，直到达到目标词表大小。

例如，某些文本中经常出现相邻的 `l` 和 `o`，算法可能先学习 `lo`，之后再把 `lo` 和 `w` 合并成 `low`。训练结束后，Tokenizer 保存的是词表和合并顺序；编码新文本时，按照这套规则把文本切成 token。

WordPiece 和 BPE 都学习子词，但合并标准不同。WordPiece 通常使用基于似然的训练准则，许多实现编码时采用贪心的最长匹配；BERT 使用的就是这类方法。最长匹配是常见实现行为，不应当被当成 WordPiece 算法唯一的定义。

Unigram 则先建立较大的候选子词集合，再逐步删除贡献较小的候选。对同一段文本，它可以评估多种切分方式并选择概率较高的方案。SentencePiece 是支持 BPE 和 Unigram 的工具框架，而不是与它们完全并列的第三种合并算法。

Tokenizer 训练结束后，至少要保存：

- vocabulary 和 token-to-ID 映射；
- BPE merge rules 或 Unigram 概率；
- normalizer 和 pre-tokenizer 配置；
- special token 字符串与 ID；
- 训练语料 manifest、代码版本和输出摘要。

评估不能只看词表是否成功生成。还要在独立的代表性语料上测量 tokens/byte、tokens/document、中文和英文的长度差异、代码片段的切分、unknown 或 fallback 比例，以及 EOS 和其他特殊 token 的频率。Tokenizer 训练语料不应混入验证集或评测集，否则切分规则本身也可能带来污染。

## 实际如何训练一个 tokenizer

实际训练 tokenizer 时，不需要先选择一个 Transformer。训练对象是 BPE 的合并规则、Unigram 的子词概率或 WordPiece 的词表准则。Transformer 只在 tokenizer 固定以后，才使用这些 token 进行预训练。

### 准备训练语料

Tokenizer 训练语料应来自预训练数据的 `train` 部分，并使用与正式数据相同的 Unicode 规范化和文本抽取规则。若原始数据是 JSONL，先抽取 `text` 字段，生成每行一篇文档的纯文本文件；不要把 validation、test 或 benchmark 文本混入 tokenizer 训练。

训练前要写清楚：

- 语料 manifest 和版本；
- 文档边界以及换行、空格如何保留；
- normalizer 和 pre-tokenizer；
- BPE、Unigram 或 WordPiece；
- 目标词表大小和计数口径；
- special token 的字符串、ID 和添加位置。

### 用 SentencePiece 训练 BPE 或 Unigram

SentencePiece 可以直接从原始文本训练 BPE 或 Unigram。下面的命令是算法示例，需要额外安装 `sentencepiece` CLI；它不在仓库默认的 `tokenizer` extra 中。将 `--model_type=bpe` 改成 `unigram` 可以训练 Unigram：

~~~bash
spm_train \
  --input=data/tokenizer/train.txt \
  --model_prefix=outputs/tokenizer-32k \
  --model_type=bpe \
  --vocab_size=32000 \
  --character_coverage=1.0 \
  --pad_id=0 \
  --bos_id=1 \
  --eos_id=2 \
  --unk_id=3
~~~

命令会生成 tokenizer 模型和词表文件。正式记录还应保存完整命令、SentencePiece 版本、输入 manifest 的 hash 和输出文件摘要。对于多语言语料，`character_coverage`、空格处理和 Unicode 规则尤其需要单独报告。

### 用仓库脚本训练 Byte-level BPE

仓库现在提供了实际生成 tokenizer 文件的入口：[scripts/train_tokenizer.py](https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch/blob/main/scripts/train_tokenizer.py)。它读取纯文本或 JSONL，训练 BPE/Unigram，并通过 `PreTrainedTokenizerFast.save_pretrained` 保存可以被 `AutoTokenizer.from_pretrained` 加载的目录。

先安装 tokenizer 依赖：

~~~bash
uv sync --extra tokenizer
~~~

然后从 JSONL 的 `text` 字段训练一个 32K Byte-level BPE：

~~~bash
uv run python scripts/train_tokenizer.py \
  --input data/tokenizer/train.jsonl \
  --output outputs/tokenizer-32k \
  --algorithm bpe \
  --vocab-size 32000 \
  --text-key text
~~~

训练 Unigram 时改用 `--algorithm unigram`；如果不希望从 byte 开始，可以加入 `--no-byte-level`。输出目录包含 `tokenizer.json`、special-token 配置和 `tokenizer_metadata.json`，后者记录输入文件 hash、算法、词表大小和 special token。

脚本的主要代码路径是：读取 JSONL 并生成临时纯文本、构造 `models.BPE` 或 `models.Unigram`、创建对应 trainer、调用 `tokenizer.train(...)`，最后用 `PreTrainedTokenizerFast` 保存模型。这个过程生成的是 tokenizer 文件，不是 Transformer checkpoint。

### 直接使用 Hugging Face Tokenizers API

也可以用 `tokenizers` 库在 Python 中显式定义 normalizer、pre-tokenizer 和 trainer：

~~~python
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(
    vocab_size=32_000,
    special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
)
tokenizer.train(["data/tokenizer/train.txt"], trainer)
tokenizer.save("outputs/tokenizer-32k/tokenizer.json")
~~~

这段代码训练的是 Byte-level BPE。若更换到 Unigram，需要更换 model 和 trainer，同时重新记录 tokenizer 的配置；不能只修改文件名或 `vocab_size`。

### 训练完成后怎样接入模型

训练完成后先做独立检查：

~~~text
tokenizer files
→ encode/decode round trip
→ token ID 范围检查
→ special token 检查
→ tokens/byte 和长度分布
→ 生成 token shards 和 manifest
→ 设置模型 vocab_size
→ 训练 dense baseline
~~~

当前仓库的 `ttt.data` 负责加载训练好的 tokenizer 并生成 token 分片；subword tokenizer 的训练入口是 `scripts/train_tokenizer.py`。训练完成后，可以把输出目录传给 `ttt.data` 的 `--tokenizer`，再生成新的 token 分片。更换 tokenizer 后必须重新生成分片，不能让旧 token ID 直接沿用新词表。

### 这些步骤在仓库中的代码位置

- [`scripts/train_tokenizer.py`](https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch/blob/main/scripts/train_tokenizer.py)：从纯文本或 JSONL 语料训练 BPE/Unigram，并保存 Hugging Face tokenizer 目录；
- [`ByteTokenizer` 与 `HuggingFaceTokenizer`](https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch/blob/main/ttt/data.py#L14-L53)：定义 byte tokenizer、加载已有 Hugging Face tokenizer，以及统一的 `encode`/`decode` 接口；
- [`load_tokenizer` 与数据转换入口](https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch/blob/main/ttt/data.py#L51-L147)：根据配置选择 tokenizer，把 JSONL 或文本转换为 token 文件；
- [`prepare_sharded_data`](https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch/blob/main/ttt/dataset.py#L82-L168)：把 token 序列写成带 hash、词表大小和 mask 信息的分片 manifest；
- [`ExperimentConfig.tokenizer`](https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch/blob/main/ttt/config.py#L17-L24)：把 tokenizer 名称和模型词表约束接入训练配置；
- [`generate.py` 的词表检查](https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch/blob/main/ttt/generate.py#L25-L27)：防止 tokenizer 词表大于模型输出维度。

本节中的 SentencePiece 命令和 `tokenizers` Python 示例解释算法；仓库实际的生成入口是 [`scripts/train_tokenizer.py`](https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch/blob/main/scripts/train_tokenizer.py)。`ttt.data` 负责加载输出目录、检查 ID 并生成分片。

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

## 特殊 token 如何进入训练协议

特殊 token 不是普通词。若 tokenizer 训练或编码时自动在开头加入 BOS，数据脚本又手动加入一次，模型会看到重复标记；若 EOS 在文档间丢失，连续 packing 会让边界变得不可见。

因此，特殊 token 的字符串、ID、添加位置和是否参与 loss 都要写入 tokenizer 配置和数据 manifest。BOS、EOS、PAD、UNK 以及对话模板中的控制 token，都必须由 tokenizer、packing 和模型共同遵守。更换 tokenizer 后，不能只修改 `vocab_size`，还要重新检查这些 ID 与数据 mask 的关系。

## 分词效率要按来源统计

设原始文本的 UTF-8 字节数为 \(B\)，token 数为 \(T\)，可用 \(T/B\) 衡量编码密度。它不是质量分数，但能说明 sequence length 的实际覆盖。

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

当前仓库提供两个加载路径：`ttt.data` 中的 `ByteTokenizer`，以及通过 tokenizer extra 调用 Hugging Face 的 `HuggingFaceTokenizer`。后者使用 `AutoTokenizer.from_pretrained`，当前命令行只接收模型名称，没有暴露 revision 参数，也不会把远端 tokenizer 文件提交进 Git。因此正式实验必须在仓库外固定名称、revision、本地缓存摘要和文件哈希，并把这些信息写入实验记录。

数据准备入口是：

~~~bash
python -m ttt.data --input corpus.jsonl --output data/pretrain --sharded --tokenizer byte
python -m ttt.data --input corpus.jsonl --output data/pretrain --sharded --tokenizer meta-llama/Meta-Llama-3-8B
~~~

三条实际路径的边界如下：

| 路径 | BOS | EOS | loss mask |
|---|---|---|---|
| `prepare_data` 普通 NPY | `encode` 默认加入 | 每篇文档追加 | 没有 mask 文件，目标位置默认全为 1 |
| `prepare_sharded_data` 普通 LM | `encode` 默认加入 | 每篇文档追加 | BOS 为 0，其余 token 为 1 |
| `prepare_sharded_data --supervised` | prompt 加入，response 不重复加入 | response 后追加 | prompt 为 0，response 和 EOS 为 1 |

当前仓库的 `scripts/train_tokenizer.py` 已提供 BPE 和 Unigram 训练入口，暂不提供 WordPiece 训练入口。本章中的 SentencePiece 命令和 `tokenizers` API 用来解释算法，正式生成文件应优先使用仓库脚本。`prepare_data` 和 `prepare_sharded_data` 会在普通文本后追加 EOS，`ByteTokenizer.encode` 默认先加入 BOS，分片数据的普通语言模型路径会把 BOS 的监督 mask 置为 0；监督数据路径则只对 response 计算 loss。Hugging Face 路径调用 `encode(add_special_tokens=False)`，再按配置手动加入 BOS，不会自动加入 EOS。更换 tokenizer 前，必须重新检查这些行为是否符合新的模板和特殊 token 约定。

当前 manifest 会记录 tokenizer 名称、词表大小、输入文件摘要和分片摘要，但不会自动记录特殊 token ID、normalizer、远端 revision 或 tokenizer 文件哈希。这些字段属于生产复现要求，应由上层实验记录补齐。

训练脚本的 `tokenizer_metadata.json` 同时记录请求词表大小和实际词表大小。接入模型时必须使用实际值，并让它与模型配置中的 `vocab_size`、数据 manifest 中的 `vocab_size` 完全一致；只把配置文件里的数字改成新词表大小，不是可行的 checkpoint 转换。

更换 tokenizer 后必须同步配置中的 vocab_size 与特殊 ID，并重新生成所有 token 分片。旧分片中的整数仍然合法，不代表它们在新词表中具有相同含义。checkpoint 也不能只改一个 JSON 字段继续加载。

下一章从 token 流进入固定窗口，处理 label shift、EOS 边界和 loss mask。
