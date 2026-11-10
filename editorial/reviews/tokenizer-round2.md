# Tokenizer 第二轮整改记录（作者自查，待独立复审）

范围：`docs/book/15_tokenizer.md`、英文对应章、`scripts/train_tokenizer.py`、新教学脚本与专属测试。依据 `main-review.md` 的 T12、F22、F23、F24、F25 及 `chapter_template.md`。本记录不构成出版验收。

## 教学链与逐项整改

| 阅读单元 | 原问题 | 本轮改动与作者检查 |
|---|---|---|
| 引入与定义 | 先列名词，读者不知表示为何重要 | 从“猫坐下”与 cats 的不同位置数切入；定义原文、token、ID、序列长度、词表大小；解释 ID 无数值语义 |
| 词表与分词器 | 未说明同词表可产生不同切分 | 用 c/a/t/ca/cat 给三种切分；前后处理、规则、decoder 各有职责；归一化损失提前铺垫 |
| 两种训练 | 容易把训练词表当训练 Transformer | 分开解释 tokenizer 统计与神经网络参数学习，固定 tokenizer 后才编码模型数据 |
| BPE 初始状态 | T12 无可手算频次 | 原创 cat×5/cats×2/cap×3/caps×1；明确定义词形频次、不可拆词末符号、47个加权位置 |
| BPE 数学计数 | 无算法对应公式 | 加权相邻 pair 求和；解释指示函数及内外两层求和；11与7都有展开 |
| BPE 后续轮次 | 没有重算频次 | 第一轮8个pair全列；后4轮分别列全部候选；6行状态表列所有词形 |
| 同频与重叠 | 不明确确定性与计数边界 | 第5轮 cap/end 与 s/end 同频3，Unicode tuple顺序裁决；aaa说明pair频次不等于不重叠替换数 |
| BPE 编码 | 编码与训练混同 | 冻结5条merge rank；catt与catcap逐步；教学ID[7,5,0]按6个初始符号字典序及新增序核对 |
| 教学产物 | 只有运行无关的数据命令 | 标准库脚本保存全轮次、频次、词表、编码轨迹、脚本哈希、Python版本；正文称理论预期，不称已实测 |
| WordPiece | 泛指似然却未指定实现 | 限定BERT原始编码器的最长匹配与##行为；明确本章不实现完整WordPiece训练 |
| Unigram 概率 | 名词介绍无定义或目标 | 五项概率和1；三种cat切分得0.35/0.05/0.002，条件归一化0.402；给DP及边界、目标，限定不展开完整EM/剪枝 |
| 字节覆盖 | F22 ByteLevel被误认完整覆盖 | 两trainer均显式initial_alphabet=256；分开解释pre-tokenizer、alphabet、decoder；不混称<0xXX> fallback |
| 归一化 | F23无条件还原原文错误 | Ａ①→A1；只以NFKC后的普通文本为roundtrip目标；关闭自动special/cleanup；Whitespace路径不承诺相同性质 |
| 词表和输出维度 | 与模型预算联系不足 | embedding查表、d→V输出、tied关系；5组参数量；普通MLP不直接决定分类类别数 |
| 生产训练入口 | F25环境可能抹去其他extras | 沿用frozen/inexact和既有解释器；语料前提写明；保留JSONL内部换行的行为明确 |
| 元数据 | 算法专属参数和身份不明 | 不适用字段为null；额外记录normalizer/decoder/alphabet/特殊ID/库版本/tokenizer哈希；不保证跨实现bitwise重训 |
| 数据接入 | F24路径覆盖与特殊协议 | 两目录、输入前提、BOS/EOS/mask表、manifest实际缺项、tokenizer更换影响完整保留 |
| 双语 | 英文可能简化为摘要 | 两版同例同公式同五轮表同命令，章节结构一致；代码注释按用户要求保留中文 |

## 代码变更

`scripts/train_tokenizer.py`：

- Byte-level BPE 与 Unigram 都保留完整256字符映射。
- 请求词表不足260时先拒绝，不创建输出目录。
- 正式保存前不创建输出目录，已有输出目录明确拒绝；训练前后核对源文件与实际训练文本摘要，发现变化则不保存。
- 设置 `clean_up_tokenization_spaces=False`，防止解码清理掩盖空白差异。
- 元数据区分BPE专属min_frequency与Unigram专属max_piece_length，记录实际词表、特殊ID、库版本及文件摘要。
- JSONL非对象与非字符串text统一给出带行号错误。
- 独立复审后补保存源文件及转换后训练文本的独立摘要，避免后读取元数据冒充训练前输入。

`examples/book/bpe_walkthrough.py`：

- 自包含原始词频表，确定性tuple tie rule；从左到右不重叠merge。
- encode只查询固定rank；训练外字母表字符明确拒绝。
- JSON保存每轮全部pair计数与前后状态，可复算正文，未嵌入伪造“实测”结果。
- 教学范围为独立词、字符级BPE、显式词末标记，不替代ByteLevel正式实现。
- 独立复审后报告改为独占创建，拒绝覆盖旧证据。

`tests/test_tokenizer_training.py`：

- 独立手算期望，检查五轮频次与位置数；逆序词频输入检验tie独立于插入顺序。
- 检查新词编码不修改model；aaa重叠计数与真实替换差异。
- 两种算法以ASCII训练、未见中文/emoji/空白/NFKC输入核验无UNK与归一化roundtrip。
- 断言完整alphabet保留、保存再加载、摘要和专属参数元数据；不足词表拒绝且不留目录。
- 补tokenizer与BPE报告拒绝覆盖的文件不变断言，并模拟JSONL转换后源文件变化，要求拒绝保存。

清洗章作者进行了独立静态复审；按其意见将Unigram正号对数似然改记为J_tok，并明确训练最大化、作为最小化损失时应取负号。其余意见对应上面的覆盖保护和输入身份检查。独立审稿及远端结果以独立记录为准。

## 来源核对

- [Sennrich等，2016](https://aclanthology.org/P16-1162/)：字符子词BPE原始来源。正文数值例及推导为本章原创。
- [Kudo，2018](https://aclanthology.org/P18-1007/)：Unigram片段概率及候选词表训练的原始论文。
- [BERT原始编码器](https://github.com/google-research/bert/blob/master/tokenization.py)：WordPiece最长匹配、##与未知词行为。
- [Tokenizers v0.22.2 trainer API](https://raw.githubusercontent.com/huggingface/tokenizers/v0.22.2/bindings/python/py_src/tokenizers/trainers/__init__.pyi)：两trainer均支持initial_alphabet。
- [Unigram trainer源码](https://github.com/huggingface/tokenizers/blob/v0.22.2/tokenizers/src/models/unigram/trainer.rs)：required_chars包含initial_alphabet，finalize保留这些基础字符。

## 验证状态与剩余门槛

本地仅做文本编辑和静态 `git diff --check`，未执行Python、pytest、ruff、训练或构建。静态diff检查通过。代码与测试已交给主agent在 `ssh root@10.210.3.10` 执行；本记录撰写时尚未获得其运行结果。

建议远端命令：

~~~bash
.venv/bin/python examples/book/bpe_walkthrough.py --output outputs/book/bpe-walkthrough.json
.venv/bin/python -m pytest tests/test_tokenizer_training.py
.venv/bin/ruff check scripts/train_tokenizer.py examples/book/bpe_walkthrough.py tests/test_tokenizer_training.py
~~~

独立审阅人已对数学、五轮数值表、Unigram范围与双语进行了静态复核，并给出上面的四项修正。仍需对修订后的脚本获得远端证据，并检查Markdown/MathJax实际渲染及宽状态表横向溢出。

独立配图代理另补双语 `bpe-learning-vs-encoding{,-zh}.svg` 与同名可编辑TikZ源。我已静态核对四词频次、首轮八个pair、五条rank、catt合并跨度与[7,5,0]，它们和本章一致；未进行SVG渲染、TeX编译或美术验收。生产tokenizer质量、完整语料训练与跨平台逐字节重训不是小测试能够证明的结论。
