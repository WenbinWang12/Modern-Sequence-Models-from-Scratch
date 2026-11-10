# Tokenizer 第二轮独立交叉复核

## 追加复验：当前处置状态

作者修正后，复核者重新读取了生产脚本、教学脚本、完整新增测试、两版相关段落。TK-R2-01–04 的实现或文字修正均已静态核对；执行证据仍待指定远端恢复。以下“静态处理完成”只关闭本表描述的缺陷，不代表测试通过或出版通过。

| 项目 | 当前证据 | 处置 |
|---|---|---|
| TK-R2-01：生产目录覆盖 | 训练前 `output_dir.exists()` 拒绝；创建目录使用 `exist_ok=False`；测试重跑并比较目录内原文件字节 | 静态处理完成，覆盖拒绝实测待跑 |
| TK-R2-02：教学报告覆盖 | 主函数早期拒绝同名路径，最终 `open("x")` 独占创建；测试保留预置旧报告并检查第二次调用拒绝 | 静态处理完成，测试待跑 |
| TK-R2-03：输入摘要时机 | 训练前记录源摘要及实际训练文本摘要，训练后分别检查未改变；metadata 保存两者；新增转换后修改源的失败测试 | 静态处理完成；消除了原报告指出的单向源修改后仍发布成功场景，不将前后检查称作不可变快照 |
| TK-R2-04：目标优化方向 | 两版改为 `mathcal J_tok`，明确最大化对数似然；最小化形式必须取负号 | 静态处理完成 |
| ID/rank 回归检查 | `catt` 明确断言 IDs `[7,5,0]` 与 ranks `[0,1]`；`catcap` 断言 ranks `[0,1,3,4]` | 断言与手算一致，待执行 |
| 跨章归一化 | 两版明确说明清洗 NFC 保留的兼容字形仍可能被 tokenizer NFKC 改变 | 静态处理完成 |

本次复验快照 SHA-256：

| 文件 | SHA-256 |
|---|---|
| 中文章 | `d1d68ab65df2f6e20c921bfe956fbccf7ae1e769b3a03f1289dd194e9eec272e` |
| 英文章 | `21e97db03f588acfc4acf9b6816cb03676a9a7f460e67fd7e5b14672eaa93840` |
| 生产训练脚本 | `8e7c722254ddfe8863eedcee105ac0309694bfd9ca81e5c0a03c288448382952` |
| BPE 教学脚本 | `169bf764b02ec35a3d0c7d603aeb8caa7278766b02cfbdf2cbe4fbfdfe5dc943` |
| 新测试 | `b88debda6255b308cd3e5110a7887d632cead7e8a966a64e1692fc0ea38cefda` |

本轮另补双语 `bpe-learning-vs-encoding{,-zh}.svg` 与同名 `.tex`，接为图 9.1。图用初始 token 单元、首轮完整计数、冻结 rank 表和 `catt` 的合并跨度解释训练与编码的不同。复核者是该图作者；图的数字与规则已交给分词章作者交叉核对，作者回报四词频次、八组 pair、五条 rank、停止条件和 `[7,5,0]` 均一致。此处只记录内容交叉核对，没有进行 SVG/XML 执行验证、TeX 编译或实际渲染；不标视觉通过。TikZ 使用相同数据与布局，是独立可编辑源，当前 SVG 不是已编译 TeX 的产物。

## 首次复核历史记录（保留修前意见与快照）

复核者：清洗章整改代理（非本章作者）。结论：数学小例与主要接口的静态核对有明显进展；仍有可复现产物保护和目标方向说明需要修正。本记录不构成执行通过或出版验收。

## 阅读范围与快照

实际阅读了两版第 15 章全文、全部数值表及公式、完整 `scripts/train_tokenizer.py`、`examples/book/bpe_walkthrough.py`、`tests/test_tokenizer_training.py` 和作者第二轮报告。为核对数据接入，还阅读 `ttt/data.py` 的 tokenizer 和普通数据路径，以及 `ttt/dataset.py` 的普通/监督分片分支。没有执行 Python、pytest、ruff 或构建。

审核时文件 SHA-256（使用静态文件摘要命令记录；并行作者后续修改须另记新快照）：

| 文件 | SHA-256 |
|---|---|
| 中文章 | `57699f338aeb77e35396f2b261e4e649c975c654afb2e5044ee502860f9083ba` |
| 英文章 | `afe81c150b2664c44d32aee30279beda74004b7111da59a668a5c31d569b6f2d` |
| 生产训练脚本 | `e5acb410d5b28fe0d4be669157ef4d2302071dc326f4990150b1ba5a0e64fe97` |
| BPE 教学脚本 | `d6781d9728de5c3aac7fef734c6050cd56e655b704c0e145cb7ace06cfef5f3d` |
| 新测试 | `50f6a6d8f372848bbf341aabc08705696801809aed33ae319ee3a99a05784205` |

## 需要修改的具体问题

| ID / 等级 | 位置 | 证据与影响 | 可行动修正 |
|---|---|---|---|
| TK-R2-01 / P1 | `scripts/train_tokenizer.py:99`；两版归档段 | `mkdir(exist_ok=True)` 后 `save_pretrained` 和 metadata 写入会覆盖已有词表。同一章强调保存完整 tokenizer 身份，但重复运行同一命令可能直接改变已有数据对应的词表文件。作者报告已承认沿用旧行为，不能因此视为风险已处理。 | 训练前拒绝已存在输出目录，创建时再用 `exist_ok=False` 防竞争；若保留覆盖需求，应有显式选项。添加旧产物字节不变的拒绝覆盖检查。 |
| TK-R2-02 / P1 | `examples/book/bpe_walkthrough.py:139–140` | 教学报告 `write_text` 无条件覆盖。脚本和输入摘要虽保存在 JSON 里，被第二次覆盖后第一次的实际证据仍会消失；与本轮其他教学脚本的新目录约定不一致。主代理亦独立指出此项。 | 对同名文件早期拒绝，并用独占写入 `open("x")`；检查第二次运行失败且原 JSON 字节不变。正文说明更换输出文件名。 |
| TK-R2-03 / P1 | `scripts/train_tokenizer.py:86–89,118` | 输入摘要在训练结束后才计算。JSONL 先被复制成临时文本；若源 JSONL 在之后改变，训练使用旧文本，metadata 却记录新输入摘要。纯文本也可能在训练与 hash 之间变化。因而当前摘要未必标识实际消费的版本。 | 训练前保存输入摘要；训练前后检查未变化，失败时不发布成功产物。更强的方案是对实际训练文本作不可变快照并同时记录其摘要及转换规则。至少加入模拟训练期间改源文件、确认不会产出成功 metadata 的测试。 |
| TK-R2-04 / P2（修正容易，应本轮完成） | 两版 L228–238，正号 `mathcal L_tok` | 公式是正的加权对数似然，却没有明确说最大化。全书通常用 `mathcal L` 表示最小化的 loss，零基础读者可能把方向理解反。公式本身与原论文方向一致，问题是符号和操作说明缺失。 | 明确写“最大化”，最好改为目标 `mathcal J_tok` 并给 `argmax`，或加负号并写最小化负对数似然；两版同步。 |

上述问题已经发给本章作者；本次复核未直接改动其独占文件。

## 按阅读单元独立核验

| 范围 | 独立核验结果 | 限定 |
|---|---|---|
| 两版 L3–33，引入与词表/算法区分 | 同一词表多种切分的反例成立；ID 仅为地址、预处理与 decoder 属于 tokenizer 身份均解释清楚；英文章未缩成摘要 | 目标读者是否理解仍需试读 |
| L35–85，初始频数表 | 手算 11 次词出现、47 个含词末标记的位置；八个 pair 次数逐项为 11/7/5/4/3/3/2/1，与公式一致 | 未运行脚本 |
| L87–126，后四轮与状态表 | 全部候选列表、五轮选择、同频 `cap` 先于 `s`、47→36→29→24→20→17 均可从四条词形手算得出 | 宽表桌面和手机效果未看 |
| 重叠 pair 反例 | `aaa` 有两处 `(a,a)`，一次不重叠替换只减少一位置；频次 2 时测试期待 pair_count=4、减少 2，成立 | 新测试未运行 |
| L128–177，rank 与教学 ID | 基础词表排序为 `</w>,a,c,p,s,t`；新增 `ca,cat,cat</w>,cap,cap</w>`；故 `catt`=[7,5,0]。`catcap` 依次用 rank 0/1/3/4，正确 | 测试目前核对 token，建议顺带显式断言正文给出的 ID 与 rank trace，以免文档与后续实现漂移 |
| WordPiece | 范围限定为 BERT 编码器；最长优先、后续 `##`、任一后续失败则整词 UNK 与原实现相符；未冒称实现完整训练器 | 完整 WordPiece 训练推导是明确未覆盖内容 |
| Unigram | 五项概率和 1；三种 cat 切分得 0.35/0.05/0.002，合计 0.402；条件化约 0.87065；最佳前缀递推和负无穷边界正确 | 需修目标最大化方向；不要求本轮把明确排除的 EM 全部补完 |
| ByteLevel 覆盖 | 固定版本源码中 alphabet 覆盖 256 字节表示；Unigram required_chars/finalize 会保留 initial_alphabet；生产两 trainer 都传入，decoder 也设置 | 静态 API/源码证据不等于库运行通过；min vocab 260 是容量下界，不保证实际大小，正文已限定 |
| NFKC roundtrip | 明确还原目标是 NFKC 原文，排除控制 token 解释和额外 cleanup；no-byte-level 分支不承诺空白/字节覆盖；测试以 ASCII 训练再验未见字符，方向合理 | 跨章建议加一句：上游 NFC 保留的圈号在这个 tokenizer 的 NFKC 下仍会丢失，避免把清洗规则当作端到端无损保证 |
| embedding / head | V×d、d×V、tied 转置与五组参数量均手算一致；MLP 中间宽度与最终分类维度分开解释 | 未核全部模型变体，不声称所有模型都采用本架构 |
| 保存与数据接入 | 实际 metadata 字段与文中对应；JSONL 转换保留内部换行；BOS/EOS/mask 表与本地两数据路径一致；两个 tokenizer 输出目录不同 | 保存保护问题见 TK-R2-01–03；训练语料路径明确为自备而非可复制首跑样例 |
| 双语对应 | 顺序、公式、数值、边界、命令一致；英文保留中文代码注释符合用户注释偏好 | 尚未检查网页中的数学、横向表格和语言切换 |

## 一手资料复核

- [Tokenizers v0.22.2 Unigram trainer 源码](https://raw.githubusercontent.com/huggingface/tokenizers/v0.22.2/tokenizers/src/models/unigram/trainer.rs)：核对 `required_chars` 和 `finalize` 对基础字符的保留。
- [Tokenizers v0.22.2 ByteLevel 源码](https://raw.githubusercontent.com/huggingface/tokenizers/v0.22.2/tokenizers/src/pre_tokenizers/byte_level.rs)：核对字节映射、256 项 alphabet 与 decoder。
- [BERT 原始 WordPiece 编码器](https://raw.githubusercontent.com/google-research/bert/master/tokenization.py)：核对最长匹配与整词未知项处理。
- [Kudo 2018 原论文](https://aclanthology.org/P18-1007.pdf)：核对片段乘积、候选切分条件化和应最大化的边缘对数似然方向。

## 远端与出版状态

主代理通知指定 CPU 主机 SSH 22 端口目前超时，因此本次没有任何远端成功日志。不能因为 `pytest` 文件已写好就说测试通过。依赖检查中使用了 `pytest.importorskip`：缺 tokenizer extras 时两项训练测试会跳过；未来验收必须同时记录 passed/skipped，确认 BPE 和 Unigram 两个参数化分支均实际执行。

修正后建议远端运行专属测试与教学命令，并保留 JSON、库版本、输入摘要和完整 pytest 结果。程序正确性、正文静态审校、网页视觉验收和读者试读分别记录；当前只完成本文所列静态核验。
