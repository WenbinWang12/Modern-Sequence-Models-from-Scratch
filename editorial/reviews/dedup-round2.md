# 去重章节第二轮整改记录

日期：2026-11-11。修改基线：`f544bc5163a61b4a67e4e11e6471868b32a56822`；本记录描述其后的未提交工作区修改。作者代理：`dedup_teaching_revision`。这是作者整改与静态核对记录，**不是独立审稿通过，更不是出版通过**。

## 范围与对应审稿意见

独占修改：

- `docs/book/13_deduplication.md`
- `docs/book/en/13_deduplication.md`
- `examples/book/dedup_walkthrough.py`
- `examples/book/dedup_documents.jsonl`
- `examples/book/dedup_checks.py`
- `docs/assets/figures/dedup-candidates.svg`
- `docs/assets/figures/dedup-candidates-zh.svg`

依据 `docs/book/chapter_template.md` 和 `editorial/reviews/main-review.md` 的 T10、T18，并继续维护 F21 的计数口径。没有改其他章节、导航、训练入口或共享工具，没有提交。

| 缺口 | 修改内容 | 证据位置／复审问题 |
|---|---|---|
| T10：exact key 没有落地代表映射 | NFC、仅 CRLF→LF；保留空白及原文；SHA 桶后完整比较文本；ID 排序代表规则；保存精确和最终映射 | 正文“比较什么”；`exact_deduplicate`；强制摘要碰撞与倒序输入检查 |
| T10：MinHash 只提名称 | 同一五文档 fixture、六个 bigram 元素、全部六对手算 Jaccard、四个明确排列和签名 | 正文集合表与排列表；`SHINGLE_LABELS`、`PERMUTATIONS` |
| T10：缺碰撞概率推导 | 非空集合、共享均匀随机排列；并集最小元素落在交集的当且仅当证明；期望与方差逐步解释 | 中英文“为什么签名相同”／“Why collision probability” |
| T10：LSH 条件未解释 | band 内 AND、band 间 OR；独立行与不相交 band；候选概率逐步推导，明确固定示例不是随机召回实测 | 双语分带表、公式、207/256 和 184/625 手算 |
| T10：候选≠重复／非传递只有口头描述 | 四个候选、两个达标对；A—B—C 的 3/4、3/4、1/2；连通分量过合并反例；只对已保留代表直接删除 | `select_representatives`；图中三阶段；正文删除与分组切分分离 |
| T18：实践仍是重复 tokenizer 命令 | 专用 stdlib 可运行脚本，JSONL fixture，保存每层中间结果与最终正文 | `dedup_walkthrough.py`、`report.json` 字段表 |
| F21：活动数不能加到去向数 | 文档守恒 5=3+1+1，另列四个复核对与两个达标对 | 正文 `counts`、脚本计数字段、检查用例 |
| 图只有装饰框 | 矢量签名矩阵、带样式区别的候选图、逐文档保留理由；同图能追踪 C 为何被保留 | 双语 SVG，SVG 本身为可编辑源；待实际网页和导出验收 |

## 逐节作者复核范围

已经静态阅读双语正文、公式、表格、图注和脚本，按如下依赖链改写。这里只记录作者检查范围，不把新增段落算成已被独立审稿人逐段批准。

1. 动机：等长文档的 9:1 副本概率仅限定在均匀文档采样；明确 token/窗口协议需要重新计算。
2. 精确表示：原始字节、比较文本、摘要、完整比较、代表原文分开；NFC 和 CRLF 范围明确，未使用 `strip()`。
3. shingle：说明空白切词与模型 tokenizer 无关；从实际文本移动宽度 2 的窗口；解释集合丢失次数与全文次序。
4. Jaccard：交／并／基数先解释；所有六对计算齐全；空集合分支拒绝，不默默给统一签名。
5. MinHash：`argmin` 返回元素；同一排列跨文档共享；四个固定排列只是可手算例，随机理论的均匀性另定义。
6. 概率：并集最早元素的等概率性、碰撞等价事件、交集概率求和；补上指示变量期望／方差的两取值计算。
7. 实用近似：seed 不等于独立均匀排列证明；保存算法、种子、版本，实用哈希与理想模型分开。
8. LSH：bucket 键包含 band 索引；候选对去重；独立假设和理论漏检概率明确；固定四排列不伪装实测。
9. 决策：候选、阈值边、文档删除三层；非传递用本例实际数值；ID 顺序策略和漏检限制明确；删除分组与切分组件不同。
10. 实践：无路径占位符，固定已提交输入；输出新目录；真实字段表；预期 JSON 明确标为推导值。
11. 生产边界：范围／粒度／误删漏检／资源；toy 常驻内存和穷举不冒充生产；大桶退化不承诺线性复杂度。

双语各节顺序、全部公式、输入、表格值、命令、输出字段和结果边界已做作者核对。英文正文完整呈现理论与实践，没有把中文浓缩成摘要。

## 来源核对

实际访问并阅读了：

- Lee et al., *Deduplicating Training Data Makes Language Models Better*：[作者论文页](https://arxiv.org/abs/2107.06499)。只用于动机，未借用效果数字作为本项目成绩。
- Moses Charikar, *Min-Wise Independent Permutations*：[原始 DEC 研究说明的存档](https://ftp.zx.net.nz/pub/archive/gatekeeper.dec.com/pub/DEC/SRC/technical-notes/SRC-1997-028-html/charikar.html)。核对 min-wise 的等概率最小值定义、精确／近似之分。
- Stanford CS336：[作者 lecture_14.py](https://github.com/stanford-cs336/lectures/blob/main/lecture_14.py) 及 [raw 文本](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)。核对 MinHash 的并集最早元素论证和 LSH 的 AND—OR 概率公式。未复制其运行断言或例子；本章五文档、排列、图和检查均为本章构造。

尝试打开 MMDS 作者站点 `ch3.pdf`、`ch3n.pdf` 及 v2.1 的章 PDF 时超时；Broder 1997 课程镜像返回访问失败。因此没有写成已通读这些 PDF。正文引用可实际访问的作者研究说明和课程源码。

## 远端执行协议与待核对结果

不得本地运行 Python 或测试。本轮作者只做文件编辑、只读源码／diff 核对和 SVG XML 语法解析，未运行本地 Python、单元检查、网站构建或图形渲染。根代理统一同步到用户指定 CPU 节点 `root@10.210.3.10`，在远端仓库根目录执行：

~~~bash
python3 examples/book/dedup_checks.py
python3 examples/book/dedup_walkthrough.py --output outputs/book-dedup-round2
~~~

输出目录须不存在。执行后保存终端日志以及 `report.json`、`deduped.jsonl`；不要把下面的手算预期写成日志：

- 7 个行为检查用例应通过。
- `input_documents=5`、`kept_documents=3`、`exact_removed_documents=1`、`near_removed_documents=1`。
- `candidate_pairs_reviewed=4`、`pairs_at_or_above_threshold=2`。
- 保留 `d01,d04,d05`；映射 `d02→d01`、`d03→d01`。
- 所有六对的 720 排列碰撞比例应分别等于精确 Jaccard；A、B 应碰撞 540 次。
- 强制所有摘要相同仍只精确删除 `d02`；倒转输入行序不改变精确代表。
- 图和中英文 Markdown 应通过远端站点严格构建、链接和数学渲染检查；移动端文字／标签／导出清晰度须看实际图，不能用 XML 解析替代。

报告生成时远端执行结果尚未由根代理回传；本记录不能把待执行改成通过。根代理已有运行材料。独立交叉审稿也待安排，尤其需要核对数学条件、排序策略解释和双语对应关系。SVG 已通过静态 XML 解析，尚未看实际网页截图，不声称视觉验收通过。

## 本轮结束时的状态补记

根代理随后独立审读去重正文与代码；清洗交叉审及修补记录见 [cleaning-round2-recheck.md](cleaning-round2-recheck.md)，统一运行器与切分输入边界的最后复核见 [runner-split-round2-recheck.md](runner-split-round2-recheck.md)。后者补充了JUnit跳过检查、明确失败状态、首尾来源摘要以及split的非有限元数据拒绝。它们不替代真实执行证据。

用户最新允许远端连接不上时先不测试，本轮不再尝试SSH；本作者始终没有运行本地Python或测试。运行与网页视觉验收继续标为待验。当前编辑冻结，由根代理统一提交。
