# 全书编辑与逐段审稿记录

当前版本仍是需要实质修订的书稿，**未通过出版验收**。本目录保存对标依据、固定版本的逐段意见和修订后复审记录；它不进入读者正文，也不把构建成功称为出版通过。

## 这次怎样对标公开书籍

读了 GitHub 上的书籍源码和具体教学章节，而不仅是 README 或目录。参考的是组织方法和证据要求，正文、例子、图与代码独立撰写，不复制参考书内容。

| 参考 | 具体观察位置 | 对本书的改进要求 |
|---|---|---|
| [Dive into Deep Learning](https://github.com/d2l-ai/d2l-en) | [导论](https://github.com/d2l-ai/d2l-en/blob/master/chapter_introduction/index.md)、[Softmax 回归](https://github.com/d2l-ai/d2l-en/blob/master/chapter_linear-classification/softmax-regression.md)、[文本序列处理](https://github.com/d2l-ai/d2l-en/blob/master/chapter_recurrent-neural-networks/text-sequence.md) | 从具体预测问题建立变量，给标量例子再写张量表达；文本处理让读者看见各阶段实际对象。我们的“术语→最终公式→命令”还缺中间教学步骤。 |
| [fastbook](https://github.com/fastai/fastbook) | [官方在线说明](https://course.fast.ai/Resources/book.html)与章节 notebooks | 可执行材料与解释一起阅读。我们的命令应连接到实际输入、输出和讨论，不能重复调用同一数据转换命令就称各章实践完整。 |
| [Build a Large Language Model From Scratch](https://github.com/rasbt/LLMs-from-scratch) | 官方仓库的章节与代码组织、纸质书对应说明 | 保持从组件到完整模型的连续依赖。普通语言模型基线需要先讲清，再引入适应状态；C/F 变体不能混称 dense。 |
| [Modern GPU Programming for ML Systems](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys) | [Background](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/main/chapter_background/index.md) | 用执行、存储和计算的关系组织解释，图需要有明确读图任务。此项是开放在线教材参考，不把它与正式出版书籍的出版状态混同。 |

我们不采用统一字数门槛，也不要求每章机械套十个标题。章节长度取决于完整讲清所需的概念和实验。新的约定见 [章节写作与审稿约定](../docs/book/chapter_template.md)。

## 审稿范围与版本

- [早期章节逐段审稿](reviews/main-review.md)：基线 <code>567fef3</code>，记号、序章、实验基础与数据部分，中英 36 页。36 项事实或实现问题、22 项教学或双语问题，附全部内容块定位。
- [高级章节审读与修订](reviews/advanced-review.md)：内外循环、缓存、MoE、实验、GPU 验证，中英 10 页；其中自审与独立复审必须区分。
- [前沿专题审读与修订](reviews/frontier-review.md)：前沿总览及三个专题，中英 8 页，分别核对论文、当前实现和待执行实验。
- 修订后的交叉意见保存在本目录的 <code>*recheck.md</code> 文件。旧快照的“问题”不因正文改动而删除；复审记录说明它是否真正解决。

以上范围不包括全部工程附录、导航、部署和最终印刷排版。报告中的代码检查主要是静态核对；未执行命令和未打开的图不能算通过。

## 已落实的修订方向

首先修正确性：纯文本误传 JSONL 分片入口、C/F 与 dense 配置混淆、虚构输出字段、错误恢复说明、概率和 mask 语义。新增真实 JSONL 教学输入，并将三种 tiny 配置明确分开。

随后补教学过程：序章从条件概率定义推导序列似然；数据分布区分文档抽样、连续流起点抽样、窗口平均和有效 token 平均；数据混合从单位、幂指数、预算、耗尽走到有限样本波动；验收用实际 loader 生成范围明确的报告。高级与前沿章追踪状态、标签到达和参数共享的真实实现。

这并不意味着全部教学问题已经关闭。BPE 的完整频次合并、MinHash 的最小手算、清洗的 Unicode 对照、数学背景的连贯铺垫、独立验证集上的完整训练评估链，以及逐图视觉审核仍需要持续修订。不得以新增字数或相同标题覆盖这些缺口。

## 目前不能声称通过的项目

| 项目 | 当前状态与所需证据 |
|---|---|
| 数学与实现一致性 | 已发现并修订多项错误，需独立复审新版本；每个关闭项要有位置和依据。 |
| 零基础教学 | 尚有 P1 缺口；按概念依赖补背景、例子、推导和实验，不能仅清除事实错误。 |
| CPU 命令执行 | 本轮连接用户指定的远端 CPU 节点超时，没有转到本地测试。 |
| GPU、恢复与性能 | 本轮未取得新的运行产物，不复用旧记录证明新命令。 |
| 图形与数学排版 | 需要实际渲染和逐图查看，静态语法检查不能代替。 |
| 双语质量 | 核对概念、数字、命令、假设与边界；不按中英字符长度比较。 |
| 目标读者试读 | 尚未完成。代理审稿只能发现一部分问题，不能替代真实初学者试读。 |

## 后续审稿如何保持可信

一轮修订结束时固定新快照，另一名审稿者从输入、推导、代码与结果逐段复核。代码路径发生变化，就更新相关命令和结果；只有文字变化，也要复查它是否改变条件或结论。每项状态使用“未审、已静态审读、待远端执行、已执行并附证据、待读者试读”等具体表述。

不能把没有发现新错误写成证明正确，也不能把一名代理的自审写成独立通过。书稿可以持续更新并供阅读，但发布网页与达到出版标准是两个不同状态。
