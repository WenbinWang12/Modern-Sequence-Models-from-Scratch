# 1　实验契约：让一次运行成为一个可以解释的实验

## 本章导读：先把“比较”写成一个问题

假设我们训练同一个 tiny 模型两次：第一次关闭 TTT 内循环，第二次打开它。如果第二次的 loss 更低，我们还不能立刻说“TTT 有效”。两次运行可能读了不同的 token、用了不同的学习率，甚至第二次从一个已经训练过的 checkpoint 开始。本章把“我想知道什么”翻译成一份机器可以检查的实验契约。

读完本章，你应该能完成一件具体的事：给一个 control/variant 对照填写研究变量、固定变量、预算和证据文件，并判断一次运行属于“可运行”“训练路径可更新、可恢复”还是“可解释”。本章只讨论实验身份；模型结构和数据处理分别在后面的章节定义。

在读配置表前，先区分几类数。**参数**是模型会学习的数，例如 embedding 矩阵的元素。**优化器**依据梯度修改这些数；AdamW 还保存历史梯度及其平方的移动平均，作为下次更新的参考，这些历史记录称为优化器状态。**dtype** 是数字的存储格式，例如 float32 的一个元素占 4 字节；改变它会影响数值精度和内存。**seed** 用来初始化伪随机数序列，运行到一半时的随机状态则说明已经走到了哪里。后面的恢复章会展示两者的区别。

## 贯穿例子：tiny-contract-001

后文使用 <code>tiny-contract-001</code> 作为示例。control 读取 <code>configs/book/tiny_control.json</code>，variant 读取 <code>configs/tiny.json</code>；两者都使用从 <code>examples/sample.txt</code> 生成的普通 NPY token 文件，运行 4 个训练 step，batch size 为 1，sequence length 为 16。两份配置只在 <code>inner.enabled</code> 上不同：control 为 <code>false</code>，variant 为 <code>true</code>。两者均保留快权重 MLP 的结构；没有这个模块的 dense 基线属于另一项结构对照。

| 字段 | control | variant |
|---|---|---|
| 实验身份 | <code>tiny-contract-001-C</code> | <code>tiny-contract-001-F</code> |
| 唯一改变 | <code>inner.enabled=false</code> | <code>inner.enabled=true</code> |
| 固定内容 | commit、tokenizer、数据、16-token window、4 steps、seed、dtype | 同左 |
| 自动记录指标 | <code>loss</code>、<code>grad_norm</code>、<code>tokens_per_second</code> | 同左 |
| 证据目录 | <code>outputs/tiny-contract-001-C/</code> | <code>outputs/tiny-contract-001-F/</code> |

这张表先于任何结果存在。它防止我们在看到结果以后才修改“比较条件”。当前 <code>tokens_per_second</code> 按 batch size × sequence length 计数，是输入位置吞吐；若数据含有被 mask 的目标，它不等于有效监督 token 吞吐。训练器没有单独输出 <code>valid_tokens</code>，正式实验需要另外统计 mask 的和。

## 先约定我们要证明什么

这本书不把“代码能够执行”当作研究结论。我们要回答的是：在相同的数据、训练预算和系统条件下，一个新的序列模块是否改变了模型保存、更新和使用上下文的方式，并且这种改变是否带来可重复的质量或效率收益。

因此，每次实验开始前必须写出一个具体主张：

> 在固定 ______ 的条件下，只改变 ______，观察 ______ 是否发生变化。

例如：在固定 tokenizer、有效训练 token、序列长度、优化器和硬件的条件下，只打开 TTT 的内循环，观察 held-out loss、快权重状态和训练成本是否变化。如果没有这句话，实验很容易同时改变多个因素，最后只能描述结果，不能解释原因。

## 实验的身份

一次运行不是一条命令，而是一个有身份的对象。可以把它写成：

$$
R=(S,C,I_{\mathrm{data}},E,K,Q_{\mathrm{budget}},O),
$$

其中：

- \(S\)：源码 commit；
- \(C\)：解析后的模型和训练配置；
- \(I_{\mathrm{data}}\)：数据、Tokenizer 和 manifest 的指纹；
- \(E\)：操作系统、Python、JAX、驱动和设备；
- \(K\)：随机种子、优化器和采样器状态；
- \(Q_{\mathrm{budget}}\)：训练 token、sequence length、batch、step 和成本预算；
- \(O\)：日志、checkpoint、评测和失败报告。

复现的目标不一定是让每个浮点数完全相同，而是让这些定义实验的对象一致，并把剩余的数值差异记录下来。

命令只是入口。两个运行即使命令相同，只要读取了不同的数据、从不同 checkpoint 开始，或使用了不同的默认配置，就不是同一个实验。

## 哪些变量必须固定

比较架构时，至少要固定或明确报告以下变量：

| 类别 | 内容 |
|---|---|
| 数据 | 来源、清洗版本、去重规则、train/validation manifest、有效 token 数 |
| 表示 | tokenizer 文件、词表大小、特殊 token、packing 和 mask 规则 |
| 模型 | hidden size、层数、参数量、层位置、初始化方式 |
| 算法 | loss、optimizer、学习率、warmup、梯度裁剪、内循环规则 |
| 预算 | sequence length、batch、训练 step、有效 token、总 FLOPs 或实测时间 |
| 系统 | GPU 型号、设备数、dtype、编译后端、并行和通信方式 |
| 状态 | seed、global step、optimizer state、采样器状态、cache 或 fast weights |

一次实验可以改变其中一项，但必须把改变写成研究变量。若同时改变架构、Tokenizer 和训练 token，就应把它称为新的配方，而不是架构消融。

一个常见的不可归因例子是：control 使用 byte tokenizer、sequence length 16，variant 换成子词 tokenizer、sequence length 32，同时 loss 下降。这个结果混合了词表、token 数量和上下文长度三个变化；它可以作为新配方记录，但不能支持“某个 TTT 组件造成下降”的结论。

## 三层完成标准

每次运行都按三个层次报告。

**可运行**：依赖、设备、配置和数据入口成立，程序可以启动并产生输出。

**训练路径可更新、可恢复**：loss 和梯度是有限值，训练状态能够更新，checkpoint 可以读取，恢复后的下一步与连续运行在预先定义的容忍范围内一致。这些是优化与状态检查；要证明模型从数据学到了规律，还需要受控训练与独立评估。

**可解释**：对照组和实验组只有声明的变量不同，评测使用独立且可信的 manifest，结果包含质量、成本和状态证据，能够支持一个边界清楚的结论。

第一层不是第二层，第二层也不是第三层。tiny 模型通过四步训练，只能说明接口和优化路径基本连通；它不能证明模型有语言能力，更不能证明某个架构优于另一个架构。

## 运行目录必须自己解释自己

正式实验还应保存源码身份、数据身份与结论。下面是建议的归档布局，其中 <code>source.json</code>、<code>data.json</code> 和 <code>report.md</code> 需要实验者或上游脚本创建，并非训练器自动输出：

~~~text
experiment.json   # 训练器保存的解析后配置和预算
source.json       # commit 和工作区状态
environment.json  # Python、包版本、设备、驱动和源码摘要
data.json         # manifest、tokenizer 和数据指纹
metrics.jsonl     # 每步 loss、梯度、速度和状态统计
status.json       # running、failed、complete 或 stopped
step_*.npz        # tiny 的模型、优化器和数据采样状态
report.md         # 面向读者的解释和结论边界
~~~

JSON、JSONL 和 checkpoint 是事实来源，Markdown 报告是阅读入口。报告中的数字必须能够回到运行目录中的原始记录；手工抄写的数字不能替代机器生成的证据。

对 <code>tiny-contract-001</code>，读者应能按下面的映射找到证据：

| 论断 | 首要证据 | 成功判据 |
|---|---|---|
| 运行使用了预期配置 | <code>experiment.json</code> | C 对应 <code>tiny_control.json</code>，F 对应 <code>tiny.json</code> |
| 设备与精度有记录 | <code>environment.json</code>、<code>experiment.json</code> | 前者有 <code>device_count</code>、<code>devices</code>，后者有模型 dtype 配置 |
| 模型确实更新 | checkpoint 参数比较，辅以 <code>metrics.jsonl</code> | 参数发生有限变化；非零梯度本身不保证学习率非零或参数已改变 |
| 运行可以恢复 | <code>latest.json</code>、checkpoint、<code>status.json</code> | resume 后 step 和下一批数据连续 |
| 结论可以比较 | <code>report.md</code>、control/variant 表 | 研究变量只有声明的一项 |

训练 checkpoint、服务 session snapshot 和实验报告不是同一种对象。训练 checkpoint 需要恢复模型、优化器、global step 和数据状态；session snapshot 需要恢复 KV、快权重、位置和请求状态；报告则解释这些状态支持什么结论。

## resume 不等于 initialize-from

`resume` 继续同一个优化过程，必须恢复模型参数、optimizer state、global step、采样器状态、数据指纹和学习率进度。

`initialize-from` 使用旧参数开启一个新实验，应该有新的运行身份；它可以重新建立 optimizer、step 和数据顺序。

如果只载入参数却把它称为 resume，后面的 loss 曲线和训练预算就无法解释。

## 失败也属于实验结果

失败可能发生在依赖安装、设备发现、XLA 编译、数据读取、梯度更新、checkpoint 保存或恢复阶段。失败目录至少保留命令、作业 ID、最后成功解析的配置、环境摘要、已有 metrics、异常阶段和是否可以安全恢复。

我们不把所有失败都归结为“随机性”。如果恢复后下一批数据不同，应先检查采样器状态和数据指纹；如果设备 OOM，应记录 batch、sequence length、dtype 和编译策略；如果 loss 为 NaN，应保留发生前后的梯度和参数统计。

失败记录的作用不是证明项目顺利，而是缩小下一次实验的未知范围。

## 结论的边界

实验报告最后必须分别回答：

1. 这次运行实际改变了什么？
2. 哪些指标发生了变化？
3. 这些变化能否归因于声明的变量？
4. 结果是否在其他 seed、数据或规模上复现？
5. 哪些更强的结论仍然没有证据？

可以使用下面的句式：

> 在固定的 tiny 配置和已声明控制变量下，只改变一个研究变量，我们观察到某个指标发生变化。该结果支持一个边界清楚的结论，但不能直接推广到尚未控制或验证的条件。

本书后面的每个章节都遵守这份契约。环境章节回答“在哪里运行”，数据与表示章节回答“模型实际看到了什么”，baseline 章节回答“系统是否真的会学习”，架构章节回答“状态如何改变”，实验章节回答“收益是否来自这个改变”。

## 本章小结

- 一个实验身份至少包含源码、解析配置、数据指纹、环境、随机与优化状态、预算和输出证据；
- control/variant 必须先写固定项和唯一改变，再启动训练；
- 可运行、可学习、可解释是三个不同门槛；
- checkpoint、session snapshot 和报告服务不同的恢复与解释目的；
- tiny-contract-001 的下一步是先在环境章节生成样例 token，并确认这份契约能被真实输出填满。
