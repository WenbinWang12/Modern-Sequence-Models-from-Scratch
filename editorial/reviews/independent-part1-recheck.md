# Part I 第二轮独立交叉复审

日期：2026-11-11。审稿者：<code>rewrite_advanced_chapters_parity</code>，不是这些 Part I 正文的作者。对象为本轮工作区中 <code>01_experiment_contract</code>、<code>01_environment</code>、<code>02_reproducible_runs</code>、<code>03_configuration</code>、<code>04_forward_and_ttt</code> 的中英文 10 页。正文保持只读，本文件只记录审稿意见。

结论：多项实现错误已修复，**不批准出版通过**。本轮静态阅读全部正文、公式、表格、代码与命令，核对对应实现；未运行本地或远端测试，未审网页渲染及 SVG 本体。根代理在收到初审意见后修订了 03/04，本记录保留初次发现并单列复核结果，不倒填“最初无问题”。

## 初次发现及已复核修复

| ID | 初次定位与问题 | 修复后的证据 | 当前判断 |
|---|---|---|---|
| C01 | 03/04两版，shape walk把批量 [B,L,d] 当直接代码对象；04表头原为“代码对象 / Code object” | 根代理改为教学批量布局，明确forward输入[L]、hidden[L,d]、logits[L,V]，training.py的vmap增加batch；attention.py以hidden.shape[0]计序列位置 | 正文静态问题关闭；直接调用/批处理执行未验 |
| C02 | 03两版导读承诺每head形状，但正文仅查表与输出，GQA/SwiGLU只有名字 | 新节计算 d_h=8，Q[L,2,8]、K/V[L,1,8]，说明共享KV；SwiGLU写两路16→32投影、SiLU门控、32→16，与attention.py/layers.py一致 | 本次具体缺项关闭；初学者理解仍需教学审阅 |
| C03 | 03英文“四类字段”含并不存在的inner steps配置；C_k/F_k与全书外循环k冲突 | 根代理替换为真实字段和C/F，不再引入歧义下标 | 静态问题关闭 |
| C04 | 旧审稿F05：C用tiny.json实际启用内循环 | configs/book/tiny_control.json与tiny.json其余字段一致，control的inner.enabled=false；环境及恢复命令引用control | 静态问题关闭；运行待验 |
| C05 | 旧审稿F09/F10：把采样当顺序索引，四步作业靠人工中断 | 02中英均从实际step2检查点恢复到新目录；BatchStream从seed推进两次，再load_state_dict比较下一batch；比step3 loss、最终TrainState和data_state；字段/路径和实现一致 | 实验设计与代码静态可接受，未执行，不能称恢复已成功 |
| C06 | 旧审稿F11/F14/F15：router suffix限制、尾块更新、rematerialization含义错误 | 03区分router与mlp；04明确尾块整块不提交更新、jax.checkpoint不截梯度；对照config/inner_loop | 这些静态错误关闭 |
| C07 | 旧审稿F06/F07/F13：错误字段、错误解释器、验证脚本越界宣称 | 两版环境使用.venv/bin/python；列真实status/metrics字段；04列validate.py实际case字段，不再称脚本做恢复 | 这些具体错误关闭；尚无运行证据 |

## 初次发现的正文问题（最新状态见增量复核）

下表保留第一次提出的问题和定位，不能把它当成当前仍开放的问题清单。R01–R05 已在本轮增量中复核关闭；R06包含审稿者对导航顺序的误判，下面明确更正；R07和运行、视觉证据仍未验收。

| ID | 优先级 | 文件、小节与首句 | 问题及可执行修改 |
|---|---|---|---|
| R01 | P0 实现表述 | en/04_forward_and_ttt.md，“The order implemented…”首句“If the configuration contains fast_mlp” | config键是use_fast_mlp，fast_mlp是layer参数字典的键；中文写“若参数中包含fast_mlp”正确。英文改为“If the layer parameter tree contains fast_mlp”；若要概括router适应，也说明只有adaptation_target=mlp时该分支使用传入fast_weights。 |
| R02 | P1 证据与术语 | 01_experiment_contract两版，“三层完成标准 / Three levels of evidence”，**可学习 / Learnable** | 有限梯度、参数改变、恢复一致只证明优化/状态路径连通；随机或方向错误的更新也可能满足。建议该层改为“训练路径与恢复正确性”，或增加受控小数据loss下降/独立评估证据后才谈学习。保留后文“不证明语言能力”的限制，但不要让层名先过度承诺。 |
| R03 | P1 零基础前置知识 | 01_experiment_contract贯穿表与变量表；01_environment系统/Python层；02“resume必须恢复AdamW一阶二阶统计” | dtype、优化器状态、动量/二阶矩在前几章作为审计要求出现，却没有解释读者如何识别这些对象。建议首次出现给出简短定义：dtype决定数字表示精度/字节；参数是被学习的数；优化器另存历史梯度统计；seed不等于运行到一半的RNG状态（02已有后者例子）。不必在环境章推导整个AdamW，但需要让“恢复哪一份历史”有直观意义。 |
| R04 | P1 双语内容覆盖 | zh/02“当前checkpoint保存什么”“如何比较两个结果”；en/02相应位置 | 英文缺少中文对训练fast初值/临时序列状态/session snapshot的整段辨析，以及定义/环境/数值差异三分法。英文01 summary提到三种对象，却缺中文01对应解释段。按语义补齐，章节顺序可合理合并，不要求逐字翻译。 |
| R05 | P1 API边界 | en/03“How the code consumes the contract”，AutoConfig.from_pretrained(tiny_control.json)示例 | AutoConfig只返回model配置，丢弃experiment.inner/training；mode=train仅选择启用重算的模型类，不意味着control的inner.enabled=false被模型对象携带。示例本身能构造模型，但应明确它仅演示架构注册，控制内循环仍用load_experiment得到的inner传给训练/序列损失。否则读者会把文件名control当作适应开关已经生效。中文若增加此例同步说明。 |
| R06 | P2 组织与过渡；含已撤回判断 | zh/03“用一个tiny shape走完整条路径”末句“下一章沿着这张shape图…”；两版summary“下一章计算logits、交叉熵和梯度” | 初审正确指出中途转场后仍有head/SwiGLU及配置审计；但错误地按文件名顺序推断03之后是04，并据此要求改写章末CE/梯度承诺。核对mkdocs.yml后撤回该部分：实际顺序为03配置（14章）→20基线（15章）→04前向（16章），基线章确实讲CE与梯度。根代理已删除中途提前的“下一章前向”句，章末正确承诺应保留。 |
| R07 | P1 图证据 | 04两版图3及“prefix/suffix”节 | 正文现已区分层轴与时间轴，但沿用图仍需独立视觉检查是否能同时展示这两个轴、x5到达才完成C=4标签；本代理未打开SVG，不批准图示已解决教学缺口。需要后续逐图审稿，不能由文字修改推断美术完成。 |

优先级口径：P0为明确实现/事实错误；P1为妨碍新手理解或证据解释的必要教学缺口；P2为组织、命名与可读性。P1不是“已经可出版后再润色”。

## 最后一轮增量复核

本轮只读取指定修订及导航配置，只修改本审稿文件，没有修改正文。以下判断逐项针对原问题，不将修复几处段落扩大为全书出版结论。

| ID | 本轮直接核对的位置与修订 | 复核结果 |
|---|---|---|
| R01 | en/04“The order implemented in this repository”现写“If the layer's parameter tree contains fast_mlp”，并明确对应架构开关是use_fast_mlp | 原先把参数树键当作配置字段的具体错误关闭；与modeling_ttt_e2e.py的分支检查一致 |
| R02 | 01中英导读、证据层次和小结均改为“训练路径可更新、可恢复 / Updatable and recoverable”，正文另说学习规律需要受控训练和独立评估 | 证据层名过度承诺问题关闭，不再由有限梯度和恢复一致推断已经学会任务 |
| R03 | 01两版在配置表前定义参数、优化器保存梯度及平方的移动平均、float32每元素4字节，以及seed与运行中随机状态的区别 | 本次要求的首次定义缺项关闭；完整优化算法推导与零基础读者实测仍属于全书后续验收 |
| R04 | 01英文补训练checkpoint、session snapshot、报告三类对象；02英文新增“Parameters and temporary fast state”以及“Which differences change the experiment?” | 原来缺少的fast初值/序列临时态/session区别，以及定义/环境/数值差异三分法已补齐，具体双语语义覆盖缺项关闭 |
| R05 | 03中英AutoConfig节明确只返回model，不携带inner/training；mode=train只选模型类；完整实验通过load_experiment读取并向训练或sequence-loss传inner | API边界问题关闭；文件名control不再被暗示为模型对象已经携带禁用更新的设置 |
| R06 | mkdocs.yml中文导航L102–105、英文导航L140–142依次为03配置、20基线、04前向；20_baseline_model开篇确有logits→softmax→NLL内容 | 撤回初审对章末“下一章CE/梯度”的错误质疑；它符合真实导航。中途提前宣布进入前向章的句子已删除，剩余组织问题关闭 |
| R07 | 本轮没有打开SVG或渲染网页 | 保持待验，不以正文修订或本次静态核对代替视觉验收 |

此次增量关闭的是R01–R05的指定缺项，并更正和关闭R06；此前C01–C07结论保持。正文中未再发现与这些修订直接相关的新硬错误。远端执行、网页数学渲染、图示美术和真实初学者阅读仍没有新增证据。

## 需要远端运行才能关闭的事项

1. 在指定CPU节点按02的连续/step2恢复流程执行，保存recovery-check.json以及原日志、commit、环境。当前代码块仅静态核对，不能发布为成功实验。
2. 在GPU节点运行04实际tiny验证，保存case和阈值；默认64/512不证明32K/128K。
3. 检查两种语言的数学渲染、代码行换行与移动端表格，以及04 SVG本体。文件中有公式不等于网页已正确渲染。
4. 至少让一位未参与实现的零基础读者按章节操作，记录第一次遇到未定义名词的位置。本次工程交叉审不能替代真实读者审阅。

## 逐块阅读索引

以下保留首次交叉审读快照的非空正文块，未用后续文字倒填原快照。表内“开放R01/R02”等是当时状态，当前状态以上面的“最后一轮增量复核”为准。定位使用文件、小节、行号与首句；修订后行号可能变化，应以小节和首句为主。每个代码块作为整体审读，块内空行不另计段落。索引生成只是覆盖记录；没有把自动遍历当成出版分数。

### docs/book/01_experiment_contract.md

| 单元 | 位置与小节 | 类型 | 首句/首式 | 本轮处理 |
|---|---|---|---|---|
| 001 | L5 · 本章导读：先把“比较”写成一个问题 | 段落 | 假设我们训练同一个 tiny 模型两次：第一次关闭 TTT 内循环，第二次打开它。如果第二次的 loss 更低，我们还不能立刻说“TTT 有效”。两次运行可能读了不同的 token、用了不同 | 本轮未新增硬错误；不代表出版通过 |
| 002 | L7 · 本章导读：先把“比较”写成一个问题 | 段落 | 读完本章，你应该能完成一件具体的事：给一个 control/variant 对照填写研究变量、固定变量、预算和证据文件，并判断一次运行属于“可运行”“可学习”还是“可解释”。本章只讨论实验身 | 本轮未新增硬错误；不代表出版通过 |
| 003 | L11 · 贯穿例子：tiny-contract-001 | 段落 | 后文使用 tiny-contract-001 作为示例。control 读取 configs/book/tiny_control.json，variant 读取 configs/tiny.j | 本轮未新增硬错误；不代表出版通过 |
| 004 | L13 · 贯穿例子：tiny-contract-001 | 表格 | ／ 字段 ／ control ／ variant ／ | 本轮未新增硬错误；不代表出版通过 |
| 005 | L21 · 贯穿例子：tiny-contract-001 | 段落 | 这张表先于任何结果存在。它防止我们在看到结果以后才修改“比较条件”。当前 tokens_per_second 按 batch size × sequence length 计数，是输入位置吞 | 本轮未新增硬错误；不代表出版通过 |
| 006 | L25 · 先约定我们要证明什么 | 段落 | 这本书不把“代码能够执行”当作研究结论。我们要回答的是：在相同的数据、训练预算和系统条件下，一个新的序列模块是否改变了模型保存、更新和使用上下文的方式，并且这种改变是否带来可重复的质量或效率 | 本轮未新增硬错误；不代表出版通过 |
| 007 | L27 · 先约定我们要证明什么 | 段落 | 因此，每次实验开始前必须写出一个具体主张： | 本轮未新增硬错误；不代表出版通过 |
| 008 | L29 · 先约定我们要证明什么 | 段落 | > 在固定 ______ 的条件下，只改变 ______，观察 ______ 是否发生变化。 | 本轮未新增硬错误；不代表出版通过 |
| 009 | L31 · 先约定我们要证明什么 | 段落 | 例如：在固定 tokenizer、有效训练 token、序列长度、优化器和硬件的条件下，只打开 TTT 的内循环，观察 held-out loss、快权重状态和训练成本是否变化。如果没有这句 | 本轮未新增硬错误；不代表出版通过 |
| 010 | L35 · 实验的身份 | 段落 | 一次运行不是一条命令，而是一个有身份的对象。可以把它写成： | 本轮未新增硬错误；不代表出版通过 |
| 011 | L37 · 实验的身份 | 公式 | R=(S,C,I_{\mathrm{data}},E,K,Q_{\mathrm{budget}},O), | 本轮未新增硬错误；不代表出版通过 |
| 012 | L41 · 实验的身份 | 段落 | 其中： | 本轮未新增硬错误；不代表出版通过 |
| 013 | L43 · 实验的身份 | 段落 | - \(S\)：源码 commit； | 本轮未新增硬错误；不代表出版通过 |
| 014 | L51 · 实验的身份 | 段落 | 复现的目标不一定是让每个浮点数完全相同，而是让这些定义实验的对象一致，并把剩余的数值差异记录下来。 | 本轮未新增硬错误；不代表出版通过 |
| 015 | L53 · 实验的身份 | 段落 | 命令只是入口。两个运行即使命令相同，只要读取了不同的数据、从不同 checkpoint 开始，或使用了不同的默认配置，就不是同一个实验。 | 本轮未新增硬错误；不代表出版通过 |
| 016 | L57 · 哪些变量必须固定 | 段落 | 比较架构时，至少要固定或明确报告以下变量： | 本轮未新增硬错误；不代表出版通过 |
| 017 | L59 · 哪些变量必须固定 | 表格 | ／ 类别 ／ 内容 ／ | 本轮未新增硬错误；不代表出版通过 |
| 018 | L69 · 哪些变量必须固定 | 段落 | 一次实验可以改变其中一项，但必须把改变写成研究变量。若同时改变架构、Tokenizer 和训练 token，就应把它称为新的配方，而不是架构消融。 | 本轮未新增硬错误；不代表出版通过 |
| 019 | L71 · 哪些变量必须固定 | 段落 | 一个常见的不可归因例子是：control 使用 byte tokenizer、sequence length 16，variant 换成子词 tokenizer、sequence lengt | 本轮未新增硬错误；不代表出版通过 |
| 020 | L75 · 三层完成标准 | 段落 | 每次运行都按三个层次报告。 | 本轮未新增硬错误；不代表出版通过 |
| 021 | L77 · 三层完成标准 | 段落 | **可运行**：依赖、设备、配置和数据入口成立，程序可以启动并产生输出。 | 本轮未新增硬错误；不代表出版通过 |
| 022 | L79 · 三层完成标准 | 段落 | **可学习**：loss 和梯度是有限值，训练状态能够更新，checkpoint 可以读取，恢复后的下一步与连续运行一致或在预先定义的容忍范围内一致。 | 开放 R02：此证据不足以命名为学习 |
| 023 | L81 · 三层完成标准 | 段落 | **可解释**：对照组和实验组只有声明的变量不同，评测使用独立且可信的 manifest，结果包含质量、成本和状态证据，能够支持一个边界清楚的结论。 | 本轮未新增硬错误；不代表出版通过 |
| 024 | L83 · 三层完成标准 | 段落 | 第一层不是第二层，第二层也不是第三层。tiny 模型通过四步训练，只能说明接口和优化路径基本连通；它不能证明模型有语言能力，更不能证明某个架构优于另一个架构。 | 本轮未新增硬错误；不代表出版通过 |
| 025 | L87 · 运行目录必须自己解释自己 | 段落 | 正式实验还应保存源码身份、数据身份与结论。下面是建议的归档布局，其中 source.json、data.json 和 report.md 需要实验者或上游脚本创建，并非训练器自动输出： | 本轮未新增硬错误；不代表出版通过 |
| 026 | L89 · 运行目录必须自己解释自己 | 代码/示意 | experiment.json   # 训练器保存的解析后配置和预算 | 本轮未新增硬错误；不代表出版通过 |
| 027 | L100 · 运行目录必须自己解释自己 | 段落 | JSON、JSONL 和 checkpoint 是事实来源，Markdown 报告是阅读入口。报告中的数字必须能够回到运行目录中的原始记录；手工抄写的数字不能替代机器生成的证据。 | 本轮未新增硬错误；不代表出版通过 |
| 028 | L102 · 运行目录必须自己解释自己 | 段落 | 对 tiny-contract-001，读者应能按下面的映射找到证据： | 本轮未新增硬错误；不代表出版通过 |
| 029 | L104 · 运行目录必须自己解释自己 | 表格 | ／ 论断 ／ 首要证据 ／ 成功判据 ／ | 本轮未新增硬错误；不代表出版通过 |
| 030 | L112 · 运行目录必须自己解释自己 | 段落 | 训练 checkpoint、服务 session snapshot 和实验报告不是同一种对象。训练 checkpoint 需要恢复模型、优化器、global step 和数据状态；sessi | 本轮未新增硬错误；不代表出版通过 |
| 031 | L116 · resume 不等于 initialize-from | 段落 | 'resume' 继续同一个优化过程，必须恢复模型参数、optimizer state、global step、采样器状态、数据指纹和学习率进度。 | 本轮未新增硬错误；不代表出版通过 |
| 032 | L118 · resume 不等于 initialize-from | 段落 | 'initialize-from' 使用旧参数开启一个新实验，应该有新的运行身份；它可以重新建立 optimizer、step 和数据顺序。 | 本轮未新增硬错误；不代表出版通过 |
| 033 | L120 · resume 不等于 initialize-from | 段落 | 如果只载入参数却把它称为 resume，后面的 loss 曲线和训练预算就无法解释。 | 本轮未新增硬错误；不代表出版通过 |
| 034 | L124 · 失败也属于实验结果 | 段落 | 失败可能发生在依赖安装、设备发现、XLA 编译、数据读取、梯度更新、checkpoint 保存或恢复阶段。失败目录至少保留命令、作业 ID、最后成功解析的配置、环境摘要、已有 metrics | 本轮未新增硬错误；不代表出版通过 |
| 035 | L126 · 失败也属于实验结果 | 段落 | 我们不把所有失败都归结为“随机性”。如果恢复后下一批数据不同，应先检查采样器状态和数据指纹；如果设备 OOM，应记录 batch、sequence length、dtype 和编译策略；如果 | 本轮未新增硬错误；不代表出版通过 |
| 036 | L128 · 失败也属于实验结果 | 段落 | 失败记录的作用不是证明项目顺利，而是缩小下一次实验的未知范围。 | 本轮未新增硬错误；不代表出版通过 |
| 037 | L132 · 结论的边界 | 段落 | 实验报告最后必须分别回答： | 本轮未新增硬错误；不代表出版通过 |
| 038 | L134 · 结论的边界 | 段落 | 1. 这次运行实际改变了什么？ | 本轮未新增硬错误；不代表出版通过 |
| 039 | L140 · 结论的边界 | 段落 | 可以使用下面的句式： | 本轮未新增硬错误；不代表出版通过 |
| 040 | L142 · 结论的边界 | 段落 | > 在固定的 tiny 配置和已声明控制变量下，只改变一个研究变量，我们观察到某个指标发生变化。该结果支持一个边界清楚的结论，但不能直接推广到尚未控制或验证的条件。 | 本轮未新增硬错误；不代表出版通过 |
| 041 | L144 · 结论的边界 | 段落 | 本书后面的每个章节都遵守这份契约。环境章节回答“在哪里运行”，数据与表示章节回答“模型实际看到了什么”，baseline 章节回答“系统是否真的会学习”，架构章节回答“状态如何改变”，实验章 | 本轮未新增硬错误；不代表出版通过 |
| 042 | L148 · 本章小结 | 段落 | - 一个实验身份至少包含源码、解析配置、数据指纹、环境、随机与优化状态、预算和输出证据； | 本轮未新增硬错误；不代表出版通过 |

### docs/book/en/01_experiment_contract.md

| 单元 | 位置与小节 | 类型 | 首句/首式 | 本轮处理 |
|---|---|---|---|---|
| 001 | L5 · Overview: turn a comparison into a question | 段落 | Suppose we train the same tiny model twice: once with the TTT inner loop disabled and once with | 本轮未新增硬错误；不代表出版通过 |
| 002 | L7 · Overview: turn a comparison into a question | 段落 | After this chapter, you should be able to fill in the research variable, fixed variables, budge | 本轮未新增硬错误；不代表出版通过 |
| 003 | L11 · Running example: tiny-contract-001 | 段落 | We use tiny-contract-001 throughout. The control reads configs/book/tiny_control.json; the trea | 本轮未新增硬错误；不代表出版通过 |
| 004 | L13 · Running example: tiny-contract-001 | 表格 | ／ Field ／ Control ／ Treatment ／ | 本轮未新增硬错误；不代表出版通过 |
| 005 | L21 · Running example: tiny-contract-001 | 段落 | Write this table before looking at a result. It prevents the comparison rule from changing afte | 本轮未新增硬错误；不代表出版通过 |
| 006 | L25 · Decide what the experiment is meant to prove | 段落 | This book does not treat “the code ran” as a research conclusion. The question is whether, unde | 本轮未新增硬错误；不代表出版通过 |
| 007 | L27 · Decide what the experiment is meant to prove | 段落 | Before starting a run, write one claim in a fixed form: | 本轮未新增硬错误；不代表出版通过 |
| 008 | L29 · Decide what the experiment is meant to prove | 段落 | > Holding ______ fixed, change only ______, and measure whether ______ changes. | 本轮未新增硬错误；不代表出版通过 |
| 009 | L31 · Decide what the experiment is meant to prove | 段落 | For example: hold the tokenizer, effective training tokens, sequence length, optimizer, and har | 本轮未新增硬错误；不代表出版通过 |
| 010 | L35 · The identity of a run | 段落 | A run is more than a command. Treat it as an object with an identity: | 本轮未新增硬错误；不代表出版通过 |
| 011 | L37 · The identity of a run | 公式 | R=(S,C,I_{\mathrm{data}},E,K,Q_{\mathrm{budget}},O), | 本轮未新增硬错误；不代表出版通过 |
| 012 | L41 · The identity of a run | 段落 | where: | 本轮未新增硬错误；不代表出版通过 |
| 013 | L43 · The identity of a run | 段落 | - \(S\): source commit; | 本轮未新增硬错误；不代表出版通过 |
| 014 | L51 · The identity of a run | 段落 | Reproduction does not always mean bitwise identical floating-point values. It means that the ob | 本轮未新增硬错误；不代表出版通过 |
| 015 | L55 · Variables that must be fixed or reported | 表格 | ／ Category ／ What to record ／ | 本轮未新增硬错误；不代表出版通过 |
| 016 | L65 · Variables that must be fixed or reported | 段落 | A comparison may change one of these categories, but the changed item must be named as the rese | 本轮未新增硬错误；不代表出版通过 |
| 017 | L67 · Variables that must be fixed or reported | 段落 | A non-attributable example is a control with a byte tokenizer and sequence length 16, compared  | 本轮未新增硬错误；不代表出版通过 |
| 018 | L71 · Three levels of evidence | 段落 | Report every run at three levels. | 本轮未新增硬错误；不代表出版通过 |
| 019 | L73 · Three levels of evidence | 段落 | **Runnable** means that dependencies, devices, configuration, and data entry points are valid a | 本轮未新增硬错误；不代表出版通过 |
| 020 | L75 · Three levels of evidence | 段落 | **Learnable** means that loss and gradients are finite, training state changes, checkpoints can | 开放 R02：此证据不足以命名为学习 |
| 021 | L77 · Three levels of evidence | 段落 | **Interpretable** means that the control and treatment differ only in the declared variable, ev | 本轮未新增硬错误；不代表出版通过 |
| 022 | L79 · Three levels of evidence | 段落 | A four-step tiny run can establish that the interface and optimization path are connected. It d | 本轮未新增硬错误；不代表出版通过 |
| 023 | L83 · The run directory must explain itself | 段落 | A formal experiment should also preserve source identity, data identity, and its interpretation | 本轮未新增硬错误；不代表出版通过 |
| 024 | L85 · The run directory must explain itself | 代码/示意 | experiment.json   # trainer-written resolved configuration and budget | 本轮未新增硬错误；不代表出版通过 |
| 025 | L96 · The run directory must explain itself | 段落 | JSON, JSONL, and checkpoints are the factual source. The Markdown report is the reading interfa | 本轮未新增硬错误；不代表出版通过 |
| 026 | L98 · The run directory must explain itself | 段落 | For tiny-contract-001, the reader should be able to trace each claim to one artifact: | 本轮未新增硬错误；不代表出版通过 |
| 027 | L100 · The run directory must explain itself | 表格 | ／ Claim ／ Primary artifact ／ Passing condition ／ | 本轮未新增硬错误；不代表出版通过 |
| 028 | L110 · Resume is different from initialize-from | 段落 | 'resume' continues the same optimization process. It must restore model parameters, optimizer s | 本轮未新增硬错误；不代表出版通过 |
| 029 | L112 · Resume is different from initialize-from | 段落 | 'initialize-from' uses old parameters to start a new experiment. It receives a new run identity | 本轮未新增硬错误；不代表出版通过 |
| 030 | L114 · Resume is different from initialize-from | 段落 | Loading parameters alone is not the same as resuming training. | 本轮未新增硬错误；不代表出版通过 |
| 031 | L118 · Failures are evidence | 段落 | Failures may occur during installation, device discovery, XLA compilation, data loading, gradie | 本轮未新增硬错误；不代表出版通过 |
| 032 | L120 · Failures are evidence | 段落 | The purpose of a failure record is to reduce the unknowns in the next run, not to make the proj | 本轮未新增硬错误；不代表出版通过 |
| 033 | L124 · Bound every conclusion | 段落 | Every report should answer: | 本轮未新增硬错误；不代表出版通过 |
| 034 | L126 · Bound every conclusion | 段落 | 1. What actually changed? | 本轮未新增硬错误；不代表出版通过 |
| 035 | L132 · Bound every conclusion | 段落 | Use the following form when needed: | 本轮未新增硬错误；不代表出版通过 |
| 036 | L134 · Bound every conclusion | 段落 | > In the tiny setting, holding \(A\), \(B\), and \(C\) fixed, changing \(D\) produced \(E\). Th | 本轮未新增硬错误；不代表出版通过 |
| 037 | L136 · Bound every conclusion | 段落 | The environment chapter answers where a run executes; the data and representation chapters answ | 本轮未新增硬错误；不代表出版通过 |
| 038 | L140 · Summary | 段落 | - A run identity contains source, resolved configuration, data fingerprints, environment, rando | 本轮未新增硬错误；不代表出版通过 |

### docs/book/01_environment.md

| 单元 | 位置与小节 | 类型 | 首句/首式 | 本轮处理 |
|---|---|---|---|---|
| 001 | L5 · 本章导读：先证明机器能运行 | 段落 | 本章不追求一次把大模型训练起来。我们先用 tiny-contract-001 做一件更小但更可靠的事：让机器报告自己是谁，生成一份短 token 文件，启动一次 tiny 训练，并留下可以解 | 本轮未新增硬错误；不代表出版通过 |
| 002 | L7 · 本章导读：先证明机器能运行 | 段落 | 读完本章，你应该能在目标 Linux 节点完成三项检查：看到 JAX 的设备列表，检查样例 token 的长度与词表范围，找到 tiny 训练的 experiment.json、enviro | 本轮未新增硬错误；不代表出版通过 |
| 003 | L11 · 预备知识与贯穿输入 | 段落 | 你只需要会打开终端、运行 shell 命令，并理解“文件夹中的 JSON 是机器写出的记录”。本章使用 examples/sample.txt；它不是质量足够高的预训练语料，而是用于验证路径 | 本轮未新增硬错误；不代表出版通过 |
| 004 | L15 · 环境不是安装步骤的附录 | 段落 | 从 scratch 项目经常出现一个奇怪现象：代码在作者机器上运行，换一台机器后不是依赖找不到，就是 loss 曲线完全不同。问题未必来自模型，可能来自 Python 小版本、默认 dtyp | 本轮未新增硬错误；不代表出版通过 |
| 005 | L17 · 环境不是安装步骤的附录 | 段落 | 所以环境是实验定义的一部分。我们希望别人拿到仓库后能回答：使用什么代码、什么依赖、什么设备、什么数据、什么命令，才能得到这个结果。只写“安装 JAX”远远不够，JAX 的 CPU 与 CUD | 本轮未新增硬错误；不代表出版通过 |
| 006 | L21 · 三层环境合同 | 段落 | 把运行环境拆成三层，排错会更容易。 | 本轮未新增硬错误；不代表出版通过 |
| 007 | L23 · 三层环境合同 | 段落 | **系统层**包括操作系统、内核、GPU 型号、驱动、CUDA、磁盘和网络文件系统。它决定设备能否被发现、编译缓存放在哪里，以及多个进程能否访问同一目录。 | 本轮未新增硬错误；不代表出版通过 |
| 008 | L25 · 三层环境合同 | 段落 | **Python 层**包括 Python 版本、JAX、Optax、Orbax 和 tokenizer 扩展。它决定 API 和数值实现。'uv.lock' 固定了依赖版本，但不替代驱动与 | 本轮未新增硬错误；不代表出版通过 |
| 009 | L27 · 三层环境合同 | 段落 | **实验层**包括 Git commit、解析后的模型配置、tokenizer、数据摘要、batch、sequence length、seed 和输出路径。它决定这一次究竟训练了什么。 | 本轮未新增硬错误；不代表出版通过 |
| 010 | L29 · 三层环境合同 | 表格 | ／ 层 ／ 典型失败 ／ 应保存的证据 ／ | 本轮未新增硬错误；不代表出版通过 |
| 011 | L35 · 三层环境合同 | 段落 | 这三层不能互相替代。一个正确的环境不保证配置正确，一个正确的配置也不保证 GPU 有足够显存。 | 本轮未新增硬错误；不代表出版通过 |
| 012 | L39 · 从仓库入口开始 | 段落 | 'scripts/bootstrap_linux.sh' 在 Linux 节点下载 uv，安装指定 Python，并按照 extra 同步依赖。它默认包含开发、checkpoint 和 se | 本轮未新增硬错误；不代表出版通过 |
| 013 | L41 · 从仓库入口开始 | 段落 | 'scripts/check_environment.py' 使用 JAX 统计设备，输出 device_count、devices 和 gpu_count。'--require-gpu'  | 本轮未新增硬错误；不代表出版通过 |
| 014 | L43 · 从仓库入口开始 | 段落 | 'scripts/record_environment.sh' 将运行时信息保存为 JSON。训练入口还会在输出目录写入 'environment.json'，这样报告不依赖终端日志。 | 本轮未新增硬错误；不代表出版通过 |
| 015 | L45 · 从仓库入口开始 | 段落 | 在目标 Linux 环境中，第一次准备可以这样做： | 本轮未新增硬错误；不代表出版通过 |
| 016 | L47 · 从仓库入口开始 | 命令 | TTT_INSTALL_CUDA=1 bash scripts/bootstrap_linux.sh | 参数/路径已静态核对；远端执行待验 |
| 017 | L55 · 从仓库入口开始 | 段落 | 成功时，JSON 应有 python、packages.jax、device_count、devices、gpu_count 和 passed。它没有独立的 backend 或 dtype  | 本轮未新增硬错误；不代表出版通过 |
| 018 | L57 · 从仓库入口开始 | 段落 | 这里明确调用 .venv/bin/python，因为 bootstrap 在子 shell 中创建环境，不会激活当前 shell。直接写 python3 可能检查的是系统 Python，随后 | 本轮未新增硬错误；不代表出版通过 |
| 019 | L61 · 为什么先生成样例数据 | 段落 | 直接拿正式语料启动，会把数据问题和模型问题混在一起。'scripts/prepare_sample.sh' 生成一个很小的一维 token 文件，让我们先检查： | 本轮未新增硬错误；不代表出版通过 |
| 020 | L63 · 为什么先生成样例数据 | 段落 | - token 是否位于 vocab_size 范围； | 本轮未新增硬错误；不代表出版通过 |
| 021 | L69 · 为什么先生成样例数据 | 段落 | 它不能证明语言能力，也不能证明清洗策略。它的作用类似电路板上的通断测试：先确认接口连接，再接入复杂输入。 | 本轮未新增硬错误；不代表出版通过 |
| 022 | L71 · 为什么先生成样例数据 | 段落 | 样例输入的第一条检查结果应能回答三个问题：token 数是否至少为 sequence_length + 1 = 17，所有 ID 是否位于 [0, 259)，以及 shift 后是否仍有有效 | 本轮未新增硬错误；不代表出版通过 |
| 023 | L75 · 第一次训练的四个阶段 | 段落 | 一次训练的第一步不等于模型已经开始稳态计算。通常包含： | 本轮未新增硬错误；不代表出版通过 |
| 024 | L77 · 第一次训练的四个阶段 | 段落 | 1. 进程启动与配置解析； | 本轮未新增硬错误；不代表出版通过 |
| 025 | L82 · 第一次训练的四个阶段 | 段落 | 因此 'metrics.jsonl' 中的第一条耗时可能很大。报告吞吐时要区分 compile time 与 steady-state tokens/s，否则一次编译时间会被平均到短实验里， | 本轮未新增硬错误；不代表出版通过 |
| 026 | L84 · 第一次训练的四个阶段 | 段落 | 训练目录还应出现： | 本轮未新增硬错误；不代表出版通过 |
| 027 | L86 · 第一次训练的四个阶段 | 代码/示意 | experiment.json   # 解析后的模型、内循环、训练和 tokenizer 配置 | 本轮未新增硬错误；不代表出版通过 |
| 028 | L95 · 第一次训练的四个阶段 | 段落 | 源码 commit 和样例文件 SHA256 需要另外保存；experiment.json 不会自动补入这两个值。environment.json 中的 source_sha256 是包源码 | 本轮未新增硬错误；不代表出版通过 |
| 029 | L99 · 运行失败时先看哪一层 | 段落 | 如果 'check_environment.py' 找不到 GPU，先看后端和设备，而不是改模型。若设备存在但初始化 OOM，检查 dtype、预分配比例、batch、sequence le | 本轮未新增硬错误；不代表出版通过 |
| 030 | L101 · 运行失败时先看哪一层 | 段落 | 若训练跑完但恢复结果不同，先对比 data fingerprint、采样器状态、optimizer state 和 global step。最后才检查随机浮点差异。把所有失败都归结为“随机性 | 本轮未新增硬错误；不代表出版通过 |
| 031 | L105 · CPU、GPU、JAX 和 mesh 各自负责什么 | 段落 | 如果你第一次接触训练环境，可以先把它们分成四个角色。CPU 是负责启动程序、准备数据和调度工作的通用处理器；GPU 有大量适合矩阵乘法的并行计算单元，负责模型的大部分张量计算；JAX 是把  | 本轮未新增硬错误；不代表出版通过 |
| 032 | L107 · CPU、GPU、JAX 和 mesh 各自负责什么 | 段落 | 例如，hidden 的形状是 [B, L, d]，其中 B=2、L=4、d=8。单卡运行时，整个数组可以放在一张 GPU 上。若建立一个有两张卡的 data mesh，常见做法是把 batc | 本轮未新增硬错误；不代表出版通过 |
| 033 | L109 · CPU、GPU、JAX 和 mesh 各自负责什么 | 段落 | JAX 还会把第一次调用分成 tracing、编译和执行。可以把它想成先把菜谱翻译成设备指令，再执行。假设首步需 40 秒、后续每步 2 秒，将它们混合会低估稳态吞吐。这是时间计算示例，未声 | 本轮未新增硬错误；不代表出版通过 |
| 034 | L113 · 本章的完成标准 | 段落 | 完成环境章需要一个可审计的目录，而不是一条绿色命令： | 本轮未新增硬错误；不代表出版通过 |
| 035 | L115 · 本章的完成标准 | 段落 | - 代码 commit 可以从输出中找到； | 本轮未新增硬错误；不代表出版通过 |
| 036 | L125 · 本章的最小 checkpoint 与证据 | 段落 | 在已安装依赖的目标节点上，按顺序执行： | 本轮未新增硬错误；不代表出版通过 |
| 037 | L127 · 本章的最小 checkpoint 与证据 | 命令 | bash scripts/prepare_sample.sh outputs/sample-tokens.npy | 参数/路径已静态核对；远端执行待验 |
| 038 | L137 · 本章的最小 checkpoint 与证据 | 段落 | 首次执行时使用尚未训练过的输出目录；重复完整实验时换一个目录，继续原运行则按下一章使用 resume。GPU 节点应先完成设备检查，再将整段 JAX_PLATFORMS=cpu bash s | 本轮未新增硬错误；不代表出版通过 |
| 039 | L139 · 本章的最小 checkpoint 与证据 | 段落 | 日志中的 tokens_seen 和 tokens_per_second 按 batch size × sequence length 计算，包含被 mask 的位置。当前样例是普通 NPY | 本轮未新增硬错误；不代表出版通过 |
| 040 | L141 · 本章的最小 checkpoint 与证据 | 段落 | 若命令失败，先检查样例路径、configs/book/tiny_control.json 和环境记录的 devices。初始化阶段失败可能尚未产生 status.json，应保留终端或调度日 | 本轮未新增硬错误；不代表出版通过 |
| 041 | L145 · 本章小结 | 段落 | - 系统层、Python 层和实验层必须分别记录； | 本轮未新增硬错误；不代表出版通过 |

### docs/book/en/01_environment.md

| 单元 | 位置与小节 | 类型 | 首句/首式 | 本轮处理 |
|---|---|---|---|---|
| 001 | L5 · Overview: prove that the machine can run | 段落 | This chapter does not try to train a large model in one command. We first make a smaller claim  | 本轮未新增硬错误；不代表出版通过 |
| 002 | L7 · Overview: prove that the machine can run | 段落 | After this chapter, you should be able to read the JAX device list, inspect the length and voca | 本轮未新增硬错误；不代表出版通过 |
| 003 | L11 · Prerequisites and running input | 段落 | You only need a terminal, basic shell commands, and the idea that JSON files in an output direc | 本轮未新增硬错误；不代表出版通过 |
| 004 | L15 · Why the environment belongs to the experiment | 段落 | The first failure in a from-scratch project is often not a wrong equation. It is a Python versi | 本轮未新增硬错误；不代表出版通过 |
| 005 | L17 · Why the environment belongs to the experiment | 段落 | Think of the environment as the laboratory bench. The model is the instrument, the dataset is t | 本轮未新增硬错误；不代表出版通过 |
| 006 | L21 · The three layers of a machine contract | 段落 | We record three layers because each one answers a different debugging question: | 本轮未新增硬错误；不代表出版通过 |
| 007 | L23 · The three layers of a machine contract | 表格 | ／ Layer ／ Beginner question ／ Typical evidence ／ | 本轮未新增硬错误；不代表出版通过 |
| 008 | L29 · The three layers of a machine contract | 段落 | The layers are related but cannot replace one another. A correct Python installation does not g | 本轮未新增硬错误；不代表出版通过 |
| 009 | L33 · What the repository provides | 段落 | The shell scripts are thin entry points; Python owns the experiment semantics so the same run c | 本轮未新增硬错误；不代表出版通过 |
| 010 | L35 · What the repository provides | 段落 | - 'scripts/bootstrap_linux.sh' installs the pinned Python and dependencies. Setting 'TTT_INSTAL | 本轮未新增硬错误；不代表出版通过 |
| 011 | L42 · What the repository provides | 段落 | The sample file is a wiring test, not a language benchmark. It can reveal an out-of-range token | 本轮未新增硬错误；不代表出版通过 |
| 012 | L44 · What the repository provides | 段落 | For the tiny configuration, the first sample-data check should answer three questions: are ther | 本轮未新增硬错误；不代表出版通过 |
| 013 | L48 · A minimal first run | 段落 | Run the following on the target Linux machine or inside a GPU allocation. For a published resul | 本轮未新增硬错误；不代表出版通过 |
| 014 | L50 · A minimal first run | 命令 | git clone https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch.git | 参数/路径已静态核对；远端执行待验 |
| 015 | L60 · A minimal first run | 段落 | The JSON should contain python, packages.jax, device_count, devices, gpu_count, and passed. The | 本轮未新增硬错误；不代表出版通过 |
| 016 | L62 · A minimal first run | 段落 | Use .venv/bin/python explicitly: bootstrap creates the environment in a child shell and does no | 本轮未新增硬错误；不代表出版通过 |
| 017 | L66 · What happens during the first step | 段落 | “The command started” does not mean that steady-state training has begun. A first step usually  | 本轮未新增硬错误；不代表出版通过 |
| 018 | L68 · What happens during the first step | 段落 | 1. parse the JSON and fill default values; | 本轮未新增硬错误；不代表出版通过 |
| 019 | L73 · What happens during the first step | 段落 | If the first step takes 40 seconds and later steps take 2 seconds, averaging them gives a misle | 本轮未新增硬错误；不代表出版通过 |
| 020 | L77 · The expected evidence directory | 段落 | A successful tiny run should leave evidence similar to: | 本轮未新增硬错误；不代表出版通过 |
| 021 | L79 · The expected evidence directory | 代码/示意 | experiment.json   # resolved model, inner-loop, training, tokenizer configuration | 本轮未新增硬错误；不代表出版通过 |
| 022 | L88 · The expected evidence directory | 段落 | Git commit and data-file SHA256 must be recorded separately; experiment.json does not add them. | 本轮未新增硬错误；不代表出版通过 |
| 023 | L92 · Reading a failure without guessing | 段落 | If 'check_environment.py' cannot find a GPU, inspect the backend and device list before changin | 本轮未新增硬错误；不代表出版通过 |
| 024 | L94 · Reading a failure without guessing | 段落 | If a resumed run diverges, compare the data fingerprint, sampler position, optimizer state, and | 本轮未新增硬错误；不代表出版通过 |
| 025 | L98 · CPU, GPU, JAX, and a device mesh | 段落 | A CPU starts the program, prepares data, and schedules work. A GPU has many parallel units suit | 本轮未新增硬错误；不代表出版通过 |
| 026 | L100 · CPU, GPU, JAX, and a device mesh | 段落 | For example, a hidden array of shape \([B,L,d]=[2,4,8]\) can reside on one GPU. A two-device da | 本轮未新增硬错误；不代表出版通过 |
| 027 | L104 · Minimal checkpoint and expected evidence | 段落 | On a node with the dependencies installed, the shortest CPU path is: | 本轮未新增硬错误；不代表出版通过 |
| 028 | L106 · Minimal checkpoint and expected evidence | 命令 | bash scripts/prepare_sample.sh outputs/sample-tokens.npy | 参数/路径已静态核对；远端执行待验 |
| 029 | L116 · Minimal checkpoint and expected evidence | 段落 | Use an output directory that has not already trained a model. Repeat a full experiment in a new | 本轮未新增硬错误；不代表出版通过 |
| 030 | L118 · Minimal checkpoint and expected evidence | 段落 | The logged tokens_seen and tokens_per_second count batch size × sequence length, including mask | 本轮未新增硬错误；不代表出版通过 |
| 031 | L120 · Minimal checkpoint and expected evidence | 段落 | If a command fails, inspect the sample path, configs/book/tiny_control.json, and devices in the | 本轮未新增硬错误；不代表出版通过 |
| 032 | L124 · Summary | 段落 | - System, Python, and experiment layers need separate records. | 本轮未新增硬错误；不代表出版通过 |

### docs/book/02_reproducible_runs.md

| 单元 | 位置与小节 | 类型 | 首句/首式 | 本轮处理 |
|---|---|---|---|---|
| 001 | L5 · 本章导读：中断一次 tiny run，再证明它真的接上了 | 段落 | “能打开 checkpoint”不是恢复实验。我们要证明的是：连续运行和中断后恢复运行，在恢复点之后读取相同的输入、使用相同的 step 和 optimizer state，并在数值容忍范围 | 本轮未新增硬错误；不代表出版通过 |
| 002 | L7 · 本章导读：中断一次 tiny run，再证明它真的接上了 | 段落 | 读完本章，你应该能区分 --resume 与 --initialize-from，解释 RNG 为什么是实验状态，并从 status.json、metrics.jsonl 和 checkpo | 本轮未新增硬错误；不代表出版通过 |
| 003 | L11 · 预备知识和结果表 | 段落 | 假设 tiny run 在 step 2 保存 checkpoint，连续路径继续到 step 4，中断路径从 step 2 恢复后也继续到 step 4： | 本轮未新增硬错误；不代表出版通过 |
| 004 | L13 · 预备知识和结果表 | 表格 | ／ 检查项 ／ 连续路径 ／ 恢复路径 ／ 成功判据 ／ | 本轮未新增硬错误；不代表出版通过 |
| 005 | L21 · 预备知识和结果表 | 段落 | 这张表说明“恢复成功”是一个状态比较，而不是文件存在检查。 | 本轮未新增硬错误；不代表出版通过 |
| 006 | L25 · “同样的命令”并不代表同样的实验 | 段落 | 两台机器执行同一条命令，可能读取不同的数据文件、使用不同默认值、从不同 checkpoint 开始，甚至在不同 step 记录第一条 loss。命令是入口，不是实验身份。 | 本轮未新增硬错误；不代表出版通过 |
| 007 | L27 · “同样的命令”并不代表同样的实验 | 段落 | 我们把一个运行写成： | 本轮未新增硬错误；不代表出版通过 |
| 008 | L29 · “同样的命令”并不代表同样的实验 | 公式 | R=(S,C,I_{\mathrm{data}},E,K,Q_{\mathrm{budget}},O), | 本轮未新增硬错误；不代表出版通过 |
| 009 | L33 · “同样的命令”并不代表同样的实验 | 段落 | 其中 \(S\) 是源码 commit，\(C\) 是解析后的配置，\(I_{\mathrm{data}}\) 是数据、tokenizer 和 manifest 的指纹，\(E\) 是环境， | 本轮未新增硬错误；不代表出版通过 |
| 010 | L37 · resume 和 initialize-from 是两个动词 | 段落 | '--resume' 表示继续同一个优化过程。它必须恢复： | 本轮未新增硬错误；不代表出版通过 |
| 011 | L39 · resume 和 initialize-from 是两个动词 | 段落 | - 模型参数； | 本轮未新增硬错误；不代表出版通过 |
| 012 | L46 · resume 和 initialize-from 是两个动词 | 段落 | 如果只载入参数，优化器会从零开始，学习率 schedule 也可能回到 warmup。这样得到的是新实验，即使输出目录名字叫 resume。 | 本轮未新增硬错误；不代表出版通过 |
| 013 | L48 · resume 和 initialize-from 是两个动词 | 段落 | '--initialize-from' 表示使用已有参数作为新实验的初始点。它重新创建 optimizer、step 和数据顺序，因此需要新的输出目录。当前入口只接受相同参数树、叶张量形状和 | 本轮未新增硬错误；不代表出版通过 |
| 014 | L52 · 一个最小的失败案例 | 段落 | 假设训练在 global step 100 保存 checkpoint。这个 checkpoint 中的参数已经看过 100 个 batch，AdamW 也积累了对应的动量。若只复制 par | 本轮未新增硬错误；不代表出版通过 |
| 015 | L54 · 一个最小的失败案例 | 段落 | 1. 优化器的一阶、二阶统计被清零，更新方向会突然改变； | 本轮未新增硬错误；不代表出版通过 |
| 016 | L59 · 一个最小的失败案例 | 段落 | 恢复检查不能只问“文件能否打开”。至少要用同一个初始 checkpoint 做一次短实验：一条路径连续运行，另一条路径在中间保存并 resume，然后比较恢复点之后的下一批输入、step、有 | 本轮未新增硬错误；不代表出版通过 |
| 017 | L61 · 一个最小的失败案例 | 段落 | 相反，从同结构的 C 初始化 F 时，应明确产生新的身份： | 本轮未新增硬错误；不代表出版通过 |
| 018 | L63 · 一个最小的失败案例 | 代码/示意 | C:      source=S1, data=D1, step=1000, optimizer=O_C | 本轮未新增硬错误；不代表出版通过 |
| 019 | L68 · 一个最小的失败案例 | 段落 | 这两行共享参数文件，但不是同一次运行；报告不能把第二行的 loss 曲线接在第一行后面。 | 本轮未新增硬错误；不代表出版通过 |
| 020 | L72 · 当前 checkpoint 保存什么 | 段落 | 'ttt/checkpoint.py' 将 Experiment、TrainState 和数据状态组合起来。实验对象包含模型和训练配置；TrainState 包含 params、opt_st | 本轮未新增硬错误；不代表出版通过 |
| 021 | L74 · 当前 checkpoint 保存什么 | 段落 | 检查点不是只有模型权重。对 TTT 来说，快权重有两种不同角色： | 本轮未新增硬错误；不代表出版通过 |
| 022 | L76 · 当前 checkpoint 保存什么 | 段落 | - 训练时的 fast weights 是一个序列内部的临时状态，外循环更新静态模型； | 本轮未新增硬错误；不代表出版通过 |
| 023 | L79 · 当前 checkpoint 保存什么 | 段落 | 把 session snapshot 当成训练 checkpoint，会让优化器和数据采样状态缺失；把训练 checkpoint 当成会话 snapshot，又会携带不该暴露给请求的全局状态 | 本轮未新增硬错误；不代表出版通过 |
| 024 | L83 · 运行目录应能自己解释自己 | 段落 | 先区分训练器当前写出的文件与实验者需要补充的记录。当前训练路径在初始化成功后写入： | 本轮未新增硬错误；不代表出版通过 |
| 025 | L85 · 运行目录应能自己解释自己 | 表格 | ／ 字段 ／ 作用 ／ | 本轮未新增硬错误；不代表出版通过 |
| 026 | L93 · 运行目录应能自己解释自己 | 段落 | 训练器没有自动生成 error.json 或独立 checkpoint/ 目录。源码 commit、文件 SHA256、完整异常栈和解释报告应另行留档；上章已保存前两项。参数初始化、数据打开 | 本轮未新增硬错误；不代表出版通过 |
| 027 | L95 · 运行目录应能自己解释自己 | 段落 | 一份可审计的输出目录至少能回答： | 本轮未新增硬错误；不代表出版通过 |
| 028 | L97 · 运行目录应能自己解释自己 | 代码/示意 | 这次用哪个 commit？ | 本轮未新增硬错误；不代表出版通过 |
| 029 | L107 · 运行目录应能自己解释自己 | 段落 | 这些问题是完整归档的目标，不是上述 JSON 已自动覆盖的字段。experiment.json 写配置，environment.json 写机器与源码摘要，checkpoint 的 data | 本轮未新增硬错误；不代表出版通过 |
| 030 | L109 · 运行目录应能自己解释自己 | 段落 | 可以把运行信息组织成一个 manifest： | 本轮未新增硬错误；不代表出版通过 |
| 031 | L111 · 运行目录应能自己解释自己 | 代码/示意 | { | 本轮未新增硬错误；不代表出版通过 |
| 032 | L124 · 运行目录应能自己解释自己 | 段落 | 这段结构是教学示意；仓库的实际字段以训练输出为准。重要的是把输入、预算和状态分开。 | 本轮未新增硬错误；不代表出版通过 |
| 033 | L128 · 失败也应成为记录 | 段落 | 作业可能在下载依赖、编译、设备初始化、数据读取、梯度更新或保存时失败。失败目录应该保留： | 本轮未新增硬错误；不代表出版通过 |
| 034 | L130 · 失败也应成为记录 | 段落 | - 运行命令和调度作业 ID； | 本轮未新增硬错误；不代表出版通过 |
| 035 | L137 · 失败也应成为记录 | 段落 | “没有输出”通常是最难恢复的状态。完整生产启动器应在初始化失败时也留下异常记录；这是外层启动协议需要补充的能力，当前训练入口没有覆盖所有这类失败。 | 本轮未新增硬错误；不代表出版通过 |
| 036 | L141 · 如何比较两个结果 | 段落 | 比较模型质量时，优先固定有效 token 预算和数据 manifest。比较系统效率时，还要固定设备、精度、batch、序列长度和编译策略。比较流式状态时，固定初始 cache、位置和 se | 本轮未新增硬错误；不代表出版通过 |
| 037 | L143 · 如何比较两个结果 | 段落 | 可以把差异分成三类： | 本轮未新增硬错误；不代表出版通过 |
| 038 | L145 · 如何比较两个结果 | 段落 | **定义差异**：词表、模型结构、训练 token 不同。它们构成新实验，不能用“同配置”描述。 | 本轮未新增硬错误；不代表出版通过 |
| 039 | L147 · 如何比较两个结果 | 段落 | **环境差异**：GPU 型号、驱动或 XLA 版本不同。质量可能仍可比较，但性能必须分开。 | 本轮未新增硬错误；不代表出版通过 |
| 040 | L149 · 如何比较两个结果 | 段落 | **数值差异**：同一输入下并行归约、低精度或非确定性带来的小差异。需要给出容忍范围，而不是把所有差异都叫失败。 | 本轮未新增硬错误；不代表出版通过 |
| 041 | L153 · 从一个确定的检查点验证恢复 | 段落 | 四步 tiny 运行可能在人工按下中断键前已经完成。这里先完成连续路径，再从它保留的 step 2 检查点在另一个目录继续。检查点保存的是当时的完整状态，因此这能检验从保存点恢复的数值语义； | 本轮未新增硬错误；不代表出版通过 |
| 042 | L155 · 从一个确定的检查点验证恢复 | 命令 | set -e | 参数/路径已静态核对；远端执行待验 |
| 043 | L171 · 从一个确定的检查点验证恢复 | 段落 | 连续路径应记录 step 1–4，恢复路径只记录 step 3–4。不要改配置中的总步数来制造中断：当前 resume 要求解析后的实验配置一致。--auto-resume 适合继续同一目录 | 本轮未新增硬错误；不代表出版通过 |
| 044 | L173 · 从一个确定的检查点验证恢复 | 段落 | 随机数生成器可看成一台确定性机器：seed 选择起始状态，每抽取一批数据就推进一次内部状态。重新设同一个 seed 会从第一批重新开始；恢复内部状态才会接着抽第三批。下面从 seed 连续抽 | 本轮未新增硬错误；不代表出版通过 |
| 045 | L175 · 从一个确定的检查点验证恢复 | 命令 | JAX_PLATFORMS=cpu .venv/bin/python - <<'PY' | 参数/路径已静态核对；远端执行待验 |
| 046 | L233 · 从一个确定的检查点验证恢复 | 段落 | 只有断言全部完成才会写入 outputs/recovery-check.json。其中 valid_targets 来自实际返回的 mask，而非训练日志中的名义 token 计数。这是待读 | 本轮未新增硬错误；不代表出版通过 |
| 047 | L237 · 本章小结 | 段落 | - resume 恢复同一个优化过程，initialize-from 创建新的实验身份； | 本轮未新增硬错误；不代表出版通过 |

### docs/book/en/02_reproducible_runs.md

| 单元 | 位置与小节 | 类型 | 首句/首式 | 本轮处理 |
|---|---|---|---|---|
| 001 | L5 · Overview: interrupt a tiny run and prove that it continues | 段落 | “The checkpoint opens” is not a recovery experiment. We need to show that an uninterrupted run  | 本轮未新增硬错误；不代表出版通过 |
| 002 | L7 · Overview: interrupt a tiny run and prove that it continues | 段落 | After this chapter, you should be able to distinguish --resume from --initialize-from, explain  | 本轮未新增硬错误；不代表出版通过 |
| 003 | L11 · Prerequisites and comparison table | 段落 | Suppose a tiny run saves at step 2. One path continues to step 4; the other resumes from step 2 | 本轮未新增硬错误；不代表出版通过 |
| 004 | L13 · Prerequisites and comparison table | 表格 | ／ Check ／ Uninterrupted ／ Resumed ／ Passing condition ／ | 本轮未新增硬错误；不代表出版通过 |
| 005 | L21 · Prerequisites and comparison table | 段落 | Recovery is therefore a state comparison, not a file-existence check. | 本轮未新增硬错误；不代表出版通过 |
| 006 | L25 · A run is an object | 段落 | A command proves that a program started. It does not explain why the result looks the way it do | 本轮未新增硬错误；不代表出版通过 |
| 007 | L27 · A run is an object | 段落 | We write that identity as | 本轮未新增硬错误；不代表出版通过 |
| 008 | L29 · A run is an object | 公式 | R=(S,C,I_{\mathrm{data}},E,K,Q_{\mathrm{budget}},O), | 本轮未新增硬错误；不代表出版通过 |
| 009 | L33 · A run is an object | 段落 | where &#36;S&#36; is the source commit, &#36;C&#36; the resolved configuration, &#36;I_{\mathrm{data}}&#36; the fingerp | 本轮未新增硬错误；不代表出版通过 |
| 010 | L35 · A run is an object | 代码/示意 | experiment id | 本轮未新增硬错误；不代表出版通过 |
| 011 | L45 · A run is an object | 段落 | Without one of these fields, reproduction becomes an approximation. | 本轮未新增硬错误；不代表出版通过 |
| 012 | L49 · Resume semantics | 段落 | resume restores parameters, optimizer state, consumed-data position, and random state. It is fo | 本轮未新增硬错误；不代表出版通过 |
| 013 | L51 · Resume semantics | 段落 | initialize-from loads parameters but creates a new optimizer, step, and data order in a new out | 本轮未新增硬错误；不代表出版通过 |
| 014 | L55 · A small failure case | 段落 | Suppose a checkpoint was written at global step 100. Its parameters have seen 100 batches, and  | 本轮未新增硬错误；不代表出版通过 |
| 015 | L57 · A small failure case | 段落 | 1. the optimizer moments are zero, so the update direction changes abruptly; | 本轮未新增硬错误；不代表出版通过 |
| 016 | L62 · A small failure case | 段落 | “The checkpoint opens” is therefore not a recovery test. Use the same short run in two paths: o | 本轮未新增硬错误；不代表出版通过 |
| 017 | L64 · A small failure case | 段落 | Initializing F from a structurally identical C creates a new identity: | 本轮未新增硬错误；不代表出版通过 |
| 018 | L66 · A small failure case | 代码/示意 | C:      source=S1, data=D1, step=1000, optimizer=O_C | 本轮未新增硬错误；不代表出版通过 |
| 019 | L71 · A small failure case | 段落 | The two runs share an initial parameter file but are not one continuous curve. A report must no | 本轮未新增硬错误；不代表出版通过 |
| 020 | L75 · Code and artifacts | 段落 | Separate current automatic output from additional records the experimenter must preserve. After | 本轮未新增硬错误；不代表出版通过 |
| 021 | L77 · Code and artifacts | 表格 | ／ File or field ／ Purpose ／ | 本轮未新增硬错误；不代表出版通过 |
| 022 | L85 · Code and artifacts | 段落 | The trainer does not create error.json or a separate checkpoint/ directory. Save Git commit, da | 本轮未新增硬错误；不代表出版通过 |
| 023 | L87 · Code and artifacts | 段落 | - ttt/provenance.py records Python, JAX, device, driver, PBS, and source information. | 本轮未新增硬错误；不代表出版通过 |
| 024 | L92 · Code and artifacts | 段落 | Machine-readable records are the source of truth. The checkpoint's data_state holds the data fi | 本轮未新增硬错误；不代表出版通过 |
| 025 | L94 · Code and artifacts | 段落 | Failures should leave their output directory and error status. A failed compile or incompatible | 本轮未新增硬错误；不代表出版通过 |
| 026 | L98 · Validate recovery from a specific checkpoint | 段落 | A four-step tiny run can finish before a person manages to interrupt it. Instead, complete the  | 本轮未新增硬错误；不代表出版通过 |
| 027 | L100 · Validate recovery from a specific checkpoint | 命令 | set -e | 参数/路径已静态核对；远端执行待验 |
| 028 | L116 · Validate recovery from a specific checkpoint | 段落 | The uninterrupted log should contain steps 1–4; the resumed log should contain only steps 3–4.  | 本轮未新增硬错误；不代表出版通过 |
| 029 | L118 · Validate recovery from a specific checkpoint | 段落 | A pseudo-random generator is a deterministic machine: a seed selects its initial state, and eac | 本轮未新增硬错误；不代表出版通过 |
| 030 | L120 · Validate recovery from a specific checkpoint | 命令 | JAX_PLATFORMS=cpu .venv/bin/python - <<'PY' | 参数/路径已静态核对；远端执行待验 |
| 031 | L178 · Validate recovery from a specific checkpoint | 段落 | Only after all assertions succeed does this write outputs/recovery-check.json. Its valid_target | 本轮未新增硬错误；不代表出版通过 |
| 032 | L182 · Summary | 段落 | Resume continues one optimization process; initialize-from creates a new one with a compatible  | 本轮未新增硬错误；不代表出版通过 |

### docs/book/03_configuration.md

| 单元 | 位置与小节 | 类型 | 首句/首式 | 本轮处理 |
|---|---|---|---|---|
| 001 | L5 · 本章导读：让每个数字都有来源 | 段落 | 在 tiny-contract-001 中，词表大小是 259、隐藏宽度是 16、attention 有 2 个 query head。读者应该能够从这三个数字一路算到 embedding、 | 本轮未新增硬错误；不代表出版通过 |
| 002 | L7 · 本章导读：让每个数字都有来源 | 段落 | 读完本章，你应该能加载一份配置，列出解析后的默认值，手算一个 batch 的 shape，并解释为什么 tokenizer 的词表大小会固定分类头最后一维。本章不讨论训练是否有效，下一章才会 | 本轮未新增硬错误；不代表出版通过 |
| 003 | L11 · 预备知识和 tiny 目标 | 段落 | 你只需要知道矩阵可以把一组数字映射成另一组数字。贯穿输入为 input_ids，形状是 [B,L]=[2,4]；本章的完成标准是写出它经过 embedding、attention/MLP 和 | 本轮未新增硬错误；不代表出版通过 |
| 004 | L15 · 为什么模型不应该“猜”参数 | 段落 | 如果模型代码在 forward 里根据张量形状猜 head 数、词表或快权重层位置，实验配置就会变成隐藏状态。一个 JSON 看起来写了 hidden_size，实际模型却在别处覆盖它，最后 | 本轮未新增硬错误；不代表出版通过 |
| 005 | L17 · 为什么模型不应该“猜”参数 | 段落 | 配置对象应该是架构合同：它写明所有会影响参数形状、计算图、状态和比较公平性的字段；加载时立即校验；保存时包含补全后的默认值。模型类只消费这份合同。 | 本轮未新增硬错误；不代表出版通过 |
| 006 | L21 · 从词表到分类头 | 段落 | 设： | 本轮未新增硬错误；不代表出版通过 |
| 007 | L23 · 从词表到分类头 | 段落 | - \(V\)：词表大小； | 本轮未新增硬错误；不代表出版通过 |
| 008 | L29 · 从词表到分类头 | 段落 | 输入 embedding： | 本轮未新增硬错误；不代表出版通过 |
| 009 | L31 · 从词表到分类头 | 公式 | E\in\mathbb R^{V\times d}. | 本轮未新增硬错误；不代表出版通过 |
| 010 | L35 · 从词表到分类头 | 段落 | 每个位置的隐藏状态位于 &#36;\mathbb R^d&#36;。SwiGLU 的中间计算可以使用 &#36;d_{\mathrm{ff}}&#36;，但残差输出必须回到 &#36;d&#36;。最后： | 本轮未新增硬错误；不代表出版通过 |
| 011 | L37 · 从词表到分类头 | 公式 | \mathbf z_{b,t}=\mathbf h_{b,t}W_{\mathrm{out}}, | 本轮未新增硬错误；不代表出版通过 |
| 012 | L42 · 从词表到分类头 | 段落 | 当前实现的 'lm_head' 使用 'bias=False'；如果以后显式启用输出偏置，可以在公式中再加 \(\mathbf b_{\mathrm{out}}\in\mathbb R^V\ | 本轮未新增硬错误；不代表出版通过 |
| 013 | L44 · 从词表到分类头 | 段落 | 这也解释了为什么不能只把 MLP 中间层扩大来“增加词表能力”。中间层的宽度改变计算和表示容量，词表投影的输出空间仍由 tokenizer 决定。 | 本轮未新增硬错误；不代表出版通过 |
| 014 | L48 · 四类字段和它们的边界 | 段落 | **模型尺寸字段**包括 vocab_size、hidden_size、intermediate_size、num_hidden_layers、num_attention_heads 和 n | 本轮未新增硬错误；不代表出版通过 |
| 015 | L50 · 四类字段和它们的边界 | 段落 | **序列字段**包括 max_position_embeddings、sliding_window、ttt_chunk_size 和 layer_types。它们决定 cache 容量、分块 | 本轮未新增硬错误；不代表出版通过 |
| 016 | L52 · 四类字段和它们的边界 | 段落 | **自适应字段**包括 use_fast_mlp、ttt_suffix_length、adaptation_target、inner.enabled 和 inner learning rat | 本轮未新增硬错误；不代表出版通过 |
| 017 | L54 · 四类字段和它们的边界 | 段落 | **系统字段**包括 dtype、attention_backend、loss block size、parallelism 和 expert_parallel_size。它们影响运行方式， | 本轮未新增硬错误；不代表出版通过 |
| 018 | L56 · 四类字段和它们的边界 | 段落 | 仓库的 TTTE2EConfig 将这些字段集中在 ttt/models/ttt_e2e/configuration_ttt_e2e.py；训练配置则由 ttt/config.py 和 tt | 本轮未新增硬错误；不代表出版通过 |
| 019 | L60 · 默认值也属于实验输入 | 段落 | 'ttt_suffix_length=None' 在启用 fast MLP 或 adaptation_target="router" 时取最后四分之一层，至少一层，其他情况下为 0；'num | 本轮未新增硬错误；不代表出版通过 |
| 020 | L62 · 默认值也属于实验输入 | 段落 | 因此 'to_dict()' 将补全后的字段写入实验记录。配置审计要问两个问题：原始输入是什么，解析后的实际值是什么。迁移到新版本时，默认规则改变也会改变模型；配置文件不变不代表实验不变。 | 本轮未新增硬错误；不代表出版通过 |
| 021 | L66 · 校验为什么要早 | 段落 | 配置类检查正整数、head 整除性、旋转位置编码的偶数维度、层类型数量、suffix 范围、dtype、特殊 token ID 和 attention 后端。 | 本轮未新增硬错误；不代表出版通过 |
| 022 | L68 · 校验为什么要早 | 段落 | 例如： | 本轮未新增硬错误；不代表出版通过 |
| 023 | L70 · 校验为什么要早 | 段落 | - hidden_size 不能被 num_attention_heads 整除，无法推导默认 head_dim； | 本轮未新增硬错误；不代表出版通过 |
| 024 | L76 · 校验为什么要早 | 段落 | 这些错误在 Python 初始化阶段失败，远比编译十分钟后才发现好。 | 本轮未新增硬错误；不代表出版通过 |
| 025 | L80 · AutoConfig 与 AutoModel | 段落 | 'AutoConfig.from_pretrained' 读取配置，并兼容早期没有 model_type 的训练配置；'AutoModelForCausalLM' 根据 mode 选择 ba | 本轮未新增硬错误；不代表出版通过 |
| 026 | L82 · AutoConfig 与 AutoModel | 段落 | 但自动接口也需要边界。它不能从一个 ttt_e2e checkpoint 自动推断你想用 training 还是 inference 语义；调用方必须显式选择。加载参数不等于加载 optim | 本轮未新增硬错误；不代表出版通过 |
| 027 | L86 · 一次配置审计 | 段落 | 审计 tiny 配置时，依次核对： | 本轮未新增硬错误；不代表出版通过 |
| 028 | L88 · 一次配置审计 | 代码/示意 | vocab_size = 259 | 本轮未新增硬错误；不代表出版通过 |
| 029 | L102 · 一次配置审计 | 段落 | 上述是从配置摘出的审计表，不是可以直接载入的 JSON。模型字段属于 model 对象，sequence_length 属于 training，更新开关属于 inner。下面把这些数字连接到 | 本轮未新增硬错误；不代表出版通过 |
| 030 | L106 · 先用两维向量理解查表和投影 | 段落 | 向量是一列有顺序的数；矩阵把多列这样的数排列起来。为了直接算出每一项，暂取四个 token、每个 token 两个数的 embedding： | 本轮未新增硬错误；不代表出版通过 |
| 031 | L108 · 先用两维向量理解查表和投影 | 公式 | \mathbf E=\begin{pmatrix}1&0\\0&1\\2&1\\-1&1\end{pmatrix}. | 本轮未新增硬错误；不代表出版通过 |
| 032 | L112 · 先用两维向量理解查表和投影 | 段落 | 输入 ID 为 0 和 2 时，查表得到第 0 行 \((1,0)\) 和第 2 行 \((2,1)\)，组成形状为 \([2,2]\) 的数组。第一个维度数位置，第二个维度数每个位置的特征 | 本轮未新增硬错误；不代表出版通过 |
| 033 | L114 · 先用两维向量理解查表和投影 | 公式 | \mathbf h\mathbf E^{\mathsf T} | 本轮未新增硬错误；不代表出版通过 |
| 034 | L120 · 先用两维向量理解查表和投影 | 段落 | 两个特征产生四个候选 token 的分数。多个位置一起计算仍用同一个投影；再加入 batch 轴，就得到 \([B,L,d]\to[B,L,V]\)。这个例子仅演示查表和输出投影，省略了中间 | 本轮未新增硬错误；不代表出版通过 |
| 035 | L124 · 用一个 tiny shape 走完整条路径 | 段落 | 使用 tiny 的模型宽度 \(V=259\)、\(d=16\)、\(d_{\mathrm{ff}}=32\)，另取教学 batch \(B=2\)、位置数 \(L=4\)；这两个演算维度不 | 本轮未新增硬错误；不代表出版通过 |
| 036 | L138 · 用一个 tiny shape 走完整条路径 | 段落 | 下一章沿着这张 shape 图追踪一个 token，理解 Attention、MLP′、普通 FFN 和残差的先后顺序。 | 本轮未新增硬错误；不代表出版通过 |
| 037 | L142 · 头维度和 MLP 中间维度 | 段落 | 上面的批量布局用于理解多个样本如何叠放。当前模型 forward 实际处理一条序列：输入是 \([L]\)，embedding 后为 \([L,d]\)，输出为 \([L,V]\)。训练代码 | C01：根代理修订后已静态复核 |
| 038 | L144 · 头维度和 MLP 中间维度 | 段落 | 对 tiny 的 \(d=16,n_h=2,n_{\mathrm{kv}}=1\)，默认每头宽度 \(d_h=8\)。Query 投影产生 \(2\times8=16\) 个数，reshap | C02：Q/K/V与SwiGLU形状已复核 |
| 039 | L146 · 头维度和 MLP 中间维度 | 段落 | SwiGLU 中的“中间宽度 32”也不是把输出改成 32。对每个行向量 \(h\in\mathbb R^{16}\)，先作两个投影，再逐元素门控，最后投影回来： | 本轮未新增硬错误；不代表出版通过 |
| 040 | L148 · 头维度和 MLP 中间维度 | 公式 | g=hW_g,\quad u=hW_u,\quad | C02：Q/K/V与SwiGLU形状已复核 |
| 041 | L153 · 头维度和 MLP 中间维度 | 段落 | 其中 \(W_g,W_u\in\mathbb R^{16\times32}\)，\(W_d\in\mathbb R^{32\times16}\)，\(\odot\) 表示逐元素乘法，\(\o | 本轮未新增硬错误；不代表出版通过 |
| 042 | L157 · 最小配置审计 | 段落 | 在远端节点上，先确认配置可以被解析，再启动训练： | 本轮未新增硬错误；不代表出版通过 |
| 043 | L159 · 最小配置审计 | 命令 | .venv/bin/python -c "from ttt.config import load_experiment; import json; print(json.dumps(load | 参数/路径已静态核对；远端执行待验 |
| 044 | L163 · 最小配置审计 | 段落 | 预期输出中应同时出现 model、inner、training 和 tokenizer，并包含补全后的派生字段。若出现 head 不整除、特殊 token 越界或 sequence leng | 本轮未新增硬错误；不代表出版通过 |
| 045 | L167 · 本章小结 | 段落 | - 配置合同同时约束参数 shape、状态边界、成本和比较公平性； | 本轮未新增硬错误；不代表出版通过 |

### docs/book/en/03_configuration.md

| 单元 | 位置与小节 | 类型 | 首句/首式 | 本轮处理 |
|---|---|---|---|---|
| 001 | L5 · Overview: give every number a source | 段落 | In tiny-contract-001, the vocabulary has 259 entries, the residual width is 16, and attention h | 本轮未新增硬错误；不代表出版通过 |
| 002 | L7 · Overview: give every number a source | 段落 | After this chapter, you should be able to load a configuration, list its resolved defaults, han | 本轮未新增硬错误；不代表出版通过 |
| 003 | L11 · Prerequisites and tiny target | 段落 | You only need the idea that a matrix maps one collection of numbers to another. The running inp | 本轮未新增硬错误；不代表出版通过 |
| 004 | L15 · Why configuration must be explicit | 段落 | Suppose a model silently chooses the number of heads from a tensor shape inside 'forward'. The  | 本轮未新增硬错误；不代表出版通过 |
| 005 | L19 · Vocabulary fixes the output space | 段落 | Let \(V\) be the tokenizer vocabulary size and \(d\) the hidden width. The input embedding is | 本轮未新增硬错误；不代表出版通过 |
| 006 | L21 · Vocabulary fixes the output space | 公式 | E\in\mathbb{R}^{V\times d}. | 本轮未新增硬错误；不代表出版通过 |
| 007 | L25 · Vocabulary fixes the output space | 段落 | At each position the residual stream is a vector in \(\mathbb{R}^{d}\). The MLP can temporarily | 本轮未新增硬错误；不代表出版通过 |
| 008 | L27 · Vocabulary fixes the output space | 公式 | \mathbf z_{b,t}=\mathbf h_{b,t}W_{\mathrm{out}},\qquad | 本轮未新增硬错误；不代表出版通过 |
| 009 | L32 · Vocabulary fixes the output space | 段落 | The current implementation has no output-projection bias. With tied embeddings, \(W_{\mathrm{ou | 本轮未新增硬错误；不代表出版通过 |
| 010 | L34 · Vocabulary fixes the output space | 段落 | For example, if \(V=259\) and \(d=16\), the tied embedding/head has \(259\times16=4{,}144\) sca | 本轮未新增硬错误；不代表出版通过 |
| 011 | L38 · Four groups of fields | 表格 | ／ Group ／ Examples ／ What changes ／ | 本轮未新增硬错误；不代表出版通过 |
| 012 | L45 · Four groups of fields | 段落 | The control C and adaptive variant F use the same static architecture. C disables inner updates | 本轮未新增硬错误；不代表出版通过 |
| 013 | L47 · Four groups of fields | 段落 | Sequence fields such as 'max_position_embeddings', 'sliding_window', and 'ttt_chunk_size' are a | 本轮未新增硬错误；不代表出版通过 |
| 014 | L51 · How the code consumes the contract | 段落 | 'ttt/models/ttt_e2e/configuration_ttt_e2e.py' defines model, MLP-prime, MoE, loop, and dtype fi | 本轮未新增硬错误；不代表出版通过 |
| 015 | L53 · How the code consumes the contract | 代码/示意 | from ttt.models import AutoConfig, AutoModelForCausalLM | 本轮未新增硬错误；不代表出版通过 |
| 016 | L60 · How the code consumes the contract | 段落 | Loading should validate positive dimensions, head divisibility, vocabulary ranges, layer positi | 本轮未新增硬错误；不代表出版通过 |
| 017 | L64 · Defaults and early errors | 段落 | Some fields are derived. 'num_key_value_heads=None' defaults to the query-head count, and 'head | 本轮未新增硬错误；不代表出版通过 |
| 018 | L66 · Defaults and early errors | 段落 | Examples of errors that should happen during Python initialization: | 本轮未新增硬错误；不代表出版通过 |
| 019 | L68 · Defaults and early errors | 段落 | - 'head_dim' is unspecified and 'hidden_size' is not divisible by 'num_attention_heads', so the | 本轮未新增硬错误；不代表出版通过 |
| 020 | L74 · Defaults and early errors | 段落 | An early 'ValueError' is cheaper and easier to diagnose than a compiler failure after several m | 本轮未新增硬错误；不代表出版通过 |
| 021 | L78 · A tiny shape audit | 段落 | Before launching training, write down the resolved values and trace one token: | 本轮未新增硬错误；不代表出版通过 |
| 022 | L80 · A tiny shape audit | 代码/示意 | vocab_size = 259 | 本轮未新增硬错误；不代表出版通过 |
| 023 | L94 · A tiny shape audit | 段落 | This is an audit extract, not loadable JSON. Model fields belong to model, sequence length belo | 本轮未新增硬错误；不代表出版通过 |
| 024 | L98 · Start with a two-feature lookup and projection | 段落 | A vector is an ordered collection of numbers; a matrix arranges numbers into rows and columns.  | 本轮未新增硬错误；不代表出版通过 |
| 025 | L100 · Start with a two-feature lookup and projection | 公式 | \mathbf E=\begin{pmatrix}1&0\\0&1\\2&1\\-1&1\end{pmatrix}. | 本轮未新增硬错误；不代表出版通过 |
| 026 | L104 · Start with a two-feature lookup and projection | 段落 | Input IDs 0 and 2 select rows \((1,0)\) and \((2,1)\), forming an array of shape \([2,2]\). The | 本轮未新增硬错误；不代表出版通过 |
| 027 | L106 · Start with a two-feature lookup and projection | 公式 | \mathbf h\mathbf E^{\mathsf T} | 本轮未新增硬错误；不代表出版通过 |
| 028 | L112 · Start with a two-feature lookup and projection | 段落 | Two features produce four candidate-token scores. Every position uses the same projection; addi | 本轮未新增硬错误；不代表出版通过 |
| 029 | L116 · Walk through the tiny dimensions | 段落 | Use the tiny model widths \(V=259\), \(d=16\), and \(d_{\mathrm{ff}}=32\), with a teaching batc | 本轮未新增硬错误；不代表出版通过 |
| 030 | L120 · Head dimensions and the MLP intermediate width | 段落 | The batched layout above explains how samples stack. The current model's forward handles one se | C01：根代理修订后已静态复核 |
| 031 | L122 · Head dimensions and the MLP intermediate width | 段落 | For tiny's \(d=16,n_h=2,n_{\mathrm{kv}}=1\), the default head width is \(d_h=8\). Query project | C02：Q/K/V与SwiGLU形状已复核 |
| 032 | L124 · Head dimensions and the MLP intermediate width | 段落 | SwiGLU's intermediate width of 32 also does not make its final output 32-dimensional. Each row  | 本轮未新增硬错误；不代表出版通过 |
| 033 | L126 · Head dimensions and the MLP intermediate width | 公式 | g=hW_g,\quad u=hW_u,\quad | C02：Q/K/V与SwiGLU形状已复核 |
| 034 | L131 · Head dimensions and the MLP intermediate width | 段落 | Here \(W_g,W_u\in\mathbb R^{16\times32}\), \(W_d\in\mathbb R^{32\times16}\), \(\odot\) denotes  | 本轮未新增硬错误；不代表出版通过 |
| 035 | L135 · Minimal configuration audit | 段落 | On the remote node, parse the resolved configuration before launching training: | 本轮未新增硬错误；不代表出版通过 |
| 036 | L137 · Minimal configuration audit | 命令 | .venv/bin/python -c "from ttt.config import load_experiment; import json; print(json.dumps(load | 参数/路径已静态核对；远端执行待验 |
| 037 | L141 · Minimal configuration audit | 段落 | The output should contain model, inner, training, and tokenizer, including resolved derived fie | 本轮未新增硬错误；不代表出版通过 |
| 038 | L145 · Summary | 段落 | - The configuration contract fixes parameter shapes, state boundaries, cost, and comparison fai | 本轮未新增硬错误；不代表出版通过 |

### docs/book/04_forward_and_ttt.md

| 单元 | 位置与小节 | 类型 | 首句/首式 | 本轮处理 |
|---|---|---|---|---|
| 001 | L5 · 先不要从 kernel 读模型 | 段落 | TTT-E2E 看起来包含很多新名词：prefix、suffix、fast weights、inner loss、MLP′、KV cache。最稳妥的读法不是先看 JAX scan 或 gr | 本轮未新增硬错误；不代表出版通过 |
| 002 | L7 · 先不要从 kernel 读模型 | 段落 | 模型首先把 token ID 查成 embedding。随后每一层将隐藏状态送进 attention 或 gated delta，再送入 MLP。后段层可能多一条 fast MLP′。最后的 | 本轮未新增硬错误；不代表出版通过 |
| 003 | L9 · 先不要从 kernel 读模型 | 代码/示意 | token ids | 本轮未新增硬错误；不代表出版通过 |
| 004 | L23 · 先不要从 kernel 读模型 | 段落 | 每个箭头都是形状、mask、cache 和梯度的接口。将 MLP′ 放在普通 MLP 前后，得到的算法并不相同；将更新发生在 token 内、chunk 末尾或下一个 chunk，状态语义也 | 本轮未新增硬错误；不代表出版通过 |
| 005 | L25 · 先不要从 kernel 读模型 | 图与图注 | 图 3　TTT 的两个时间尺度。 每个 chunk 先用当前快权重预测，再用已观测标签更新 \(\phi\)；\(\theta\) 则跨 batch 由外循环更新。 | 仅核对图注；视觉/渲染待验 |
| 006 | L32 · 当前代码的前向顺序 | 段落 | 'TTTE2EForCausalLM._block' 先执行 attention branch。若层类型是 full_attention 或 sliding_attention，则调用 at | 本轮未新增硬错误；不代表出版通过 |
| 007 | L34 · 当前代码的前向顺序 | 段落 | 若参数中包含 fast_mlp，代码取当前 fast weight 初始化的 MLP′，对 hidden 做 norm、SwiGLU 和可选 post norm，再加回 residual。之 | 本轮未新增硬错误；不代表出版通过 |
| 008 | L36 · 当前代码的前向顺序 | 段落 | 这意味着当前实现的概念顺序是： | 本轮未新增硬错误；不代表出版通过 |
| 009 | L38 · 当前代码的前向顺序 | 代码/示意 | Attention / Delta | 本轮未新增硬错误；不代表出版通过 |
| 010 | L44 · 当前代码的前向顺序 | 段落 | 这与“用 MLP′ 替换原 FFN”不同。普通 FFN 仍然存在，MLP′ 是附加的适应路径。若 fast weights 不存在，配置和参数树也不会凭空生成一条快路径。 | 本轮未新增硬错误；不代表出版通过 |
| 011 | L48 · Prefix 与 suffix 的作用 | 段落 | 配置将层分成 prefix 和 suffix。prefix 一次性运行静态层，产生每个位置的 hidden；suffix 使用这些 hidden 和可更新快权重继续计算。 | 本轮未新增硬错误；不代表出版通过 |
| 012 | L50 · Prefix 与 suffix 的作用 | 段落 | 这种划分减少了内循环需要重复的计算，但也带来一个约束：prefix 的 cache 和 hidden 必须与 suffix 的位置、mask 对齐。后缀不是一段独立文本，它接收的是同一序列中 | 本轮未新增硬错误；不代表出版通过 |
| 013 | L52 · Prefix 与 suffix 的作用 | 段落 | 'sequence_loss' 先补齐输入，并运行 model.prefix 得到隐藏表示，再将表示按 ttt_chunk_size 分块。suffix 在 scan 中逐块处理。快权重从  | 本轮未新增硬错误；不代表出版通过 |
| 014 | L56 · 为什么快权重不是普通参数 | 段落 | 普通参数由外循环 AdamW 更新，跨 batch、跨训练 step 保存。快权重在一个序列内由 inner loss 更新，可以在下一个 chunk 变化；是否跨 request 保留，则 | 本轮未新增硬错误；不代表出版通过 |
| 015 | L58 · 为什么快权重不是普通参数 | 段落 | 可以用两个时间索引区分： | 本轮未新增硬错误；不代表出版通过 |
| 016 | L60 · 为什么快权重不是普通参数 | 公式 | \theta_{k+1}=\mathrm{AdamW}(\theta_k,\nabla_\theta \mathcal{L}_{\mathrm{outer}}), | 本轮未新增硬错误；不代表出版通过 |
| 017 | L64 · 为什么快权重不是普通参数 | 公式 | \phi_{c+1}=\phi_c-\eta_{\mathrm{inner}} | 本轮未新增硬错误；不代表出版通过 |
| 018 | L69 · 为什么快权重不是普通参数 | 段落 | \(k\) 计数外循环训练步，\(c\) 计数序列内的完整更新块；\(X_c,Y_c,M_c\) 是该块的输入、目标和 mask。这里先写普通梯度下降形式，实际更新还有梯度裁剪，下一章展开。 | 本轮未新增硬错误；不代表出版通过 |
| 019 | L73 · 梯度边界和因果性 | 段落 | 一个安全的内循环顺序是： | 本轮未新增硬错误；不代表出版通过 |
| 020 | L75 · 梯度边界和因果性 | 代码/示意 | 使用当前 fast state 预测一个块的目标 | 本轮未新增硬错误；不代表出版通过 |
| 021 | L82 · 梯度边界和因果性 | 段落 | 如果当前预测的标签已经被用于修改产生该预测的状态，模型就得到了未来信息。因果性测试应修改较晚位置的 token，再比较它之前所有位置的 logits；较早预测不能因为未来标签改变。这里“较早 | 本轮未新增硬错误；不代表出版通过 |
| 022 | L84 · 梯度边界和因果性 | 段落 | 当前 inner_step 对 fast weights 使用 jax.value_and_grad，再调用 sgd_update；外循环继续对 sequence_loss 求导，计算图可穿 | 本轮未新增硬错误；不代表出版通过 |
| 023 | L88 · 一个手算的 chunk 例子 | 段落 | 设 sequence length 为 8，chunk size 为 4。下面的 0–7 是代码数组索引；按全书记号，数学位置对应 \(1\)–\(8\)。代码中的输入索引 0–7 被分成两 | 本轮未新增硬错误；不代表出版通过 |
| 024 | L90 · 一个手算的 chunk 例子 | 代码/示意 | chunk 0: positions 0,1,2,3 | 本轮未新增硬错误；不代表出版通过 |
| 025 | L95 · 一个手算的 chunk 例子 | 段落 | chunk 0 使用 &#36;\phi_0&#36; 产生 &#36;\mathcal{L}_0&#36;，并更新成 &#36;\phi_1&#36;。chunk 1 使用 &#36;\phi_1&#36; 产生 &#36;\mathcal{L}_1&#36;。另设序 | 本轮未新增硬错误；不代表出版通过 |
| 026 | L97 · 一个手算的 chunk 例子 | 段落 | 这个例子解释了为什么 chunk size 不是单纯的 batch 参数。它改变了快权重观察上下文的频率，也改变了内循环反向传播和 cache 的边界。 | 本轮未新增硬错误；不代表出版通过 |
| 027 | L101 · 如何验证前向不是“形状正确但语义错误” | 段落 | 至少做四种检查： | 本轮未新增硬错误；不代表出版通过 |
| 028 | L103 · 如何验证前向不是“形状正确但语义错误” | 段落 | 1. 分别核对无 fast MLP 的 dense 模型和保留模块但关闭更新的 C；二者结构不同，不要求 logits 相等； | 本轮未新增硬错误；不代表出版通过 |
| 029 | L108 · 如何验证前向不是“形状正确但语义错误” | 段落 | 这些检查是验证目标，不全由一条脚本覆盖。下面列出当前 GPU 脚本真正执行和记录的内容；其他对照应有独立证据。 | 本轮未新增硬错误；不代表出版通过 |
| 030 | L112 · 一个 token 的 shape walk | 段落 | 用 tiny 配置的 \(d=16\)、\(V=259\)、batch \(B=1\) 和位置数 \(L=4\) 走一遍： | 本轮未新增硬错误；不代表出版通过 |
| 031 | L114 · 一个 token 的 shape walk | 表格 | ／ 阶段 ／ 教学对象 ／ 批量形状 ／ 读者要检查的事实 ／ | 本轮未新增硬错误；不代表出版通过 |
| 032 | L122 · 一个 token 的 shape walk | 段落 | 这张表描述批量计算的概念布局。当前模型 forward 处理单序列 input_ids，输入形状为 \([L]\)，内部 hidden 为 \([L,d]\)，logits 为 \([L,V | C01：根代理修订后已静态复核 |
| 033 | L126 · 一个可计算的因果性断言 | 段落 | 对一个包含 4 个输入位置的序列，用同一初始参数和 mask 跑两次，只把输入索引 3 的 token 从 18 改为 19。索引 0、1、2 都在改变位置之前，因此三者的 logits 都 | 本轮未新增硬错误；不代表出版通过 |
| 034 | L128 · 一个可计算的因果性断言 | 代码/示意 | max_abs_diff(logits_a[0:3, :], logits_b[0:3, :]) <= tolerance | 本轮未新增硬错误；不代表出版通过 |
| 035 | L133 · 一个可计算的因果性断言 | 段落 | 这里 max_abs_diff 是“取逐元素差的绝对值再取最大值”的说明性记法，不是仓库 API。第一行失败意味着未来信息影响了较早预测；只有从索引 3 开始才允许变化。容差应与 dtype | 本轮未新增硬错误；不代表出版通过 |
| 036 | L137 · 本章最小验证 | 段落 | 在远端 GPU 节点运行 bash scripts/validate_gpu.sh outputs/gpu-validation.json。它执行 ttt/validate.py 中内置的  | 本轮未新增硬错误；不代表出版通过 |
| 037 | L139 · 本章最小验证 | 段落 | 成功报告的顶层是 environment、passed、cases。每个 case 有 name、tokens、loss、finite_gradients、logits_max_error、 | 本轮未新增硬错误；不代表出版通过 |
| 038 | L141 · 本章最小验证 | 段落 | 脚本不输出完整配置、mask、初始参数、state shape 或恢复比较；用例设定与随机 seed 在源码中，必须保存源码版本才能追溯。它要求至少一个可用 GPU，并不验证特定 GPU 型 | 本轮未新增硬错误；不代表出版通过 |
| 039 | L145 · 本章小结 | 段落 | - 前向顺序决定每条残差路径看到的输入和状态； | 本轮未新增硬错误；不代表出版通过 |

### docs/book/en/04_forward_and_ttt.md

| 单元 | 位置与小节 | 类型 | 首句/首式 | 本轮处理 |
|---|---|---|---|---|
| 001 | L5 · Do not start with a kernel | 段落 | TTT-E2E contains several new terms: prefix, suffix, fast weights, inner loss, MLP-prime, and KV | 本轮未新增硬错误；不代表出版通过 |
| 002 | L7 · Do not start with a kernel | 段落 | The model first looks up a token ID in an embedding table. Each block then sends the hidden sta | 本轮未新增硬错误；不代表出版通过 |
| 003 | L9 · Do not start with a kernel | 代码/示意 | token ids | 本轮未新增硬错误；不代表出版通过 |
| 004 | L23 · Do not start with a kernel | 段落 | Every arrow is an interface for a shape, a mask, a cache, and a gradient. Moving MLP-prime befo | 本轮未新增硬错误；不代表出版通过 |
| 005 | L25 · Do not start with a kernel | 图与图注 | Figure 3. Two clocks in TTT. Each chunk predicts with the current fast weights and updates \( \ | 仅核对图注；视觉/渲染待验 |
| 006 | L32 · The order implemented in this repository | 段落 | The forward path in TTTE2EForCausalLM._block first executes the attention branch. A layer whose | 本轮未新增硬错误；不代表出版通过 |
| 007 | L34 · The order implemented in this repository | 段落 | If the layer's parameter tree contains fast_mlp, the block reads the current fast-weight versio | 本轮未新增硬错误；不代表出版通过 |
| 008 | L36 · The order implemented in this repository | 段落 | The conceptual order is therefore: | 本轮未新增硬错误；不代表出版通过 |
| 009 | L38 · The order implemented in this repository | 代码/示意 | attention / gated delta | 本轮未新增硬错误；不代表出版通过 |
| 010 | L44 · The order implemented in this repository | 段落 | This is an additional adaptation path, not a replacement of the original feed-forward network.  | 本轮未新增硬错误；不代表出版通过 |
| 011 | L48 · Prefix and suffix | 段落 | The configuration divides the block stack into a prefix and a suffix. The prefix runs the stati | 本轮未新增硬错误；不代表出版通过 |
| 012 | L50 · Prefix and suffix | 段落 | This division avoids repeating all static computation inside the inner loop, but it creates an  | 本轮未新增硬错误；不代表出版通过 |
| 013 | L52 · Prefix and suffix | 段落 | The current sequence_loss pads inputs, runs model.prefix, then reshapes its hidden features int | 本轮未新增硬错误；不代表出版通过 |
| 014 | L56 · Why fast weights are not ordinary parameters | 段落 | Ordinary parameters are shared by many examples and updated by the outer optimizer. They surviv | 本轮未新增硬错误；不代表出版通过 |
| 015 | L58 · Why fast weights are not ordinary parameters | 段落 | Use two time indices to keep the clocks separate: | 本轮未新增硬错误；不代表出版通过 |
| 016 | L60 · Why fast weights are not ordinary parameters | 公式 | \theta_{k+1} | 本轮未新增硬错误；不代表出版通过 |
| 017 | L69 · Why fast weights are not ordinary parameters | 公式 | \phi_{c+1} | 本轮未新增硬错误；不代表出版通过 |
| 018 | L79 · Why fast weights are not ordinary parameters | 段落 | Here \(k\) counts outer training steps and \(c\) counts complete inner-update chunks; \(X_c,Y_c | 本轮未新增硬错误；不代表出版通过 |
| 019 | L83 · Prefix prediction, label use, and causality | 段落 | A safe causal update has this order: | 本轮未新增硬错误；不代表出版通过 |
| 020 | L85 · Prefix prediction, label use, and causality | 代码/示意 | use the current fast state to predict a chunk's targets | 本轮未新增硬错误；不代表出版通过 |
| 021 | L92 · Prefix prediction, label use, and causality | 段落 | The label used for an inner update must be available at the point where the update happens. If  | 本轮未新增硬错误；不代表出版通过 |
| 022 | L94 · Prefix prediction, label use, and causality | 段落 | The test is simple to state: change a later token and compare logits at every earlier position. | 本轮未新增硬错误；不代表出版通过 |
| 023 | L96 · Prefix prediction, label use, and causality | 段落 | The inner step uses jax.value_and_grad with respect to fast weights, then applies sgd_update. T | 本轮未新增硬错误；不代表出版通过 |
| 024 | L100 · A small chunk calculation | 段落 | Let the sequence length be \(8\) and the chunk size be \(4\). The numbers \(0\) through \(7\) b | 本轮未新增硬错误；不代表出版通过 |
| 025 | L102 · A small chunk calculation | 代码/示意 | chunk 0: positions 0, 1, 2, 3 | 本轮未新增硬错误；不代表出版通过 |
| 026 | L107 · A small chunk calculation | 段落 | Chunk 0 predicts with \(\phi_0\), obtains \(\mathcal{L}_0\), and updates to \(\phi_1\). Chunk 1 | 本轮未新增硬错误；不代表出版通过 |
| 027 | L109 · A small chunk calculation | 段落 | This is why chunk size is an algorithmic variable, not merely a batching detail. A smaller chun | 本轮未新增硬错误；不代表出版通过 |
| 028 | L113 · How to validate that the forward pass is semantically correct | 段落 | A useful validation set contains at least four checks: | 本轮未新增硬错误；不代表出版通过 |
| 029 | L115 · How to validate that the forward pass is semantically correct | 段落 | 1. Check the dense model without fast MLP and C with the module present but updates disabled se | 本轮未新增硬错误；不代表出版通过 |
| 030 | L120 · How to validate that the forward pass is semantically correct | 段落 | These are validation goals, not a claim that one script implements them all. The next section l | 本轮未新增硬错误；不代表出版通过 |
| 031 | L124 · Shape walk for one token | 段落 | Use the tiny configuration with \(d=16\), \(V=259\), batch \(B=1\), and \(L=4\) positions: | 本轮未新增硬错误；不代表出版通过 |
| 032 | L126 · Shape walk for one token | 表格 | ／ Stage ／ Teaching object ／ Batched shape ／ Check ／ | 本轮未新增硬错误；不代表出版通过 |
| 033 | L134 · Shape walk for one token | 段落 | This table describes the conceptual batched layout. The current model's forward takes single-se | C01：根代理修订后已静态复核 |
| 034 | L138 · A numerical causality assertion | 段落 | For a sequence with four input positions, run twice with identical initial parameters and mask, | 本轮未新增硬错误；不代表出版通过 |
| 035 | L140 · A numerical causality assertion | 代码/示意 | max_abs_diff(logits_a[0:3, :], logits_b[0:3, :]) <= tolerance | 本轮未新增硬错误；不代表出版通过 |
| 036 | L145 · A numerical causality assertion | 段落 | Here max_abs_diff means the maximum elementwise absolute difference; it is explanatory notation | 本轮未新增硬错误；不代表出版通过 |
| 037 | L149 · Minimal validation | 段落 | On a remote GPU node, run bash scripts/validate_gpu.sh outputs/gpu-validation.json. This execut | 本轮未新增硬错误；不代表出版通过 |
| 038 | L151 · Minimal validation | 段落 | The successful report has top-level environment, passed, and cases. Each case records name, tok | 本轮未新增硬错误；不代表出版通过 |
| 039 | L153 · Minimal validation | 段落 | The script does not export full configuration, masks, initial parameters, state shapes, or rest | 本轮未新增硬错误；不代表出版通过 |
| 040 | L157 · Summary | 段落 | - Forward order determines what each residual path and state can observe. | 本轮未新增硬错误；不代表出版通过 |

共记录 394 个正文块。未修改上述10页正文，未执行任何本地测试，未提交。新增记录的静态空白检查见根代理最终检查；它不能验证运行或出版质量。
