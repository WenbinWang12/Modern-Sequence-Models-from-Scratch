# Frontier 中英文逐段审稿记录

审阅日期：2026-11-11。源码基线：567fef3。范围：frontier 下四个专题的中英文八页。此记录是两轮源码与书稿审查的证据，不是出版通过证书，也不是实验通过报告。

第一轮检查从旧稿的“复现中”标签、概念概要和未来计划入手，重建问题、推导、代码对应与实验协议。第二轮逐式、逐命令检查真实实现，补上查询向量、梯度裁剪、Loop 执行顺序与不满块行为。中英文保留相同的论文版本、公式、命令和证据边界。

本轮没有运行本地或远端测试。仅执行源码阅读、资料核对与 git diff --check；代码片段仍需在获授权的远端机器产生原始报告。出版准备还需要独立复审、网页数学与视觉检查、数据/训练实验，以及跨章叙事审查。不得把本记录解释为“已经达到可发表标准”。

## 来源核验

以下版本在 arXiv 版本页核对。书稿引用只概述相关研究问题，并没有证明论文的所有定理，也没有复现其训练设置。

| 来源 | 核对版本和日期 | 本次采用的范围 |
|---|---|---|
| [Learning to (Learn at Test Time)](https://arxiv.org/abs/2407.04620v4) | v4，2025-08-31 | 参数作为状态，TTT-Linear/MLP |
| [Test-Time Training with KV Binding Is Secretly Linear Attention](https://arxiv.org/abs/2602.21204v4) | v4，2026-05-12 | KV-binding 条件化关系，不泛化到所有 TTT |
| [End-to-End Test-Time Training for Long Context](https://arxiv.org/abs/2512.23675v2) | v2，2025-12-31 | next-token 测试时更新；3B/164B/128K 为作者设置 |
| [Reasoning with Latent Thoughts](https://arxiv.org/abs/2502.17416v1) | v1，2025-02-24 | 共享参数深度循环研究问题；局部 loop 不是完整复现 |

## 逐段定位方法

B 编号按当前 Markdown 的空行分隔块计数，标题也参与编号；段落、显示公式、表格与完整 fenced command 各算一块。每行同时给出中英文首句片段与源文件行号。标题只检查层次，非标题内容逐项记录如下。行号对应本轮提交前的稿件，后续增删以首句与 B 编号共同定位。

## index：中英逐块对照

文件：[中文](../../docs/frontier/index.md)、[English](../../docs/frontier/en/index.md)。标题顺序均为动机→来源/主张→数学或实现→最小运行→对照/决策，标题块未发现层级跳跃。

| 块与定位 | 首句/式/命令（中文；English） | 审稿问题、修复证据与未核验项 |
|---|---|---|
| B2，中 L3 / 英 L3，paragraph | 资料快照：2026-11-11。代码审阅基线：567fef3。状态：研究方案与局部实现；未宣称完成论文复现。<br>Source snapshot: 2026-11-11. Code review baseline: 567f… | P1：用固定日期、审阅提交和“未宣称复现”替代笼统“复现中”；这里的状态仅为源码审阅。 |
| B3，中 L5 / 英 L5，paragraph | 一个新方法在论文中获得更低的损失，并不说明把它的模块加入现有模型就会得到同样的收益。数据、训练预算、状态更新顺…<br>A lower loss in a paper does not imply that inserting i… | P2：补上研究主张为何不能直接移植的因果链，区分数学、实现、质量三类证据。 |
| B4，中 L7 / 英 L7，paragraph | 本部分依赖主线中的 next-token 损失、梯度、因果注意力和实验记录。如果暂时看不懂下面的公式，先读\[基…<br>This part relies on next-token loss, gradients, causal … | P2：列明 next-token/梯度/因果注意力前置知识并链接主线；明确最小运行不是练习或结果表。 |
| B6，中 L11 / 英 L11，paragraph | 论文链接固定到版本，避免网页更新后同一引用指向不同论证。表中的状态来自代码阅读，不等同于本轮执行通过。<br>Paper links are pinned to versions so a later update do… | P1：引用固定版本；避免把静态审阅误写为本轮运行通过。 |
| B7，中 L13 / 英 L13，table | ¦ 专题 ¦ 版本化来源 ¦ 要检验的主张 ¦ 当前状态 ¦ ¦---¦---¦---¦---¦ ¦ \[TTT…<br>¦ Topic ¦ Versioned source ¦ Question ¦ Current status … | P1：三行均对齐版本化论文、中文/英文专题链接和当前代码范围；不把三者均写为复现成功。 |
| B8，中 L19 / 英 L19，paragraph | 已实现（implemented）表示存在实际执行路径；实验框架（scaffold）表示有入口或配置，但尚缺完整…<br>Implemented means an executable code path exists. Scaff… | P1：逐项定义 Implemented/Scaffold/Planned，避免一项能力存在就给整个研究问题完成标签。 |
| B10，中 L23 / 英 L23，paragraph | 主线教读者建立可靠的实验系统，前沿专题则询问系统应尝试什么新假设。例如，TTT 与 Loop 都增加计算，但前…<br>The main book builds a trustworthy experiment system. F… | P2：用状态变化轴解释 TTT 与 Loop 的差别；“思考更久”仅是直觉，不能代替具体更新定义。 |
| B11，中 L25 / 英 L25，paragraph | 因此每个专题先给出最小问题，再建立对应实现。一个单 token 代数恒等式可以检查更新规则；一个 tiny 模…<br>Each topic therefore starts with a small question and a… | P1：明确代数/功能/质量证据的递进；单 token 例子不具有模型质量效力。 |
| B13，中 L29 / 英 L29，paragraph | 以下命令在已经按\[环境章节\](../book/01_environment.md)安装依赖的远端 GPU 节点…<br>Run these commands from the repository root on a remote… | P1：明确远端 GPU、依赖和仓库根目录，CPU 也仅远端；本轮没有执行声明。 |
| B14，中 L31 / 英 L31，command | mkdir -p outputs/frontier-smoke git rev-parse HEAD > ou…<br>mkdir -p outputs/frontier-smoke git rev-parse HEAD > ou… | P1：命令与 scripts/validate_gpu.sh 实际接口一致，输出参数真实；增加 source-commit；待远端执行。 |
| B15，中 L37 / 英 L37，paragraph | 脚本使用程序内定义的 tiny 架构，默认上下文检查为 64 和 512 token，并包含 Gated De…<br>The script uses tiny architectures defined in code, wit… | P1：按 ttt/validate.py 校正架构来源、长度及用例；特别标出 dense 用例仍可含快权重，不能充当禁用 TTT 训练基线。 |
| B16，中 L39 / 英 L39，paragraph | 成功时应得到 environment.json、source-commit.txt 和 validation.…<br>Expected artifacts are environment.json, source-commit.… | P1：按 validate_gpu.sh/ttt.validate 返回字段列产物；因果检查是断言而非不存在的 JSON 字段。 |
| B17，中 L41 / 英 L41，paragraph | 失败时先检查 error_type 与 error，再查环境中的 GPU/后端和失败用例。进程在报告写入前被杀…<br>On failure, inspect error_type and error, then the back… | P2：补错误字段和进程被杀导致无报告的处理；文件缺失不算成功。 |
| B19，中 L45 / 英 L45，paragraph | 质量实验固定数据切分、tokenizer、初始化策略、有效 token 数和优化器。成本实验还要固定硬件、精度…<br>Quality experiments fix data splits, tokenizer, initial… | P1：区分相同参数、FLOPs、wall time；补数据与优化器控制，禁止混用预算。 |
| B20，中 L47 / 英 L47，table | ¦ 证据层 ¦ 需要的指标 ¦ 通过后能说什么 ¦ 仍不能说什么 ¦ ¦---¦---¦---¦---¦ ¦ …<br>¦ Evidence level ¦ Required measurements ¦ Supported co… | P1：四行分别约束可支持/不可支持结论；没有虚构任务分数。 |
| B21，中 L54 / 英 L54，paragraph | 每个正式结果还应指向配置、数据指纹和原始报告。当前专题没有新增实测结果；空缺用“未运行”表示，不填写预测数字。…<br>Every formal result must also identify its configuratio… | 待核验：缺本协议原始报告，标明未运行；下一步为各专题最小检查，不能据此宣布可发表。 |

## ttt_linear_attention：中英逐块对照

文件：[中文](../../docs/frontier/ttt_linear_attention.md)、[English](../../docs/frontier/en/ttt_linear_attention.md)。标题顺序均为动机→来源/主张→数学或实现→最小运行→对照/决策，标题块未发现层级跳跃。

| 块与定位 | 首句/式/命令（中文；English） | 审稿问题、修复证据与未核验项 |
|---|---|---|
| B2，中 L3 / 英 L3，paragraph | 资料快照：2026-11-11。代码审阅基线：567fef3。状态：局部代数示例可执行；独立模型等价性与质量实…<br>Source snapshot: 2026-11-11. Code review baseline: 567f… | P1：状态缩窄到可执行教学片段；独立训练比较尚未实现，不延续整体“复现中”。 |
| B4，中 L7 / 英 L7，paragraph | 设模型刚读到“门牌号是 42”，后面需要回答门牌号。注意力可以保存先前位置的 key/value；另一种做法是…<br>Suppose a model reads “the door number is 42” and must … | P2：门牌号例子只引出关联记忆，随后列出会破坏等价的读写/损失/归一化条件。 |
| B5，中 L9 / 英 L9，paragraph | 本节先研究一个可手算的线性模型。读完后应能指出哪一步产生了 delta rule，以及为什么这不能直接证明仓库…<br>We first study a linear model small enough to calculate… | P2：说明读者目标并链接 next-token 交叉熵背景；与平方重建的关系由后文推导承担。 |
| B7，中 L13 / 英 L13，paragraph | \[Learning to (Learn at Test Time), v4，2025-08-31\](https…<br>\[Learning to (Learn at Test Time), v4, 2025-08-31\](http… | P1：核对 2407.04620v4 与 2602.21204v4 的题目、版本、日期；二者均在中英出现。 |
| B8，中 L15 / 英 L15，paragraph | 这里检验的是受明确假设限制的更新关系。不能把论文标题扩大为“所有梯度更新都等于标准线性注意力”，更不能跳过从论…<br>Our check concerns an update under explicit assumptions… | P1：限制 paper-title 推论范围，不把 KV-binding 结论推广到全部 TTT-E2E。 |
| B10，中 L19 / 英 L19，paragraph | 令列向量 \(k_t\in\mathbb R^{d_k}\)、\(v_t\in\mathbb R^{d_v}\…<br>Let column vectors \(k_t\in\mathbb R^{d_k}\) and \(v_t\… | P1：补列向量、矩阵维度和词表 V/值向量 v 的区分，后续外积方向有明确定义。 |
| B11，中 L21 / 英 L21，formula | \ell_t(W)=\frac12\¦Wk_t-v_t\¦_2^2. $$<br>\ell_t(W)=\frac12\¦Wk_t-v_t\¦_2^2. $$ | 公式 F1：平方损失含 1/2；未冒称本地 token CE。数学检查完成，代码执行待核验。 |
| B12，中 L25 / 英 L25，paragraph | 把残差记为 \(e_t=Wk_t-v_t\)。矩阵元素 \(W_{ij}\) 只影响残差的第 \(i\) 项，…<br>Write the residual as \(e_t=Wk_t-v_t\). Element \(W_{ij… | P2：用矩阵单元素偏导解释外积梯度，避免只给最终公式。 |
| B13，中 L28 / 英 L28，formula | \nabla_W\ell_t=(Wk_t-v_t)k_t^\mathsf T. $$<br>\nabla_W\ell_t=(Wk_t-v_t)k_t^\mathsf T. $$ | 公式 F2：残差维度 dv 与 key 维度 dk 外积得到 dv×dk，与 W 一致。 |
| B14，中 L32 / 英 L32，paragraph | 一步学习率为 \(\eta\) 的 SGD 更新是<br>One SGD step with learning rate \(\eta\) is | P2：显式说明 eta 为本教学 SGD 学习率，不混用外循环 optimizer。 |
| B15，中 L34 / 英 L34，formula | W_t=W_{t-1}-\eta(W_{t-1}k_t-v_t)k_t^\mathsf T =W_{t-1}(…<br>W_t=W_{t-1}-\eta(W_{t-1}k_t-v_t)k_t^\mathsf T =W_{t-1}(… | 公式 F3：展开得到 W(I−eta kkᵀ)+eta vkᵀ；与梯度形式逐项对应。 |
| B16，中 L39 / 英 L39，paragraph | 右侧第一项先修正原状态，第二项写入新关联。这是线性快模型在上述损失下的 delta 更新。相比之下，简单的加法…<br>The first term corrects existing state; the second writ… | P1：区分 delta 与最简单加法状态；未证明它与带特征映射/门控的所有线性注意力不同。 |
| B17，中 L42 / 英 L41，paragraph | 取 \(W_0=0\)、\(k_1=(1,0)^\mathsf T\)、\(v_1=(2,-1)^\maths…<br>For \(W_0=0\), \(k_1=(1,0)^\mathsf T\), \(v_1=(2,-1)^\m… | P2：给定完整初态、输入和学习率，使读者能手算。 |
| B18，中 L44 / 英 L43，formula | W_1=\begin{pmatrix}1&0\\-0.5&0\end{pmatrix}. $$<br>W_1=\begin{pmatrix}1&0\\-0.5&0\end{pmatrix}. $$ | 公式 F4：首步得到 [[1,0],[-0.5,0]]；首步零初态不能揭示误差修正差异。 |
| B19，中 L48 / 英 L47，paragraph | 再使用同一个 key 和 \(v_2=(0,1)^\mathsf T\)，delta 更新得到 \(W_2=\…<br>Using the same key again with \(v_2=(0,1)^\mathsf T\) g… | P1：补第二步 delta [[0.5,0],[0.25,0]] 与缩放加法 [[1,0],[0,0]] 的反例，防止只检查第一步。 |
| B20，中 L52 / 英 L51，paragraph | 若查询列向量为 \(q_t\in\mathbb R^{d_k}\)，可用更新后的 \(W_t\) 读出 \(W…<br>For a query column vector \(q_t\in\mathbb R^{d_k}\), re… | 二轮修复：补 q 的维度与先写后读定义；明确局部 TTT-E2E chunk 是先评分再更新。 |
| B22，中 L56 / 英 L55，table | ¦ 能力 ¦ 状态 ¦ 真实入口与限制 ¦ ¦---¦---¦---¦ ¦ 带门控 delta 递推 ¦ 已实…<br>¦ Capability ¦ Status ¦ Entry point and limit ¦ ¦---¦--… | P1：源码映射 ttt/inner_loop.py 与 gated_delta.py；后者还有卷积、归一化、衰减、门控，不能说实现了纯线性教学式。 |
| B24，中 L65 / 英 L64，paragraph | 登录远端 CPU 机器后，在已安装 NumPy 的环境运行下面的完整片段。它检查两种写法是否相同，并检查与加法…<br>After logging into the remote CPU machine, run this com… | P1：远端 CPU + NumPy 前提明确；片段未执行。 |
| B25，中 L67 / 英 L66，command | mkdir -p outputs/frontier-linear python - <<'PY' import…<br>mkdir -p outputs/frontier-linear python - <<'PY' import… | 命令 C1：自包含两个更新路径、两步检查和失败断言；产物 algebra.json；中英代码逐字相同，待远端执行。 |
| B26，中 L96 / 英 L95，paragraph | 成功时两次 max_errors 应接近零，delta_state 与手算一致。若失败，先检查列向量约定、外积…<br>Both max_errors should be near zero and delta_state sho… | P1：预期值来自手算而非伪装实测；失败定位为方向/时序；不冒充语言质量或论文并行形式检验。 |
| B28，中 L100 / 英 L99，paragraph | 完整实验应先固定 \(d_k,d_v\)、初态、key/value/query、学习率、loss 和读写顺序。…<br>The complete experiment must first fix \(d_k,d_v\), ini… | P1：列明状态维度、初态、输入、loss、时序等必要控制；多组件变更不可单独归因。 |
| B29，中 L102 / 英 L101，table | ¦ 问题 ¦ 判据 ¦ 当前记录 ¦ ¦---¦---¦---¦ ¦ 两步 SGD 与展开递推是否相同？ ¦ …<br>¦ Question ¦ Criterion ¦ Current record ¦ ¦---¦---¦---¦… | P1：误差阈值与代码 assert 一致；Gated Delta 对照和训练质量都标为未建立/未运行。 |
| B30，中 L108 / 英 L107，paragraph | 任何一步若使用不同初态、mask 或更新时序，应判为协议不匹配，而不是方法被证伪。下一次先保存代数报告，再实现…<br>Different initial states, masks, or update schedules in… | 待核验：保存远端代数报告、实现两条独立模型路径；协议不一致不能直接说方法被证伪。 |

## ttt_e2e_long_context：中英逐块对照

文件：[中文](../../docs/frontier/ttt_e2e_long_context.md)、[English](../../docs/frontier/en/ttt_e2e_long_context.md)。标题顺序均为动机→来源/主张→数学或实现→最小运行→对照/决策，标题块未发现层级跳跃。

| 块与定位 | 首句/式/命令（中文；English） | 审稿问题、修复证据与未核验项 |
|---|---|---|
| B2，中 L3 / 英 L3，paragraph | 资料快照：2026-11-11。代码审阅基线：567fef3。状态：训练与运行时已实现；32K/128K 质量…<br>Source snapshot: 2026-11-11. Code review baseline: 567f… | P1：将运行时已实现与 32K/128K 质量未完成分开；不混同整体复现。 |
| B4，中 L7 / 英 L7，paragraph | 假设答案位于一本书的开头，问题出现在结尾。模型即使成功接受整本书，也可能丢掉答案；即使生成正确答案，也可能是训…<br>Suppose an answer appears at the start of a book and th… | P2：从书首答案例子拆出历史利用、污染、泄漏和资源问题，不把能输入视为能力证明。 |
| B5，中 L9 / 英 L9，paragraph | 本节沿用\[内外循环\](../book/05_inner_outer_loop.md)和\[KV cache\](.…<br>This section uses the state definitions from \[inner and… | P2：前置链接覆盖内外循环和 KV 状态；明确本节是证据链，不是保证可运行的 GPU 型号清单。 |
| B7，中 L13 / 英 L13，paragraph | \[End-to-End Test-Time Training for Long Context, v2，202…<br>\[End-to-End Test-Time Training for Long Context, v2, 20… | P1：固定 2512.23675v2，3B/164B/128K 均归属原论文实验，没有当成本仓库测量。 |
| B8，中 L15 / 英 L15，paragraph | 本地待检验假设是：相同模型和 checkpoint，在相同上下文上启用更新后，后续未见 token 的预测或任…<br>The local hypothesis is that enabling updates for the s… | P1：适应开/关是同 checkpoint 配对评估，不能替代分别训练的方法比较。 |
| B10，中 L19 / 英 L19，paragraph | 设输入被切成 chunk \(c\)，静态参数为 \(\theta\)，快权重为 \(\phi_c\)。\(X…<br>Divide inputs into chunks \(c\), with static parameters… | 二轮修复：定义 Xc/Yc/Mc 和参数状态，说明主公式只适用于未触发裁剪时。 |
| B11，中 L21 / 英 L21，formula | \mathcal L_c=\mathcal L(X_c,Y_c,M_c;\theta,\phi_c),\qqu…<br>\mathcal L_c=\mathcal L(X_c,Y_c,M_c;\theta,\phi_c),\qqu… | 公式 F1：chunk 先评分后更新；phi 下标 c 不与 token t 混用。 |
| B12，中 L26 / 英 L26，paragraph | 实际 sgd_update 还把梯度乘以 \(\min(1,\tau/\sqrt{\¦g\¦_2^2+10^{…<br>The actual sgd_update also multiplies the gradient by \… | 二轮修复：根据 inner_loop.sgd_update 补真实梯度裁剪 min(1,tau/sqrt(norm²+1e−12))；标签不得先用于自己的评分。 |
| B13，中 L28 / 英 L28，paragraph | 快权重维度可以固定，但这不意味着整个模型的内存固定：还必须统计 KV、prefix 激活、卷积状态、未完成 c…<br>Fixed fast-weight dimensions do not imply fixed whole-m… | P1：固定快权重不推出总内存固定，列 KV/激活/卷积/未完成块/缓冲。 |
| B14，中 L30 / 英 L30，paragraph | 一个 full-attention 层的理想 KV 存储约为<br>For a single sample, ideal KV storage for one full-atte… | P2：内存式范围限定单样本单个 full-attention 层。 |
| B15，中 L32 / 英 L32，formula | M_{\mathrm{KV}}=2L n_{\mathrm{kv}}d_h b_{\mathrm{dtype}…<br>M_{\mathrm{KV}}=2L n_{\mathrm{kv}}d_h b_{\mathrm{dtype}… | 公式 F2：2×L×KV头数×head维度×字节数，仅 K 与 V；不是峰值显存。 |
| B16，中 L36 / 英 L36，paragraph | 这里暂取单样本，\(n_{\mathrm{kv}}\) 为 KV 头数，\(b_{\mathrm{dtype}…<br>where \(n_{\mathrm{kv}}\) counts KV heads and \(b_{\mat… | P1：手算 32K=96 MiB/层、128K=384 MiB/层；明确精度与维度假设，不含权重/梯度/激活。 |
| B17，中 L38 / 英 L38，paragraph | 滑窗层可限制 KV 历史，但实际分配还取决于实现。当前 configs/runs/tiny_gpu.json …<br>Sliding-window layers can bound retained KV history, bu… | P1：configs/runs/tiny_gpu.json 仍有 full_attention 前缀；不能由滑窗后缀推出整模型长度无关延迟。 |
| B19，中 L42 / 英 L42，table | ¦ 项目 ¦ 状态 ¦ 代码与范围 ¦ ¦---¦---¦---¦ ¦ chunk 预测与梯度更新 ¦ 已实现…<br>¦ Item ¦ Status ¦ Code and scope ¦ ¦---¦---¦---¦ ¦ Chun… | P1：入口逐一对应 inner_loop/runtime/validate/eval_retrieval；不满块仅评分不提交更新；自定义 probe 不是完整 RULER。 |
| B20，中 L50 / 英 L50，paragraph | 从零训练本地模型不需要官方权重；只有官方 checkpoint 对齐实验才需要取得相应文件、访问权限、toke…<br>Local from-scratch training does not require official w… | P1：从零训练不依赖官方权重，官方 checkpoint 对齐另列访问/tokenizer/转换要求。 |
| B22，中 L54 / 英 L54，paragraph | 先在准备好的远端 GPU 节点运行功能检查：<br>First run functional checks on a prepared remote GPU no… | P1：命令仅在准备好的远端 GPU 节点执行，未隐含本地测试。 |
| B23，中 L56 / 英 L56，command | mkdir -p outputs/frontier-context git rev-parse HEAD > …<br>mkdir -p outputs/frontier-context git rev-parse HEAD > … | 命令 C1：ttt.validate 实际 --require-gpu/--lengths/--output 参数；固定64/512，记提交；未执行。 |
| B24，中 L63 / 英 L63，paragraph | 报告应有 passed: true，以及 context_64、context_512 的有限梯度、strea…<br>Expect passed: true and cases context_64 and context_51… | P1：阈值 1e−3、context_*、finite_gradients、state_bytes 与 validate.py 对齐；错误字段不虚构。 |
| B25，中 L65 / 英 L65，paragraph | 完成训练并有一个真实 checkpoint 后，下面是配对评估模板。两个路径必须替换为已有文件；这里没有暗示它…<br>After training produces a real checkpoint, the followin… | P1：checkpoint 与 heldout path 明确标为待替换模板，不假装随仓库生成。 |
| B26，中 L67 / 英 L67，command | CHECKPOINT=/absolute/path/to/trained-checkpoint VALIDAT…<br>CHECKPOINT=/absolute/path/to/trained-checkpoint VALIDAT… | 命令 C2：eval_retrieval.py 的 loss/sample/sequence-length/no-adapt 参数真实；两个命令仅更新开关不同；待真实输入执行。 |
| B27，中 L78 / 英 L78，paragraph | 两个输出的差异只回答该 checkpoint 在该采样协议下启用更新的效果。验证数据不能用训练用的 sampl…<br>The difference answers only the effect of enabling upda… | P1：sample 训练文本不能充验证；8样本仅测入口，不代表稳定质量；不绕过位置/词表校验。 |
| B29，中 L82 / 英 L82，table | ¦ 阶段 ¦ 必须已有的证据 ¦ 常见失败 ¦ 下一步 ¦ ¦---¦---¦---¦---¦ ¦ tiny，…<br>¦ Stage ¦ Evidence required first ¦ Typical failure ¦ N… | P1：32K→128K 准入含位置、数据、编译、内存、质量和多 seed；记录失败而非外推。 |
| B30，中 L89 / 英 L89，paragraph | 这里不写“某张 GPU 必定足够”：精度、全注意力层数、batch、编译策略和训练/推理模式都会改变门槛。把 …<br>No GPU is declared “certainly sufficient”: precision, f… | P1：不给未经测量的显卡容量保证；扩大 tiny --lengths 不是 3B 复现。 |
| B32，中 L93 / 英 L93，paragraph | 同一表中记录有效 token、held-out loss、检索分数、预热后 prefill/decode 时间…<br>Record valid tokens, held-out loss, retrieval scores, w… | P1：分开有效 token/质量/编译/稳态/峰值/状态；ttt.profile 合成输入不测质量。 |
| B33，中 L95 / 英 L95，table | ¦ 研究结论 ¦ 当前证据 ¦ 决策 ¦ ¦---¦---¦---¦ ¦ 本地实现是否支持 TTT 训练与流式…<br>¦ Research conclusion ¦ Current evidence ¦ Decision ¦ ¦… | P1：三项决策明确有入口而无本次结果，32K/128K 未决，3B 不作复现声明。 |
| B34，中 L101 / 英 L101，paragraph | 下一次先保存 64/512 的原始报告，再选择一个已训练小模型做适应开关的配对 held-out 评估。只有这…<br>The next run saves the raw 64/512 report, then selects … | 待核验：原始64/512报告与小模型 heldout 配对结果尚缺；两者可解释后才扩大长度。 |

## loop_transformer：中英逐块对照

文件：[中文](../../docs/frontier/loop_transformer.md)、[English](../../docs/frontier/en/loop_transformer.md)。标题顺序均为动机→来源/主张→数学或实现→最小运行→对照/决策，标题块未发现层级跳跃。

| 块与定位 | 首句/式/命令（中文；English） | 审稿问题、修复证据与未核验项 |
|---|---|---|
| B2，中 L3 / 英 L3，paragraph | 资料快照：2026-11-11。代码审阅基线：567fef3。状态：三种共享参数的局部重复已实现；论文协议和质…<br>Source snapshot: 2026-11-11. Code review baseline: 567f… | P1：明确三种共享重复已实现，论文协议/质量未复现。 |
| B4，中 L7 / 英 L7，paragraph | 修改文章时，第二遍阅读可能发现第一遍遗漏的问题。共享参数的深度循环也有类似的直觉：同一个变换再次处理已经改变的…<br>A second reading of a draft may reveal a problem missed… | P2：修改文章比喻同时讲失效条件，避免重复=推理的拟人化推论。 |
| B5，中 L9 / 英 L9，paragraph | 普通深层 Transformer 为不同层保存不同参数。Looped Transformer 则在某段深度上重…<br>An ordinary deep Transformer stores different parameter… | P1：分离参数、计算与激活成本；问题限定到指定预算。 |
| B7，中 L13 / 英 L13，paragraph | 本节固定引用 \[Reasoning with Latent Thoughts: On the Power of…<br>This section pins \[Reasoning with Latent Thoughts: On t… | P1：固定2502.17416v1及日期；局部实验不冒称完整架构复现。 |
| B8，中 L15 / 英 L15，paragraph | 设某个 token 位置的初始表示为 \(h^{\[0\]}\)，一个共享模块为 \(F_\theta\)。深度方…<br>Let the initial representation at a token position be \… | P2：先定义 h/Ftheta/r，再给深度递推。 |
| B9，中 L17 / 英 L17，formula | h^{\[j+1\]}=F_\theta(h^{\[j\]}),\qquad j=0,\ldots,r-1. $$<br>h^{\[j+1\]}=F_\theta(h^{\[j\]}),\qquad j=0,\ldots,r-1. $$ | 公式 F1：共享 Ftheta 重复 r 次，数学上参数不随 r 复制。 |
| B10，中 L21 / 英 L21，paragraph | 这里 \(j\) 是重复次数，token 位置仍用 \(t\)；\(F_\theta\) 可以处理整段序列，因…<br>Here \(j\) counts repetitions; token position is still … | P1：j 是深度重复，t 是 token；定义被省略的序列维，区分参数与激活共享。 |
| B11，中 L23 / 英 L23，paragraph | TTT 则沿序列推进更新快权重 \(\phi_c\)，其中 \(c\) 表示 chunk。Loop 改变深度计…<br>TTT instead updates fast weights \(\phi_c\) as chunks \… | P1：phi_c 沿chunk更新，Loop沿深度；先独立对照再研究组合。 |
| B13，中 L27 / 英 L27，paragraph | ttt/models/ttt_e2e/modeling_ttt_e2e.py 对下表的重复使用同一个参数对象。…<br>ttt/models/ttt_e2e/modeling_ttt_e2e.py repeats the same… | P1修正事实错误：旧文说真正共享层尚待实现；_run_layer/_block 现在已复用相同参数对象。 |
| B14，中 L29 / 英 L29，table | ¦ 配置 ¦ 已实现的计算 ¦ 重复的范围 ¦ ¦---¦---¦---¦ ¦ none ¦ 每层执行一次 ¦…<br>¦ Configuration ¦ Implemented computation ¦ Repeated sc… | P1：none/early_ffn/middle_attention/middle_block 四行对应实际源码，不把不同范围都叫完整 stack loop。 |
| B15，中 L36 / 英 L36，paragraph | 特别要区分“重复每一层”和“重复一段层”。选择第 0、1 层且重复两次时，当前 middle_block 的顺…<br>Repeating each layer differs from repeating a stack of … | P1：给0→0→1→1与0→1→0→1反例，解释 per-layer repetition 不等于重复多层模块。 |
| B16，中 L38 / 英 L38，paragraph | 配置使用左闭右开区间 \[loop_start, loop_end)，且循环区间必须落在静态前缀中。注意力重复和…<br>The configuration uses the half-open interval \[loop_sta… | P1：configuration限制静态前缀；init_layer_cache按访问分配独立cache；不声称参数共享等于缓存共享。 |
| B18，中 L42 / 英 L42，paragraph | 先关闭快权重和内循环，只改变第一层的重复次数。以下命令应在已准备好环境的远端 GPU 节点、仓库根目录执行。本…<br>First disable fast weights and inner updates, then chan… | P1：明确禁用 use_fast_mlp 与 inner.enabled，指定远端GPU并区分sample/heldout。 |
| B19，中 L44 / 英 L44，command | mkdir -p outputs/frontier-loop git rev-parse HEAD > out…<br>mkdir -p outputs/frontier-loop git rev-parse HEAD > out… | 命令 C1：从 configs/tiny.json 生成三组真实配置；ttt_suffix_length=0满足禁用约束；prepare_sample用单npy合法输入，非JSONL分片；待执行。 |
| B20，中 L74 / 英 L74，paragraph | 这三个模型都有两个独立 block，仅第一个 block 分别执行 1、2、4 次，因此完整 block 的访…<br>All three models have two independent blocks. Only the … | P1：实际访问次数2/3/5；仅第一层重复，不能误写2/4/8；未匹配FLOPs。 |
| B21，中 L76 / 英 L76，paragraph | 每个 run-r* 目录应包含 experiment.json、environment.json、metric…<br>Each run-r* directory should contain experiment.json, e… | P1：train.py实际产物/4步NPZ文件名/status complete；变更配置后以latest为准。 |
| B22，中 L78 / 英 L78，paragraph | 另可执行 bash scripts/validate_gpu.sh outputs/frontier-loop…<br>Separately, bash scripts/validate_gpu.sh outputs/fronti… | P1：validate_gpu是另组tiny功能检查，不得混作r1/r2/r4训练配置逐项验证。 |
| B24，中 L82 / 英 L82，paragraph | 第一组正式实验保持数据划分、tokenizer、独立参数、初始化规则、训练 token、优化器与精度一致，记录…<br>The first formal comparison fixes data splits, tokenize… | P1：同训练tokens与同FLOPs分开设计，不声称一个控制同时满足二者。 |
| B25，中 L84 / 英 L84，paragraph | 还应固定序列长度、batch、设备、缓存策略、残差与归一化位置，以及是否对中间重复提供额外监督。本仓库命令只使…<br>Also fix sequence length, batch, device, cache policy, … | P1：固定norm/residual/cache/监督等；本命令仅最终语言模型loss，不含每repeat额外监督。 |
| B26，中 L86 / 英 L86，paragraph | 质量指标至少包括独立验证集交叉熵和任务分数；资源指标包括独立参数量、训练 token、FLOPs、编译时间、预…<br>Quality measurements include held-out cross-entropy and… | P1：质量指标必须heldout；四步sample无泛化效力，较低loss需连计算成本报告。 |
| B27，中 L88 / 英 L88，table | ¦ 待判定问题 ¦ 当前证据 ¦ 决策 ¦ ¦---¦---¦---¦ ¦ loop_mode 是否真正共享参…<br>¦ Question ¦ Current evidence ¦ Decision ¦ ¦---¦---¦---… | P1：已实现参数共享与论文复现/质量/组合四项分开决策，无伪造结果。 |
| B29，中 L97 / 英 L97，paragraph | 若训练出现非有限值、不能产生完整检查点，或者流式输出与对应整段执行不一致，应先判为实现或配置失败。若在完整预算…<br>Nonfinite training, failure to write a complete checkpo… | P1：非有限/缺checkpoint/流式不一致为实现失败；质量无稳定提升不能通过挑seed掩盖。 |
| B30，中 L99 / 英 L99，paragraph | 目前还没有本文协议下的多 seed 质量报告、准确的跨配置 FLOPs 对齐，也没有整个多层 stack 的循…<br>This protocol still lacks multiple-seed quality reports… | 待核验：缺真实多seed/FLOPs对齐/整stack loop复现；先保存三组tiny产物再选质量数据。 |

## 代码证据与仍需补齐的证据

- [modeling_ttt_e2e.py](../../ttt/models/ttt_e2e/modeling_ttt_e2e.py)：_block、_run_layer 和 _empty_layer_state 支持 Loop 范围/参数/缓存说明。尚缺本书三组配置的原始运行报告。
- [configuration_ttt_e2e.py](../../ttt/models/ttt_e2e/configuration_ttt_e2e.py)：关闭快 MLP 必须设置 suffix=0，Loop 区间限制到静态前缀。配方由合法字段构造，但本轮未启动配置解析测试。
- [inner_loop.py](../../ttt/inner_loop.py)：交叉熵、SGD 裁剪、先评分后更新、不满块不提交更新，与简化 KV 平方重建有明确差别。
- [gated_delta.py](../../ttt/models/ttt_e2e/gated_delta.py)：额外的卷积、归一化、decay、beta、输出门控尚没有通过关闭逐项建立教学式对照。
- [validate.py](../../ttt/validate.py) 与 [validate_gpu.sh](../../scripts/validate_gpu.sh)：字段、失败阈值、case、默认长度和报告路径按真实实现写明。没有新增通过报告。
- [eval_retrieval.py](../../ttt/eval_retrieval.py)：两条配对命令用真实参数。没有随书生成的 heldout manifest 或 checkpoint，模板明确要求已有文件。
- [train.py](../../ttt/train.py)、[tiny.json](../../configs/tiny.json)、[prepare_sample.sh](../../scripts/prepare_sample.sh)：Loop 产物、步数和单文件样本路线可定位；没有用 sample.txt 作为 JSONL 分片输入。
- [tiny_gpu.json](../../configs/runs/tiny_gpu.json)：含 full_attention 前缀，阻止“固定快状态⇒模型显存/延迟恒定”的错误外推。

尚待独立复审：浏览器下公式（尤其小矩阵）渲染、中文/英文术语统一、是否应为状态轴与重复次序增加矢量图；这些未以静态文件检查替代。下一轮实验审稿需附远端环境、提交、完整配置、原始 JSON、失败日志和样本来源。没有这些产物时，质量表继续保持“未运行/未决”。
