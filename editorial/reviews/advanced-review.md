# Advanced chapters: paragraph review record

审查范围：<code>docs/book/05_inner_outer_loop.md</code> 至 <code>09_gpu_validation.md</code> 及对应英文，共 10 页。日期：2026-11-11。身份：本轮重写者自审，**不是独立审稿通过，不是全书出版验收**。

本记录把所有非空正文块分为段落、公式、表格、代码/文本图、命令、图与图注。每一项给出文件、行号、小节与首句/首式，便于另一位审稿者逐项回到原文。索引自动枚举，核对要点由阅读实现与手算后填写；自动枚举本身不构成正确性证据。标题作为定位信息，没有另算一条“通过”。

## 已修复的问题及依据

| 问题 | 已做修复 | 静态证据 |
|---|---|---|
| 一阶近似误写为断开全部快权重依赖 | 区分保留初始化 Jacobian 与 detach 全部状态；新增 .4669/.6225/0 手算 | 05 二阶导数节；inner_loop.py 可微更新 |
| 裁剪公式与实现 epsilon 位置不符 | 改为 sqrt(sum(g²)+1e-12)，另注明裁剪系数导数 | inner_loop.py:sgd_update |
| C=4 更新时点提前 | x4 时缺 x5 标签；收到 x5 先更新再处理新输入 | runtime.py:observe/_finish_chunk |
| 概念 cache 布局冒充实际布局 | 明确 [capacity,n_kv,dh] 与 positions；区分有效长度、容量、窗口 | attention.py:KVCache/empty_cache |
| 中英文 MoE 排列例子不同 | 均使用专家 [E1,E2,E1,E2]、顺序 [0,2,1,3]、组长 [2,2] | moe.py 的 stable argsort/scatter |
| capacity 与稀疏计算能力说得过满 | 容量仅背景；当前无 drop；高阶 JVP 参考路径可能计算全专家 | moe.py、grouped_gemm.py |
| 两GPU被重复计算为两路data和两路expert | 加乘积约束和两种实际shard表；指出data轴参数分片 | distributed.py:tree_shardings |
| 推理profile冒充训练吞吐 | 明确warmup/单次推理、XLA估算与完整训练的区别 | profile.py |
| GPU验证报告被赋予不存在的恢复/长上下文能力 | 按真实字段与默认64/512写；1b恢复跳过、L=4 | validate.py、validate_mesh.py |
| 固定token与固定时间结果混用 | 新增明确标注为假设的质量成本表，算出50s/83.3s/600k差异 | 08 质量成本节 |
| 配方 C_k 与外循环步 k 混淆 | 用具体 C2/F2、C4/F4，并解释数字为full-attention层数 | experiments.py:batch1 |
| attention mask 与loss mask记号混淆 | additive attention mask 另记为 \(\mathcal M\) | 06二维手算 |
| 交叉审稿发现“至少2张可见GPU”条件不足 | 07/09两版和1b命令均明确恰好2张可见设备；说明8卡data=4与batch=2不兼容，保留调度器设备映射 | distributed.py:create_mesh、validate_mesh.py固定batch=2；由其他代理指出后修正，待其复核 |
| 交叉审稿发现把mlp适应误称更新专家MLP | 07两版明确mlp更新MoE前额外稠密MLP′，router更新适配器，delta_router更新递归记忆；moe.experts只由外循环训练 | modeling_ttt_e2e.py:initial_fast_weights/_block；由其他代理指出后修正，待其复核 |

## 仍未解决或需要外部证据

- 本轮未执行本地测试，也未执行远端 CPU/GPU 命令。命令语法、参数和预期字段仅按源码核对；运行正确性仍待远端报告。
- 本轮沿用现有 SVG，只改正文与图注。艺术质量、印刷清晰度、色盲友好性、手机布局和双语溢出尚未验收，不能称精美配图已达出版要求。
- 需要另一位代理独立逐段审稿，以及按统一记号表做最终交叉核对。中英文节顺序、数字、命令与证据边界已自审，翻译自然度仍可继续修改。
- 目标参数/长度/batch的长时间运行、稳态时间分布、真实数据游标恢复、跨拓扑恢复和128K实验均不由当前tiny检查证明。
- 对完全初学读者，Jacobian、分片、bootstrap 的前置知识仍需要全书基础章节配合。章节内给出局部解释，不宣称读者研究已经完成。
- 现有示意图中的capacity/状态边界可能仍不如新正文精确，后续独立视觉审稿应按新正文逐项检查箭头与标签。

## 分节核对依据

### 05

| 证据编号 | 本轮实际核对内容 |
|---|---|
| E05.01 | 以序列与训练步区分状态；对照 model.initial_fast_weights、training.py，确认初值从静态参数提取。 |
| E05.02 | 手算 CE=.6931→.4741、梯度-.5；显式 logits Jacobian 的 [V,n_phi] 形状；不是用参数拟合 token ID。 |
| E05.03 | 逐行核对 sgd_update：epsilon 在 sqrt 内；全树范数与逐元素裁剪区分；平方误差例仅作算术。 |
| E05.04 | C=4 输入需要 x5 标签；sequence_loss 尾块仅评分并保留旧快权重，runtime 等下一 token。 |
| E05.05 | inner_step 输出 pre-update cache；track_post_loss 是同块诊断；scan/rematerialization 不冒充持久化。 |
| E05.06 | 100×2+2×10=220，除102；对照 training.py 的token加权、单列nll与两个学习率日程。 |
| E05.07 | 保留 J0 的一阶近似和 detach(phi1) 分开；手算 .4669/.6225/0；裁剪生效时额外链式依赖已注明。 |
| E05.08 | 逐项核对 validate.py/script 参数与现有 JSON 字段；无不存在的 causality 字段或 snapshot 检查。 |
| E05.09 | 实际 C2/F2 名称和 full-attention 层数；功能、质量与恢复证据区分。 |

### 06

| 证据编号 | 本轮实际核对内容 |
|---|---|
| E06.01 | 接收 token 与保存状态的动机；图沿用现有 SVG，图注补充快权重/未完成chunk，视觉审校待完成。 |
| E06.02 | 手算 QK^T/sqrt2、完整2×2因果表与softmax；对角线可见、预测目标下一位；attention mask 与loss mask不同名。 |
| E06.03 | 核对 attention.py 的单session token-major布局；2MiB/层、24MiB/12层；有效长度/容量/窗口分开。 |
| E06.04 | 逐调用核对 runtime.observe：x4后updates=0，x5后updates=1；不重算历史KV；需同协议比较。 |
| E06.05 | 核对 SessionState 与session_store.py格式2/第一本地设备；恢复测试入口和CLI检查不混淆。 |
| E06.06 | 代码路径均存在；GPU报告 fields/阈值/失败先查项及未验证边界与validate.py一致。 |

### 07

| 证据编号 | 本轮实际核对内容 |
|---|---|
| E07.01 | 专家参数3Edf与激活参数3kdf只作结构估算；明确router/通信成本；沿用SVG待视觉审校。 |
| E07.02 | topk(logits+load_bias)与原始softmax权重分开；四token排序[0,2,1,3]；top2加权例与top1梯度边界。 |
| E07.03 | capacity为背景不冒充现有功能；no-drop路径；aux=1/1.8手算；top-k频率归一化及mask/z-loss。 |
| E07.04 | 对应moe.py排序/scatter；grouped_gemm.py普通ragged_dot与JVP全专家参考路径分开，不能由稀疏参数推算训练FLOPs。 |
| E07.05 | 双GPU(data1,expert2)，gate [4,16,16]→[2,16,16]；真实mesh命令和tiny恢复报告。 |
| E07.06 | 分别研究MoE前额外稠密MLP′适应、router适配器更新、Delta memory与循环计算；moe.experts由外循环训练，不混合归因。 |

### 08

| 证据编号 | 本轮实际核对内容 |
|---|---|
| E08.01 | 教学loss差引入可归因问题；图用于证据闭环，未宣称已完成正式实验。 |
| E08.02 | C2/F2实际配方名称；初始化相同不等于轨迹相同；数据/tokenizer/reset均需固定。 |
| E08.03 | 教学表明确非实测：1M/20k=50、1M/12k=83.3；固定时间不能复用固定token终点loss；有效目标与调度位置分开。 |
| E08.04 | 配对种子-0.10/-0.12/-0.08，sd .02/SE .0115；小样本不自动宣称95%区间。 |
| E08.05 | experiments CLI选项/manifest/pending真实；仅生成配置；向上取整和Llama tokenizer前提明确。 |
| E08.06 | profile使用ch19实际tiny checkpoint，8+4<16；单session推理而非训练吞吐；XLA估算/峰值局限。 |
| E08.07 | scaling.py公式归一化/输入键/八条且三规模三训练量；检查不自动保证交叉设计；held-out与bootstrap边界。 |

### 09

| 证据编号 | 本轮实际核对内容 |
|---|---|
| E09.01 | 三层证据区分；正式准入表是人工要求，不伪装现有一键功能。 |
| E09.02 | validate.py默认64/512，dense仍含TTT；三项断言阈值与失败字段；无128K/恢复/质量过度结论。 |
| E09.03 | lower.compile与单次执行字段；35+10秒例，2048与455.1t/s；异步计时须同步；稳态分布尚未产出。 |
| E09.04 | 设备数为data×expert；2GPU不能2×2；tree_shardings data轴也切参数，不写成纯复制DP。 |
| E09.05 | mesh tiny两步及原/恢复下一步全状态比较；固定batch不覆盖真实data cursor/进程中断/拓扑迁移。 |
| E09.06 | 1b命令真实；L4/maxpos16；checkpoint_roundtrip应false；不是长上下文复现。 |
| E09.07 | 逐项注明目标配置显存、稳态、数据游标恢复和provenance需要新证据，不从tiny passed推导。 |

## 交叉审稿后的修订状态

独立审稿者 paragraph_gate_rewritten_main 已静态复核07/09双卡前提与07 MLP′适应对象的修复，两个具体硬问题在正文层关闭。09另明确四卡data=2也能整除batch=2，避免把教程选择的双卡拓扑说成代码唯一允许拓扑。05高阶导数、06标签到达、07 JVP成本经其源码核对未发现新增硬错误。以上不是全书出版通过，也不关闭运行、网页和美术待验项。

## 逐段、逐公式、逐表、逐命令索引

下表状态统一为“已自审；参照上节独立复核范围”。证据编号对应上面的具体核对内容。对命令，静态核对不代表已经运行；对图，核对图注不代表视觉验收。

### docs/book/05_inner_outer_loop.md

| 单元 | 位置与小节 | 类型 | 首句/首式或命令 | 核对依据 |
|---|---|---|---|---|
| 05ZH-001 | [L5](../../docs/book/05_inner_outer_loop.md#L5) · 从一次预测区分两个时间尺度 | 段落 | 上一章把一个 token 沿着前向路径送到输出头。现在设模型读到一段介绍新术语的文本：它刚刚预测过一个词，也已经看到了正确答案。我们希望这次经验能帮助它理解后面的内容，同时希望这种“如何利用经验”的能 | E05.01 |
| 05ZH-002 | [L7](../../docs/book/05_inner_outer_loop.md#L7) · 从一次预测区分两个时间尺度 | 段落 | 本章沿用 tiny 配置的隐藏宽度 \(d=16\) 和 chunk 长度 \(C=4\)。读完后，你应能写出一个 chunk 使用哪个快状态、哪一个标签触发更新，以及外循环梯度如何经过这次更新。需要 | E05.01 |
| 05ZH-003 | [L9](../../docs/book/05_inner_outer_loop.md#L9) · 从一次预测区分两个时间尺度 | 段落 | 用 \(\theta_k\) 表示第 \(k\) 个外循环训练步的静态参数，用 \(\phi_c\) 表示当前序列第 \(c\) 个 chunk 开始时的快权重。这里 \(c\) 数 chunk，不数 | E05.01 |
| 05ZH-004 | [L11](../../docs/book/05_inner_outer_loop.md#L11) · 从一次预测区分两个时间尺度 | 代码/文本图 | 固定 θk | E05.01 |
| 05ZH-005 | [L19](../../docs/book/05_inner_outer_loop.md#L19) · 从一次预测区分两个时间尺度 | 段落 | 内循环改变当前序列的状态；外循环改变下一次训练时使用的静态参数和快权重初值。一次训练样本完成后的 \(\phi\) 不会直接成为下一条独立样本的起点；推理时是否继续使用它，则取决于是否继续同一个 se | E05.01 |
| 05ZH-006 | [L23](../../docs/book/05_inner_outer_loop.md#L23) · 从 token 交叉熵推到快权重更新 | 段落 | 先用一个刻意简化的二分类语言模型连接“预测”和“梯度”。词表只有 \(A,B\)，快参数为一个标量 \(\phi\)，输出 logits 为 \(\mathbf z=(0,\phi)\)。它没有真正的 | E05.02 |
| 05ZH-007 | [L25](../../docs/book/05_inner_outer_loop.md#L25) · 从 token 交叉熵推到快权重更新 | 公式 | p_\phi(B)=\frac{e^\phi}{1+e^\phi},\qquad | E05.02 |
| 05ZH-008 | [L30](../../docs/book/05_inner_outer_loop.md#L30) · 从 token 交叉熵推到快权重更新 | 段落 | 对两项分别求导： | E05.02 |
| 05ZH-009 | [L32](../../docs/book/05_inner_outer_loop.md#L32) · 从 token 交叉熵推到快权重更新 | 公式 | \frac{d\ell}{d\phi} | E05.02 |
| 05ZH-010 | [L38](../../docs/book/05_inner_outer_loop.md#L38) · 从 token 交叉熵推到快权重更新 | 段落 | 若 \(\phi_0=0\)，两个候选概率各为 \(0.5\)，损失为 \(\log2\approx0.6931\)，梯度为 \(-0.5\)。学习率为 \(1\) 的一步 SGD 得到 \(\phi | E05.02 |
| 05ZH-011 | [L40](../../docs/book/05_inner_outer_loop.md#L40) · 从 token 交叉熵推到快权重更新 | 段落 | 若两个位置分别预测 \(B\) 和 \(A\)，并且这个极简模型在两个位置都输出同一个 \((0,\phi)\)，则在 \(\phi=0\) 时平均梯度为 \([(-0.5)+0.5]/2=0\)。没 | E05.02 |
| 05ZH-012 | [L42](../../docs/book/05_inner_outer_loop.md#L42) · 从 token 交叉熵推到快权重更新 | 公式 | \nabla_\phi\mathcal L_{\mathrm{inner}} | E05.02 |
| 05ZH-013 | [L51](../../docs/book/05_inner_outer_loop.md#L51) · 从 token 交叉熵推到快权重更新 | 段落 | 右边先计算每个词表 logit 的误差，再经由 logits 对快矩阵的 Jacobian 传回参数。前例中第一个 logit 的导数为 \(0\)，第二个为 \(1\)，所以只留下 \(p(B)-1 | E05.02 |
| 05ZH-014 | [L53](../../docs/book/05_inner_outer_loop.md#L53) · 从 token 交叉熵推到快权重更新 | 段落 | Jacobian 可以看成一张导数表：第 \(v\) 行、第 \(i\) 列记录第 \(i\) 个参数略微变化时，第 \(v\) 个 logit 怎样变化。若把快矩阵展平成 \(n_\phi\) 个参 | E05.02 |
| 05ZH-015 | [L57](../../docs/book/05_inner_outer_loop.md#L57) · 裁剪改变的是步幅，不是标签 | 段落 | 以平方误差 \(\ell(\phi)=\tfrac12(\phi-3)^2\) 为另一个教学例子，\(\phi_0=1\) 时梯度为 \(-2\)。学习率 \(0.5\) 的普通 SGD 把状态更新到 | E05.03 |
| 05ZH-016 | [L59](../../docs/book/05_inner_outer_loop.md#L59) · 裁剪改变的是步幅，不是标签 | 段落 | 若所有快权重叶子的梯度记为 \(g_j\)，代码先求全局范数 | E05.03 |
| 05ZH-017 | [L61](../../docs/book/05_inner_outer_loop.md#L61) · 裁剪改变的是步幅，不是标签 | 公式 | \widetilde{\／g\／}_2=\sqrt{\sum_j\sum_i g_{j,i}^2+10^{-12}}, | E05.03 |
| 05ZH-018 | [L69](../../docs/book/05_inner_outer_loop.md#L69) · 裁剪改变的是步幅，不是标签 | 段落 | 平方根内的小量与代码一致，用于避免零梯度处高阶求导的数值问题。当 \(\kappa=0.5\) 时，上面的标量梯度被乘以约 \(0.25\)，所以更新后约为 \(1.25\)。clip norm 限制 | E05.03 |
| 05ZH-019 | [L73](../../docs/book/05_inner_outer_loop.md#L73) · 四个输入位置需要五个 token | 段落 | 长度为 \(C\) 的预测块需要 \(C\) 个标签。对输入 \(x_1,x_2,x_3,x_4\)，标签是 \(x_2,x_3,x_4,x_5\)。因此在线读取到 \(x_4\) 时，仍不知道最后一 | E05.04 |
| 05ZH-020 | [L75](../../docs/book/05_inner_outer_loop.md#L75) · 四个输入位置需要五个 token | 表格 | ／ 阶段 ／ 输入的数学位置 ／ 预测时使用的状态 ／ 更新所需标签 ／ 后续状态 ／ | E05.04 |
| 05ZH-021 | [L81](../../docs/book/05_inner_outer_loop.md#L81) · 四个输入位置需要五个 token | 段落 | 训练时整条序列已经在内存里，仍必须维持“本块先预测，后续块才使用更新结果”的因果约束。sequence_loss 将尾部补齐以获得固定形状；它会计算真实位置的 loss，却用条件选择保留不完整块的旧快 | E05.04 |
| 05ZH-022 | [L85](../../docs/book/05_inner_outer_loop.md#L85) · 公式对应哪些函数与数组 | 段落 | ttt/inner_loop.py 的 inner_step 执行 suffix、计算交叉熵、对 fast weights 求导并调用 sgd_update。输出 cache 来自更新前的预测；历史  | E05.05 |
| 05ZH-023 | [L87](../../docs/book/05_inner_outer_loop.md#L87) · 公式对应哪些函数与数组 | 表格 | ／ 代码对象 ／ 单序列形状或类型 ／ 语义 ／ | E05.05 |
| 05ZH-024 | [L96](../../docs/book/05_inner_outer_loop.md#L96) · 公式对应哪些函数与数组 | 段落 | sequence_loss 用两层 lax.scan 扫过 chunk，并用 jax.checkpoint 在反向时重算部分中间值。这里的 checkpoint 是计算重算，不是磁盘训练检查点；它节省 | E05.05 |
| 05ZH-025 | [L100](../../docs/book/05_inner_outer_loop.md#L100) · 外循环：不同样本如何合成一个目标 | 段落 | 若样本 \(b\) 有 \(n_b\) 个有效标签，其平均损失为 \(\mathcal L_b\)，则全 batch 的 token 平均是 | E05.06 |
| 05ZH-026 | [L102](../../docs/book/05_inner_outer_loop.md#L102) · 外循环：不同样本如何合成一个目标 | 公式 | \mathcal L_{\mathrm{batch}} | E05.06 |
| 05ZH-027 | [L107](../../docs/book/05_inner_outer_loop.md#L107) · 外循环：不同样本如何合成一个目标 | 段落 | 100 个标签、平均 loss 为 2 的样本，与 2 个标签、平均 loss 为 10 的样本，合并后是 \(220/102\approx2.157\)，不是 \((2+10)/2=6\)。分母保护 | E05.06 |
| 05ZH-028 | [L109](../../docs/book/05_inner_outer_loop.md#L109) · 外循环：不同样本如何合成一个目标 | 段落 | ttt/training.py 对这个目标求静态参数梯度，再交给外循环 Optax optimizer。MoE 正则项会加入训练目标，但 nll 单独记录。内循环学习率还有独立的 warmup：从 i | E05.06 |
| 05ZH-029 | [L113](../../docs/book/05_inner_outer_loop.md#L113) · 外循环为什么包含二阶导数 | 段落 | 设快权重初值为 \(\phi_0(\theta)\)，先对未触发裁剪的一步 SGD 展开依赖： | E05.07 |
| 05ZH-030 | [L115](../../docs/book/05_inner_outer_loop.md#L115) · 外循环为什么包含二阶导数 | 公式 | \phi_1=\phi_0(\theta)-\eta g(\theta,\phi_0(\theta)), | E05.07 |
| 05ZH-031 | [L120](../../docs/book/05_inner_outer_loop.md#L120) · 外循环为什么包含二阶导数 | 段落 | 为了看清每条路径，令 \(J_0=d\phi_0/d\theta\)，并假设本次更新的 \(\eta\) 不依赖 \(\theta\)。则 | E05.07 |
| 05ZH-032 | [L122](../../docs/book/05_inner_outer_loop.md#L122) · 外循环为什么包含二阶导数 | 公式 | \frac{d\phi_1}{d\theta} | E05.07 |
| 05ZH-033 | [L130](../../docs/book/05_inner_outer_loop.md#L130) · 外循环为什么包含二阶导数 | 公式 | \frac{d\mathcal L_{\mathrm{outer}}}{d\theta} | E05.07 |
| 05ZH-034 | [L137](../../docs/book/05_inner_outer_loop.md#L137) · 外循环为什么包含二阶导数 | 段落 | \(g\) 已是一阶导数，再对它求导就出现二阶项。full meta-gradient 保留这些依赖。常见的一阶近似忽略 \(g\) 的导数，仍保留 \(J_0\)；若对整个 \(\phi_1\) 调 | E05.07 |
| 05ZH-035 | [L139](../../docs/book/05_inner_outer_loop.md#L139) · 外循环为什么包含二阶导数 | 段落 | 回到前面的二分类模型，可以把这三个结果手算出来。设 \(\phi_0=\theta=0\)，内循环标签为 \(B\)、学习率为 1，更新后仍有 \(\phi_1=0.5\)。因为 \(g(\phi)= | E05.07 |
| 05ZH-036 | [L141](../../docs/book/05_inner_outer_loop.md#L141) · 外循环为什么包含二阶导数 | 段落 | 只为观察后续预测的导数，暂时把外目标选为下一个位置的标签 \(A\)，即 \(\mathcal L_{\rm future}=\log(1+e^{\phi_1})\)。它对 \(\phi_1\) 的导 | E05.07 |
| 05ZH-037 | [L143](../../docs/book/05_inner_outer_loop.md#L143) · 外循环为什么包含二阶导数 | 段落 | 实际更新若触发裁剪，应把上式的 \(g\) 换成 \(a(g)g\)，其导数也包含裁剪系数的依赖；不能直接把未裁剪推导当作全部实现。当前内循环没有提供一个可直接切换 FOMAML 的配置项；外循环沿可 | E05.07 |
| 05ZH-038 | [L147](../../docs/book/05_inner_outer_loop.md#L147) · 远端最小验证与如何读报告 | 段落 | 在已配置依赖的远端 GPU 节点、仓库根目录运行： | E05.08 |
| 05ZH-039 | [L149](../../docs/book/05_inner_outer_loop.md#L149) · 远端最小验证与如何读报告 | 命令 | bash scripts/validate_gpu.sh outputs/ch17/validation.json | E05.08 |
| 05ZH-040 | [L153](../../docs/book/05_inner_outer_loop.md#L153) · 远端最小验证与如何读报告 | 段落 | 这会运行 ttt.validate 自带的 tiny case，不会读取你的正式训练语料，也不会启动 125M 训练。报告顶层应有 passed、environment 和 cases。各 case  | E05.08 |
| 05ZH-041 | [L155](../../docs/book/05_inner_outer_loop.md#L155) · 远端最小验证与如何读报告 | 段落 | 成功要求 loss 和梯度有限、流式误差不超过脚本的 \(10^{-3}\) 门槛，并且改变未来 token 后的前缀 logits 通过脚本内的容差断言。因果性没有独立 JSON 布尔字段；只有断言 | E05.08 |
| 05ZH-042 | [L159](../../docs/book/05_inner_outer_loop.md#L159) · 小结：状态变化还不是质量提升 | 段落 | 本章建立了 token loss → 快权重梯度 → 后续 chunk 状态 → 外循环导数这条链。功能报告只能证明这些小配置完成了列出的检查；没有证明正式语料的泛化收益，没有验证所有长度，也没有执行 | E05.09 |
| 05ZH-043 | [L161](../../docs/book/05_inner_outer_loop.md#L161) · 小结：状态变化还不是质量提升 | 段落 | 质量对照 C2/F2（或 C4/F4）要固定架构、初始化、数据、预算和外循环日程，仅改变内循环是否更新；名称中的数字指配方的 full-attention 层数。删除 MLP′ 与关闭更新是两项不同消 | E05.09 |

### docs/book/en/05_inner_outer_loop.md

| 单元 | 位置与小节 | 类型 | 首句/首式或命令 | 核对依据 |
|---|---|---|---|---|
| 05EN-001 | [L5](../../docs/book/en/05_inner_outer_loop.md#L5) · Two clocks behind one prediction | 段落 | The previous chapter followed a token through the forward path. Now imagine a paragraph that introdu | E05.01 |
| 05EN-002 | [L7](../../docs/book/en/05_inner_outer_loop.md#L7) · Two clocks behind one prediction | 段落 | We keep the tiny hidden width \(d=16\) and chunk length \(C=4\). By the end of this chapter you shou | E05.01 |
| 05EN-003 | [L9](../../docs/book/en/05_inner_outer_loop.md#L9) · Two clocks behind one prediction | 段落 | Let \(\theta_k\) denote static parameters at outer training step \(k\), and let \(\phi_c\) be fast w | E05.01 |
| 05EN-004 | [L11](../../docs/book/en/05_inner_outer_loop.md#L11) · Two clocks behind one prediction | 代码/文本图 | hold theta_k fixed | E05.01 |
| 05EN-005 | [L19](../../docs/book/en/05_inner_outer_loop.md#L19) · Two clocks behind one prediction | 段落 | The inner loop changes sequence state. The outer loop changes the static parameters and learned fast | E05.01 |
| 05EN-006 | [L23](../../docs/book/en/05_inner_outer_loop.md#L23) · From token cross-entropy to a fast-weight update | 段落 | Start with an intentionally simplified two-class language model. Its vocabulary contains \(A,B\); on | E05.02 |
| 05EN-007 | [L25](../../docs/book/en/05_inner_outer_loop.md#L25) · From token cross-entropy to a fast-weight update | 公式 | p_\phi(B)=\frac{e^\phi}{1+e^\phi},\qquad | E05.02 |
| 05EN-008 | [L30](../../docs/book/en/05_inner_outer_loop.md#L30) · From token cross-entropy to a fast-weight update | 段落 | Differentiating the two terms gives | E05.02 |
| 05EN-009 | [L32](../../docs/book/en/05_inner_outer_loop.md#L32) · From token cross-entropy to a fast-weight update | 公式 | \frac{d\ell}{d\phi} | E05.02 |
| 05EN-010 | [L38](../../docs/book/en/05_inner_outer_loop.md#L38) · From token cross-entropy to a fast-weight update | 段落 | At \(\phi_0=0\), both candidates have probability \(0.5\), the loss is \(\log2\approx0.6931\), and t | E05.02 |
| 05EN-011 | [L40](../../docs/book/en/05_inner_outer_loop.md#L40) · From token cross-entropy to a fast-weight update | 段落 | If two positions have targets \(B,A\), and this toy model uses the same \((0,\phi)\) at both positio | E05.02 |
| 05EN-012 | [L42](../../docs/book/en/05_inner_outer_loop.md#L42) · From token cross-entropy to a fast-weight update | 公式 | \nabla_\phi\mathcal L_{\mathrm{inner}} | E05.02 |
| 05EN-013 | [L51](../../docs/book/en/05_inner_outer_loop.md#L51) · From token cross-entropy to a fast-weight update | 段落 | First compute each vocabulary logit's error, then propagate it through the Jacobian from fast parame | E05.02 |
| 05EN-014 | [L53](../../docs/book/en/05_inner_outer_loop.md#L53) · From token cross-entropy to a fast-weight update | 段落 | A Jacobian is a table of derivatives: entry \(v,i\) describes how logit \(v\) changes when parameter | E05.02 |
| 05EN-015 | [L57](../../docs/book/en/05_inner_outer_loop.md#L57) · Clipping changes the step, not the label | 段落 | As a separate arithmetic illustration, consider \(\ell(\phi)=\tfrac12(\phi-3)^2\). At \(\phi_0=1\),  | E05.03 |
| 05EN-016 | [L59](../../docs/book/en/05_inner_outer_loop.md#L59) · Clipping changes the step, not the label | 段落 | For gradient leaves \(g_j\), the code computes the global norm and update | E05.03 |
| 05EN-017 | [L61](../../docs/book/en/05_inner_outer_loop.md#L61) · Clipping changes the step, not the label | 公式 | \widetilde{\／g\／}_2=\sqrt{\sum_j\sum_i g_{j,i}^2+10^{-12}}, | E05.03 |
| 05EN-018 | [L69](../../docs/book/en/05_inner_outer_loop.md#L69) · Clipping changes the step, not the label | 段落 | The small value inside the square root matches code and avoids numerical problems in higher derivati | E05.03 |
| 05EN-019 | [L73](../../docs/book/en/05_inner_outer_loop.md#L73) · Four input positions require five tokens | 段落 | A block of \(C\) predictions needs \(C\) labels. Inputs \(x_1,x_2,x_3,x_4\) require targets \(x_2,x_ | E05.04 |
| 05EN-020 | [L75](../../docs/book/en/05_inner_outer_loop.md#L75) · Four input positions require five tokens | 表格 | ／ Stage ／ Mathematical input positions ／ State used for prediction ／ Labels needed for update ／ Resu | E05.04 |
| 05EN-021 | [L81](../../docs/book/en/05_inner_outer_loop.md#L81) · Four input positions require five tokens | 段落 | Training already has the entire sequence in memory but must preserve the same causal order: predict  | E05.04 |
| 05EN-022 | [L85](../../docs/book/en/05_inner_outer_loop.md#L85) · Mapping equations to functions and arrays | 段落 | In ttt/inner_loop.py, inner_step runs the suffix, computes cross-entropy, differentiates fast weight | E05.05 |
| 05EN-023 | [L87](../../docs/book/en/05_inner_outer_loop.md#L87) · Mapping equations to functions and arrays | 表格 | ／ Code object ／ Single-sequence shape or type ／ Meaning ／ | E05.05 |
| 05EN-024 | [L96](../../docs/book/en/05_inner_outer_loop.md#L96) · Mapping equations to functions and arrays | 段落 | sequence_loss uses two levels of lax.scan and jax.checkpoint to recompute intermediate values during | E05.05 |
| 05EN-025 | [L100](../../docs/book/en/05_inner_outer_loop.md#L100) · Combining examples in the outer objective | 段落 | If sample \(b\) has \(n_b\) valid targets and mean loss \(\mathcal L_b\), the batch's token mean is | E05.06 |
| 05EN-026 | [L102](../../docs/book/en/05_inner_outer_loop.md#L102) · Combining examples in the outer objective | 公式 | \mathcal L_{\mathrm{batch}} | E05.06 |
| 05EN-027 | [L107](../../docs/book/en/05_inner_outer_loop.md#L107) · Combining examples in the outer objective | 段落 | Combining 100 targets at mean loss 2 with 2 targets at mean loss 10 gives \(220/102\approx2.157\), n | E05.06 |
| 05EN-028 | [L109](../../docs/book/en/05_inner_outer_loop.md#L109) · Combining examples in the outer objective | 段落 | ttt/training.py differentiates the objective with respect to static parameters and passes gradients  | E05.06 |
| 05EN-029 | [L113](../../docs/book/en/05_inner_outer_loop.md#L113) · Why the outer derivative contains second derivatives | 段落 | Let the initialization be \(\phi_0(\theta)\). First expand one SGD update when clipping is inactive: | E05.07 |
| 05EN-030 | [L115](../../docs/book/en/05_inner_outer_loop.md#L115) · Why the outer derivative contains second derivatives | 公式 | \phi_1=\phi_0(\theta)-\eta g(\theta,\phi_0(\theta)), | E05.07 |
| 05EN-031 | [L120](../../docs/book/en/05_inner_outer_loop.md#L120) · Why the outer derivative contains second derivatives | 段落 | Set \(J_0=d\phi_0/d\theta\), and assume \(\eta\) does not depend on \(\theta\) for this step. Then | E05.07 |
| 05EN-032 | [L122](../../docs/book/en/05_inner_outer_loop.md#L122) · Why the outer derivative contains second derivatives | 公式 | \frac{d\phi_1}{d\theta} | E05.07 |
| 05EN-033 | [L130](../../docs/book/en/05_inner_outer_loop.md#L130) · Why the outer derivative contains second derivatives | 公式 | \frac{d\mathcal L_{\mathrm{outer}}}{d\theta} | E05.07 |
| 05EN-034 | [L137](../../docs/book/en/05_inner_outer_loop.md#L137) · Why the outer derivative contains second derivatives | 段落 | Because \(g\) is already a first derivative, differentiating it introduces second-order terms. A ful | E05.07 |
| 05EN-035 | [L139](../../docs/book/en/05_inner_outer_loop.md#L139) · Why the outer derivative contains second derivatives | 段落 | The earlier binary model makes the three outcomes calculable by hand. Set \(\phi_0=\theta=0\), inner | E05.07 |
| 05EN-036 | [L141](../../docs/book/en/05_inner_outer_loop.md#L141) · Why the outer derivative contains second derivatives | 段落 | To isolate the future-prediction derivative, temporarily choose the next position's target \(A\) as  | E05.07 |
| 05EN-037 | [L143](../../docs/book/en/05_inner_outer_loop.md#L143) · Why the outer derivative contains second derivatives | 段落 | When clipping activates, replace \(g\) with \(a(g)g\); its derivative includes the clipping scale's  | E05.07 |
| 05EN-038 | [L147](../../docs/book/en/05_inner_outer_loop.md#L147) · Minimal remote validation and report reading | 段落 | On a remote GPU node with dependencies installed, run from the repository root: | E05.08 |
| 05EN-039 | [L149](../../docs/book/en/05_inner_outer_loop.md#L149) · Minimal remote validation and report reading | 命令 | bash scripts/validate_gpu.sh outputs/ch17/validation.json | E05.08 |
| 05EN-040 | [L153](../../docs/book/en/05_inner_outer_loop.md#L153) · Minimal remote validation and report reading | 段落 | This runs the built-in tiny cases in ttt.validate. It does not load your formal training corpus or s | E05.08 |
| 05EN-041 | [L155](../../docs/book/en/05_inner_outer_loop.md#L155) · Minimal remote validation and report reading | 段落 | Success requires finite loss and gradients, streaming error within the script's \(10^{-3}\) limit, a | E05.08 |
| 05EN-042 | [L159](../../docs/book/en/05_inner_outer_loop.md#L159) · Summary: an update is not a quality improvement | 段落 | We have connected token loss, fast-weight gradients, later chunk state, and the outer derivative. Th | E05.09 |
| 05EN-043 | [L161](../../docs/book/en/05_inner_outer_loop.md#L161) · Summary: an update is not a quality improvement | 段落 | A C2/F2 (or C4/F4) comparison fixes architecture, initialization, data, budget, and outer schedules, | E05.09 |

### docs/book/06_attention_cache.md

| 单元 | 位置与小节 | 类型 | 首句/首式或命令 | 核对依据 |
|---|---|---|---|---|
| 06ZH-001 | [L5](../../docs/book/06_attention_cache.md#L5) · 为什么生成一句话需要保存状态 | 段落 | 上一章已经回答了快权重何时更新。现在把运行方式从“一次处理整段文本”换成“逐个接收 token”。假设模型已读到“今天的天气”，正在预测下一个词；得到新 token 后，它需要结合整个已知前缀继续预测 | E06.01 |
| 06ZH-002 | [L7](../../docs/book/06_attention_cache.md#L7) · 为什么生成一句话需要保存状态 | 段落 | 本章继续使用隐藏宽度 \(d=16\)、TTT chunk 长度 \(C=4\) 的 tiny 案例。我们先用更小的二维向量手算 attention，再追踪一个 session 从 \(x_1\) 到 | E06.01 |
| 06ZH-003 | [L9](../../docs/book/06_attention_cache.md#L9) · 为什么生成一句话需要保存状态 | 图与图注 | 图 4　从整段计算到逐 token 计算。 沿着新 token 的箭头追踪查询，再沿历史箭头找它可读取的 key/value。图展示缓存的用途；真实 TTT session 还必须保存快权重、位置和未 | E06.01 |
| 06ZH-004 | [L16](../../docs/book/06_attention_cache.md#L16) · 从“找相关内容”到两个位置的手算 | 段落 | Attention 可以理解为一次带权检索。当前位置先生成一个 query，表示“我需要什么”；历史位置生成 key，表示“这里有什么”；value 是检索后真正传递的信息。三者都由可学习的线性变换得 | E06.02 |
| 06ZH-005 | [L18](../../docs/book/06_attention_cache.md#L18) · 从“找相关内容”到两个位置的手算 | 公式 | Q=HW_Q,\qquad K=HW_K,\qquad V=HW_V. | E06.02 |
| 06ZH-006 | [L22](../../docs/book/06_attention_cache.md#L22) · 从“找相关内容”到两个位置的手算 | 段落 | 这里 \(H\in\mathbb R^{L\times d}\) 是一条序列的 hidden states，\(L\) 是长度；在一个 attention head 中，\(Q,K\in\mathbb | E06.02 |
| 06ZH-007 | [L24](../../docs/book/06_attention_cache.md#L24) · 从“找相关内容”到两个位置的手算 | 段落 | 取 \(L=d_h=2\)，并给出已经投影好的矩阵： | E06.02 |
| 06ZH-008 | [L26](../../docs/book/06_attention_cache.md#L26) · 从“找相关内容”到两个位置的手算 | 公式 | Q=\begin{bmatrix}1&0\\1&0\end{bmatrix},\quad | E06.02 |
| 06ZH-009 | [L32](../../docs/book/06_attention_cache.md#L32) · 从“找相关内容”到两个位置的手算 | 段落 | 尚未加 mask 的分数是 | E06.02 |
| 06ZH-010 | [L34](../../docs/book/06_attention_cache.md#L34) · 从“找相关内容”到两个位置的手算 | 公式 | S=\frac{QK^\top}{\sqrt2} | E06.02 |
| 06ZH-011 | [L39](../../docs/book/06_attention_cache.md#L39) · 从“找相关内容”到两个位置的手算 | 段落 | 语言模型中的位置 1 不能读取位置 2。否则模型在预测下一个 token 时，可能直接从未来输入中看到答案。因果 mask 是加在分数上的矩阵；对合法位置加 0，对不合法位置加负无穷： | E06.02 |
| 06ZH-012 | [L41](../../docs/book/06_attention_cache.md#L41) · 从“找相关内容”到两个位置的手算 | 表格 | ／ 查询位置 \ 历史位置 ／ \(s=1\) ／ \(s=2\) ／ | E06.02 |
| 06ZH-013 | [L46](../../docs/book/06_attention_cache.md#L46) · 从“找相关内容”到两个位置的手算 | 段落 | 用 \(\mathcal M\) 表示这里的 attention 加性 mask，以区别数据章节的 loss mask。逐行做 softmax，得到权重 \(A\) 和输出 \(O\)： | E06.02 |
| 06ZH-014 | [L48](../../docs/book/06_attention_cache.md#L48) · 从“找相关内容”到两个位置的手算 | 公式 | \mathcal M=\begin{bmatrix}0&-\infty\\0&0\end{bmatrix},\qquad | E06.02 |
| 06ZH-015 | [L54](../../docs/book/06_attention_cache.md#L54) · 从“找相关内容”到两个位置的手算 | 公式 | O=AV=\begin{bmatrix}10&0\\6.698&3.302\end{bmatrix}. | E06.02 |
| 06ZH-016 | [L58](../../docs/book/06_attention_cache.md#L58) · 从“找相关内容”到两个位置的手算 | 段落 | 位置 1 的输出完全来自第一个 value；位置 2 混合了两个 value。这里保留对角线是正确的：输入位置 \(t\) 的表示包含 \(x_t\)，用于预测 \(x_{t+1}\)。不能把“看见当 | E06.02 |
| 06ZH-017 | [L62](../../docs/book/06_attention_cache.md#L62) · KV cache 保存什么，又没有保存什么 | 段落 | 在普通 causal attention 中，生成新 token 时只需要计算它自己的 \(q,k,v\)，并把新的 \(k,v\) 追加到缓存。历史 key/value 保留，新的 query 与当 | E06.03 |
| 06ZH-018 | [L64](../../docs/book/06_attention_cache.md#L64) · KV cache 保存什么，又没有保存什么 | 段落 | 为便于估算，批量推理常用概念布局 \([B,n_{\rm kv},L,d_h]\)，其中 \(B\) 是 session 数，\(n_{\rm kv}\) 是 KV head 数。但本仓库的 KVCa | E06.03 |
| 06ZH-019 | [L66](../../docs/book/06_attention_cache.md#L66) · KV cache 保存什么，又没有保存什么 | 段落 | 若 key/value 每元素占 \(b\) 字节，所有层长度与 head 配置相同，仅 KV 数据的估算为 | E06.03 |
| 06ZH-020 | [L68](../../docs/book/06_attention_cache.md#L68) · KV cache 保存什么，又没有保存什么 | 公式 | M_{\rm KV}=2B\,n_{\rm layers}\,n_{\rm kv}\,L\,d_h\,b. | E06.03 |
| 06ZH-021 | [L72](../../docs/book/06_attention_cache.md#L72) · KV cache 保存什么，又没有保存什么 | 段落 | 例如 \(B=1\)、\(n_{\rm kv}=2\)、\(L=4096\)、\(d_h=64\)、\(b=2\)，每层为 \(2{,}097{,}152\) 字节，即 \(2\) MiB；12 层合 | E06.03 |
| 06ZH-022 | [L74](../../docs/book/06_attention_cache.md#L74) · KV cache 保存什么，又没有保存什么 | 段落 | 还要区分三个长度：已观察 token 数、已分配 cache capacity、当前 attention 可见窗口。它们可能不同。TTTSession 会增长缓存容量，sliding attentio | E06.03 |
| 06ZH-023 | [L78](../../docs/book/06_attention_cache.md#L78) · Prefill 和 decode 必须遵守同一条时间线 | 段落 | Prefill 表示处理一个已经给定的前缀，decode 表示处理随后到来的单个 token；它们描述的是调用方式，不是两套因果规则。本仓库 prefill 与 decode 最终都走 observe | E06.04 |
| 06ZH-024 | [L80](../../docs/book/06_attention_cache.md#L80) · Prefill 和 decode 必须遵守同一条时间线 | 段落 | 继续上一章 \(C=4\) 的例子。四个输入 \(x_1,x_2,x_3,x_4\) 的预测标签是 \(x_2,x_3,x_4,x_5\)。因此读完 \(x_4\) 时，最后一个标签仍未到达。下面假设 | E06.04 |
| 06ZH-025 | [L82](../../docs/book/06_attention_cache.md#L82) · Prefill 和 decode 必须遵守同一条时间线 | 表格 | ／ 调用完成后 ／ position ／ pending_count ／ 待更新标签 ／ fast weights ／ updates ／ 有效缓存 ／ | E06.04 |
| 06ZH-026 | [L89](../../docs/book/06_attention_cache.md#L89) · Prefill 和 decode 必须遵守同一条时间线 | 段落 | 收到 \(x_5\) 时，程序先用它补齐前一 chunk 的标签并完成更新，然后用 \(\phi_1\) 处理 \(x_5\)，预测 \(x_6\)。这没有让 \(x_5\) 泄漏进对 \(x_5\) | E06.04 |
| 06ZH-027 | [L91](../../docs/book/06_attention_cache.md#L91) · Prefill 和 decode 必须遵守同一条时间线 | 段落 | 历史 suffix cache 可能由较早的快权重产生。本实现保留这些结果，不在每次更新后重新编码整个历史。因此，“拿最新 \(\phi\) 从头运行整段文本”不必与 session 相等。应比较的是 | E06.04 |
| 06ZH-028 | [L95](../../docs/book/06_attention_cache.md#L95) · 一个 session 是一组共同演化的状态 | 段落 | 仅保存 KV cache 不足以恢复 TTT 推理。下面这些字段共同决定下一次调用的结果： | E06.05 |
| 06ZH-029 | [L97](../../docs/book/06_attention_cache.md#L97) · 一个 session 是一组共同演化的状态 | 表格 | ／ 状态 ／ 为什么需要 ／ | E06.05 |
| 06ZH-030 | [L106](../../docs/book/06_attention_cache.md#L106) · 一个 session 是一组共同演化的状态 | 段落 | ttt/session_store.py 把数组和元数据一起保存。加载时检查模型版本、模型配置、内循环设置、数组形状与 dtype、token 范围、位置和 chunk 边界。其格式版本为 2，并且恢 | E06.05 |
| 06ZH-031 | [L108](../../docs/book/06_attention_cache.md#L108) · 一个 session 是一组共同演化的状态 | 段落 | 若要检查恢复正确性，应该在 chunk 中间保存，再把同一后缀分别交给原 session 和恢复后的 session，比较 logits、快权重、更新次数和位置；还应覆盖恰好等满 chunk 但最后标 | E06.05 |
| 06ZH-032 | [L112](../../docs/book/06_attention_cache.md#L112) · 从实现位置走到可检查的证据 | 表格 | ／ 问题 ／ 代码位置 ／ | E06.06 |
| 06ZH-033 | [L120](../../docs/book/06_attention_cache.md#L120) · 从实现位置走到可检查的证据 | 段落 | 完成环境章节的安装后，在远程 GPU 机器的仓库根目录运行以下命令。此处不要求在本地电脑运行测试。 | E06.06 |
| 06ZH-034 | [L122](../../docs/book/06_attention_cache.md#L122) · 从实现位置走到可检查的证据 | 命令 | bash scripts/validate_gpu.sh outputs/ch18/validation.json | E06.06 |
| 06ZH-035 | [L126](../../docs/book/06_attention_cache.md#L126) · 从实现位置走到可检查的证据 | 段落 | 成功时生成 validation.json 与同目录的 environment.json；前者的顶层 passed 为 true。每个 case 包含 logits_max_error 和 inner | E06.06 |
| 06ZH-036 | [L128](../../docs/book/06_attention_cache.md#L128) · 从实现位置走到可检查的证据 | 段落 | 若流式差异失败，先检查 token 提交顺序、chunk 标签偏移和更新时点；若只在跨窗口或缓存扩容后失败，再检查 positions、capacity 和 mask。若尚未进入 case 就报 CU | E06.06 |
| 06ZH-037 | [L130](../../docs/book/06_attention_cache.md#L130) · 从实现位置走到可检查的证据 | 段落 | 本章完成了从 attention 公式到 session 时间线的闭环。下一章会把同一个 hidden state 交给多个候选专家：除了读什么历史，还需要回答“由哪一组参数处理这个 token”。 | E06.06 |

### docs/book/en/06_attention_cache.md

| 单元 | 位置与小节 | 类型 | 首句/首式或命令 | 核对依据 |
|---|---|---|---|---|
| 06EN-001 | [L5](../../docs/book/en/06_attention_cache.md#L5) · Why generating a sentence requires state | 段落 | The previous chapter established when fast weights change. We now change the execution pattern from  | E06.01 |
| 06EN-002 | [L7](../../docs/book/en/06_attention_cache.md#L7) · Why generating a sentence requires state | 段落 | We retain the tiny example with hidden width \(d=16\) and TTT chunk length \(C=4\). First we calcula | E06.01 |
| 06EN-003 | [L9](../../docs/book/en/06_attention_cache.md#L9) · Why generating a sentence requires state | 图与图注 | Figure 4. From sequence processing to token processing. Follow the new token to its query and follow | E06.01 |
| 06EN-004 | [L16](../../docs/book/en/06_attention_cache.md#L16) · From retrieving relevant information to two positions | 段落 | Attention is a weighted retrieval operation. A query represents what the current position needs, a k | E06.02 |
| 06EN-005 | [L18](../../docs/book/en/06_attention_cache.md#L18) · From retrieving relevant information to two positions | 公式 | Q=HW_Q,\qquad K=HW_K,\qquad V=HW_V. | E06.02 |
| 06EN-006 | [L22](../../docs/book/en/06_attention_cache.md#L22) · From retrieving relevant information to two positions | 段落 | Here \(H\in\mathbb R^{L\times d}\) contains one sequence's hidden states, \(L\) is its length, and o | E06.02 |
| 06EN-007 | [L24](../../docs/book/en/06_attention_cache.md#L24) · From retrieving relevant information to two positions | 段落 | Set \(L=d_h=2\), using already projected matrices: | E06.02 |
| 06EN-008 | [L26](../../docs/book/en/06_attention_cache.md#L26) · From retrieving relevant information to two positions | 公式 | Q=\begin{bmatrix}1&0\\1&0\end{bmatrix},\quad | E06.02 |
| 06EN-009 | [L32](../../docs/book/en/06_attention_cache.md#L32) · From retrieving relevant information to two positions | 段落 | Before masking, the scores are | E06.02 |
| 06EN-010 | [L34](../../docs/book/en/06_attention_cache.md#L34) · From retrieving relevant information to two positions | 公式 | S=\frac{QK^\top}{\sqrt2} | E06.02 |
| 06EN-011 | [L39](../../docs/book/en/06_attention_cache.md#L39) · From retrieving relevant information to two positions | 段落 | Position 1 must not read position 2 in an autoregressive language model. Otherwise, an input from th | E06.02 |
| 06EN-012 | [L41](../../docs/book/en/06_attention_cache.md#L41) · From retrieving relevant information to two positions | 表格 | ／ Query position \ historical position ／ \(s=1\) ／ \(s=2\) ／ | E06.02 |
| 06EN-013 | [L46](../../docs/book/en/06_attention_cache.md#L46) · From retrieving relevant information to two positions | 段落 | Write this additive attention mask as \(\mathcal M\), distinguishing it from the data chapters' loss | E06.02 |
| 06EN-014 | [L48](../../docs/book/en/06_attention_cache.md#L48) · From retrieving relevant information to two positions | 公式 | \mathcal M=\begin{bmatrix}0&-\infty\\0&0\end{bmatrix},\qquad | E06.02 |
| 06EN-015 | [L54](../../docs/book/en/06_attention_cache.md#L54) · From retrieving relevant information to two positions | 公式 | O=AV=\begin{bmatrix}10&0\\6.698&3.302\end{bmatrix}. | E06.02 |
| 06EN-016 | [L58](../../docs/book/en/06_attention_cache.md#L58) · From retrieving relevant information to two positions | 段落 | Position 1 uses only the first value; position 2 mixes both. Keeping the diagonal is correct: the re | E06.02 |
| 06EN-017 | [L62](../../docs/book/en/06_attention_cache.md#L62) · What a KV cache does and does not store | 段落 | In ordinary causal attention, a new token needs its own \(q,k,v\); its key and value are appended to | E06.03 |
| 06EN-018 | [L64](../../docs/book/en/06_attention_cache.md#L64) · What a KV cache does and does not store | 段落 | For memory estimates, batched inference is often described with the conceptual layout \([B,n_{\rm kv | E06.03 |
| 06EN-019 | [L66](../../docs/book/en/06_attention_cache.md#L66) · What a KV cache does and does not store | 段落 | If each key/value element uses \(b\) bytes, and all layers share the same length and head configurat | E06.03 |
| 06EN-020 | [L68](../../docs/book/en/06_attention_cache.md#L68) · What a KV cache does and does not store | 公式 | M_{\rm KV}=2B\,n_{\rm layers}\,n_{\rm kv}\,L\,d_h\,b. | E06.03 |
| 06EN-021 | [L72](../../docs/book/en/06_attention_cache.md#L72) · What a KV cache does and does not store | 段落 | With \(B=1\), \(n_{\rm kv}=2\), \(L=4096\), \(d_h=64\), and \(b=2\), this is \(2{,}097{,}152\) bytes | E06.03 |
| 06EN-022 | [L74](../../docs/book/en/06_attention_cache.md#L74) · What a KV cache does and does not store | 段落 | Three lengths must remain distinct: observed token count, allocated cache capacity, and the attentio | E06.03 |
| 06EN-023 | [L78](../../docs/book/en/06_attention_cache.md#L78) · Prefill and decode follow the same timeline | 段落 | Prefill processes an already supplied prefix; decode processes a subsequently arriving token. These  | E06.04 |
| 06EN-024 | [L80](../../docs/book/en/06_attention_cache.md#L80) · Prefill and decode follow the same timeline | 段落 | Continue with \(C=4\). The four inputs \(x_1,x_2,x_3,x_4\) have targets \(x_2,x_3,x_4,x_5\). After o | E06.04 |
| 06EN-025 | [L82](../../docs/book/en/06_attention_cache.md#L82) · Prefill and decode follow the same timeline | 表格 | ／ After the call ／ position ／ pending_count ／ Pending targets ／ Fast weights ／ updates ／ Valid cache | E06.04 |
| 06EN-026 | [L89](../../docs/book/en/06_attention_cache.md#L89) · Prefill and decode follow the same timeline | 段落 | When \(x_5\) arrives, the program first completes the previous chunk's labels, performs its update,  | E06.04 |
| 06EN-027 | [L91](../../docs/book/en/06_attention_cache.md#L91) · Prefill and decode follow the same timeline | 段落 | Historical suffix caches may have been produced under earlier fast weights. This implementation reta | E06.04 |
| 06EN-028 | [L95](../../docs/book/en/06_attention_cache.md#L95) · A session is jointly evolving state | 段落 | Saving KV alone cannot restore TTT inference. The following fields jointly determine the next call: | E06.05 |
| 06EN-029 | [L97](../../docs/book/en/06_attention_cache.md#L97) · A session is jointly evolving state | 表格 | ／ State ／ Why it matters ／ | E06.05 |
| 06EN-030 | [L106](../../docs/book/en/06_attention_cache.md#L106) · A session is jointly evolving state | 段落 | ttt/session_store.py saves arrays together with metadata. Loading checks model revision, configurati | E06.05 |
| 06EN-031 | [L108](../../docs/book/en/06_attention_cache.md#L108) · A session is jointly evolving state | 段落 | A meaningful restoration check saves in the middle of a chunk, sends the same suffix through the ori | E06.05 |
| 06EN-032 | [L112](../../docs/book/en/06_attention_cache.md#L112) · Connecting implementation to evidence | 表格 | ／ Question ／ Code location ／ | E06.06 |
| 06EN-033 | [L120](../../docs/book/en/06_attention_cache.md#L120) · Connecting implementation to evidence | 段落 | After completing environment setup, run this command at the repository root on a remote GPU machine. | E06.06 |
| 06EN-034 | [L122](../../docs/book/en/06_attention_cache.md#L122) · Connecting implementation to evidence | 命令 | bash scripts/validate_gpu.sh outputs/ch18/validation.json | E06.06 |
| 06EN-035 | [L126](../../docs/book/en/06_attention_cache.md#L126) · Connecting implementation to evidence | 段落 | Success produces validation.json and a sibling environment.json; the former has top-level passed: tr | E06.06 |
| 06EN-036 | [L128](../../docs/book/en/06_attention_cache.md#L128) · Connecting implementation to evidence | 段落 | For a streaming mismatch, first inspect token order, target offsets, and update timing. If failure o | E06.06 |
| 06EN-037 | [L130](../../docs/book/en/06_attention_cache.md#L130) · Connecting implementation to evidence | 段落 | We have connected the attention equations to a session timeline. The next chapter adds another decis | E06.06 |

### docs/book/07_moe.md

| 单元 | 位置与小节 | 类型 | 首句/首式或命令 | 核对依据 |
|---|---|---|---|---|
| 07ZH-001 | [L5](../../docs/book/07_moe.md#L5) · 参数更多，为什么未必每次都算得更多 | 段落 | 上一章讨论 token 如何读取历史。现在保留 attention，改变后面的前馈网络：准备多组专家参数，让每个 token 只选择其中少数几组。一个专家通常就是一个 MLP；决定选谁的小网络叫 ro | E07.01 |
| 07ZH-002 | [L7](../../docs/book/07_moe.md#L7) · 参数更多，为什么未必每次都算得更多 | 段落 | 这使“模型存了多少参数”和“一个 token 激活多少参数”成为不同的问题。设有 \(n_{\rm exp}\) 个 SwiGLU 专家，每个专家隐藏宽度为 \(d\)、中间宽度为 \(d_{\rm  | E07.01 |
| 07ZH-003 | [L9](../../docs/book/07_moe.md#L9) · 参数更多，为什么未必每次都算得更多 | 段落 | 本章先让四个 token 在两个专家之间选一个，手工追踪排序和还原；再扩展到仓库使用的 top-2 和四专家双卡验证。读者需要上一章的 hidden state 与 softmax，以及前馈层的矩阵形 | E07.01 |
| 07ZH-004 | [L11](../../docs/book/07_moe.md#L11) · 参数更多，为什么未必每次都算得更多 | 图与图注 | 图 5　一次 MoE 前向计算。 请同时追踪 token 身份与专家分组：计算时可以重排，返回残差流前必须恢复。图中的容量概念用于解释常见设计；本仓库当前参考路径不会按容量丢弃 token。 | E07.01 |
| 07ZH-005 | [L18](../../docs/book/07_moe.md#L18) · 从路由分数到加权输出 | 段落 | 对 token 的 hidden state \(h_t\in\mathbb R^d\)，router 产生 \(n_{\rm exp}\) 个 logits \(r_{t,e}\)，再用 softm | E07.02 |
| 07ZH-006 | [L20](../../docs/book/07_moe.md#L20) · 从路由分数到加权输出 | 公式 | r_t=W_rh_t,\qquad | E07.02 |
| 07ZH-007 | [L25](../../docs/book/07_moe.md#L25) · 从路由分数到加权输出 | 段落 | 数学式采用列向量约定；代码通常把 token 放在矩阵的行上，由 linear 处理对应转置。当前实现用 \(r_{t,e}+b_e\) 选择 top-\(k_{\rm sel}\)，其中 \(b_e | E07.02 |
| 07ZH-008 | [L27](../../docs/book/07_moe.md#L27) · 从路由分数到加权输出 | 公式 | S_t=\operatorname{TopK}(r_t+b,k_{\rm sel}),\qquad | E07.02 |
| 07ZH-009 | [L34](../../docs/book/07_moe.md#L34) · 从路由分数到加权输出 | 段落 | \(F_e\) 是第 \(e\) 个专家。选择分数与混合权重必须分开理解：负载偏置可以改变谁被选中，但不是直接把同一个偏置加进混合 softmax。以下例子取 \(b=0\)，所以按概率排序与按 lo | E07.02 |
| 07ZH-010 | [L36](../../docs/book/07_moe.md#L36) · 从路由分数到加权输出 | 表格 | ／ 原 token 下标（代码） ／ \(p_{t,1}\) ／ \(p_{t,2}\) ／ top-1 专家 ／ | E07.02 |
| 07ZH-011 | [L43](../../docs/book/07_moe.md#L43) · 从路由分数到加权输出 | 段落 | 这里只选一个专家，所选权重重新归一化后为 1。两个专家分别收到 token \([0,2]\) 与 \([1,3]\)。专家分组后的行顺序是 \([0,2,1,3]\)，分组大小是 \([2,2]\) | E07.02 |
| 07ZH-012 | [L45](../../docs/book/07_moe.md#L45) · 从路由分数到加权输出 | 段落 | top-2 时，每个 token 产生两条“token—专家”分配，可能暂时复制输入行。例如三个专家概率为 \((0.6,0.3,0.1)\)，选前两者后权重变为 \((2/3,1/3)\)。若对应专 | E07.02 |
| 07ZH-013 | [L47](../../docs/book/07_moe.md#L47) · 从路由分数到加权输出 | 段落 | top-1 例子只用于解释重排。在本实现的归一化规则下，单个所选权重恒为 1，离散选择本身也没有通常意义下的梯度，因此该路径上的语言模型损失不会通过混合权重训练 router；辅助项仍可提供梯度。不能 | E07.02 |
| 07ZH-014 | [L51](../../docs/book/07_moe.md#L51) · 容量与负载：两个不同的问题 | 段落 | 若所有 token 都偏爱同一专家，逻辑上仍可逐个处理，但该专家可能成为计算和显存瓶颈。带固定容量的实现常用 | E07.03 |
| 07ZH-015 | [L53](../../docs/book/07_moe.md#L53) · 容量与负载：两个不同的问题 | 公式 | C_{\rm exp}= | E07.03 |
| 07ZH-016 | [L58](../../docs/book/07_moe.md#L58) · 容量与负载：两个不同的问题 | 段落 | 给每个专家分配槽位，\(\gamma\) 为 capacity factor，\(T_{\rm tok}\) 是一次路由中的 token 数。上例 \(T_{\rm tok}=4\)、\(k_{\rm | E07.03 |
| 07ZH-017 | [L60](../../docs/book/07_moe.md#L60) · 容量与负载：两个不同的问题 | 段落 | 这是常见设计空间，不能误写为本仓库已有的功能。moe_forward 当前采用无丢弃分组路径：按实际分配计数构造 group_sizes，没有 capacity factor 配置，也没有自动丢弃溢出 | E07.03 |
| 07ZH-018 | [L62](../../docs/book/07_moe.md#L62) · 容量与负载：两个不同的问题 | 段落 | 为鼓励分配分散，代码计算平均路由概率 \(\bar p_e\) 和分配频率 \(\rho_e\)。若所有 token 都有效，则 | E07.03 |
| 07ZH-019 | [L64](../../docs/book/07_moe.md#L64) · 容量与负载：两个不同的问题 | 公式 | \bar p_e=\frac1{T_{\rm tok}}\sum_t p_{t,e},\qquad | E07.03 |
| 07ZH-020 | [L72](../../docs/book/07_moe.md#L72) · 容量与负载：两个不同的问题 | 段落 | 有 mask 时，分子分母都按有效 token 权重计数。四 token 例子中 \(\bar p=(0.55,0.45)\)、\(\rho=(0.5,0.5)\)，所以 \(\mathcal L_{ | E07.03 |
| 07ZH-021 | [L74](../../docs/book/07_moe.md#L74) · 容量与负载：两个不同的问题 | 段落 | 代码还计算 \(\mathcal L_z=T_{\rm tok}^{-1}\sum_t(\log\sum_e e^{r_{t,e}})^2\)，有 mask 时同样取加权平均，用于约束 router  | E07.03 |
| 07ZH-022 | [L78](../../docs/book/07_moe.md#L78) · 为什么要把小矩阵乘法分组 | 段落 | 路由之后若逐 token 调用 MLP，会产生许多很小的矩阵乘法，难以充分利用 GPU。分组计算先把属于同一专家的输入排在一起，再对各组应用对应的专家矩阵。对四 token 例子，逻辑上是 | E07.04 |
| 07ZH-023 | [L80](../../docs/book/07_moe.md#L80) · 为什么要把小矩阵乘法分组 | 公式 | \begin{bmatrix}h_0\\h_2\end{bmatrix}W_1,\qquad | E07.04 |
| 07ZH-024 | [L85](../../docs/book/07_moe.md#L85) · 为什么要把小矩阵乘法分组 | 段落 | Grouped GEMM 把“多组输入乘各自权重”的描述交给底层算子，而不是把专家参数混成一个普通 dense 层。moe.py 先稳定排序专家 ID，得到 token 索引和分组大小；计算 SwiG | E07.04 |
| 07ZH-025 | [L87](../../docs/book/07_moe.md#L87) · 为什么要把小矩阵乘法分组 | 段落 | 但“代码用了 grouped GEMM”仍不等于“高阶训练只计算激活专家的 FLOPs”。grouped_gemm.py 的普通原始前向使用 jax.lax.ragged_dot；为支持批处理及高阶求 | E07.04 |
| 07ZH-026 | [L91](../../docs/book/07_moe.md#L91) · 两张 GPU 上，专家到底分在哪里 | 段落 | 仓库 tiny mesh 验证有四个专家，\(d=d_{\rm ff}=16\)。按代码的权重存储约定，一组 gate 权重全局形状为 \([4,16,16]\)，维度依次是专家、输出、输入。使用两张 | E07.05 |
| 07ZH-027 | [L93](../../docs/book/07_moe.md#L93) · 两张 GPU 上，专家到底分在哪里 | 表格 | ／ 对象 ／ 全局形状 ／ GPU 0 本地形状 ／ GPU 1 本地形状 ／ | E07.05 |
| 07ZH-028 | [L99](../../docs/book/07_moe.md#L99) · 两张 GPU 上，专家到底分在哪里 | 段落 | 这里 mesh 是 \(\text{data}=1,\text{expert}=2\)，所以输入没有沿 data 轴再切成两份。五个 token 提供四个 next-token 目标；这是上一章标签对 | E07.05 |
| 07ZH-029 | [L101](../../docs/book/07_moe.md#L101) · 两张 GPU 上，专家到底分在哪里 | 段落 | 远程作业应恰好有两张可见 GPU，且环境已经安装，再在仓库根目录运行。脚本把所有可见设备放入 mesh；八张可见卡会形成 data=4、expert=2，而内置 batch=2 无法沿 data 轴切 | E07.05 |
| 07ZH-030 | [L103](../../docs/book/07_moe.md#L103) · 两张 GPU 上，专家到底分在哪里 | 命令 | bash scripts/validate_mesh.sh \ | E07.05 |
| 07ZH-031 | [L109](../../docs/book/07_moe.md#L109) · 两张 GPU 上，专家到底分在哪里 | 段落 | 预期报告含 passed: true、scale: "tiny"、mesh、expert_shards 和 checkpoint_roundtrip: true。正好两张 GPU 时应看到 data= | E07.05 |
| 07ZH-032 | [L111](../../docs/book/07_moe.md#L111) · 两张 GPU 上，专家到底分在哪里 | 段落 | 如果设备数不能被专家并行度整除，先查实际可见设备和 mesh 配置；若专家矩阵不能切分，查专家数与分片轴；若恢复失败，先查检查点目录及原始报错，不要绕过恢复检查。成功只验证这一短序列、该精度和该拓扑的 | E07.05 |
| 07ZH-033 | [L115](../../docs/book/07_moe.md#L115) · 在同一个框架中区分几种研究问题 | 段落 | moe.py 还允许在 router logits 上加入 fast router 修正或 Delta memory 读出，允许共享专家输出参与残差。三条适应路线应明确区分：mlp 更新的是 MoE  | E07.06 |
| 07ZH-034 | [L117](../../docs/book/07_moe.md#L117) · 在同一个框架中区分几种研究问题 | 段落 | 这些开关改变了适应位置或状态机制，不应一次全部打开后，把结果概括为“MoE 更好”。从上述固定路由例子出发，先验证普通 MoE 的分组与还原，再分别比较稠密 MLP′ 适应、路由适配器更新、记忆路由和 | E07.06 |
| 07ZH-035 | [L119](../../docs/book/07_moe.md#L119) · 在同一个框架中区分几种研究问题 | 段落 | 本章的核心证据链是：分数决定选择，选择决定分配，分配决定分组，分组结果还原为 token 输出；负载与成本必须沿着这条链一起测量。下一章会把质量与成本放到同一张实验表中。 | E07.06 |

### docs/book/en/07_moe.md

| 单元 | 位置与小节 | 类型 | 首句/首式或命令 | 核对依据 |
|---|---|---|---|---|
| 07EN-001 | [L5](../../docs/book/en/07_moe.md#L5) · More stored parameters need not mean more work per token | 段落 | The previous chapter explained how a token reads history. We now keep attention and change the feed- | E07.01 |
| 07EN-002 | [L7](../../docs/book/en/07_moe.md#L7) · More stored parameters need not mean more work per token | 段落 | Stored parameter count and activated parameter count are different quantities. With \(n_{\rm exp}\)  | E07.01 |
| 07EN-003 | [L9](../../docs/book/en/07_moe.md#L9) · More stored parameters need not mean more work per token | 段落 | We first route four tokens to one of two experts and trace sorting and restoration by hand. Then we  | E07.01 |
| 07EN-004 | [L11](../../docs/book/en/07_moe.md#L11) · More stored parameters need not mean more work per token | 图与图注 | Figure 5. One MoE forward pass. Track both token identity and expert grouping: computation may reord | E07.01 |
| 07EN-005 | [L18](../../docs/book/en/07_moe.md#L18) · From routing scores to a weighted output | 段落 | For hidden state \(h_t\in\mathbb R^d\), the router produces \(n_{\rm exp}\) logits and converts them | E07.02 |
| 07EN-006 | [L20](../../docs/book/en/07_moe.md#L20) · From routing scores to a weighted output | 公式 | r_t=W_rh_t,\qquad | E07.02 |
| 07EN-007 | [L25](../../docs/book/en/07_moe.md#L25) · From routing scores to a weighted output | 段落 | This equation uses column vectors. Code generally places tokens in matrix rows, with linear handling | E07.02 |
| 07EN-008 | [L27](../../docs/book/en/07_moe.md#L27) · From routing scores to a weighted output | 公式 | S_t=\operatorname{TopK}(r_t+b,k_{\rm sel}),\qquad | E07.02 |
| 07EN-009 | [L34](../../docs/book/en/07_moe.md#L34) · From routing scores to a weighted output | 段落 | \(F_e\) denotes expert \(e\). Selection scores and mixing weights are distinct: a load bias can chan | E07.02 |
| 07EN-010 | [L36](../../docs/book/en/07_moe.md#L36) · From routing scores to a weighted output | 表格 | ／ Original token index (code) ／ \(p_{t,1}\) ／ \(p_{t,2}\) ／ Top-1 expert ／ | E07.02 |
| 07EN-011 | [L43](../../docs/book/en/07_moe.md#L43) · From routing scores to a weighted output | 段落 | A single selected expert receives normalized weight 1. The two experts receive tokens \([0,2]\) and  | E07.02 |
| 07EN-012 | [L45](../../docs/book/en/07_moe.md#L45) · From routing scores to a weighted output | 段落 | With top-2, each token produces two token–expert assignments, temporarily duplicating input rows. Fo | E07.02 |
| 07EN-013 | [L47](../../docs/book/en/07_moe.md#L47) · From routing scores to a weighted output | 段落 | Top-1 here is a sorting example. Under this implementation's renormalization, its selected weight is | E07.02 |
| 07EN-014 | [L51](../../docs/book/en/07_moe.md#L51) · Capacity and load are different questions | 段落 | When every token prefers the same expert, sequential processing remains logically possible, but that | E07.03 |
| 07EN-015 | [L53](../../docs/book/en/07_moe.md#L53) · Capacity and load are different questions | 公式 | C_{\rm exp}= | E07.03 |
| 07EN-016 | [L58](../../docs/book/en/07_moe.md#L58) · Capacity and load are different questions | 段落 | slots per expert, where \(\gamma\) is the capacity factor and \(T_{\rm tok}\) the token count in one | E07.03 |
| 07EN-017 | [L60](../../docs/book/en/07_moe.md#L60) · Capacity and load are different questions | 段落 | This is design context, not an existing repository feature. moe_forward currently groups assignments | E07.03 |
| 07EN-018 | [L62](../../docs/book/en/07_moe.md#L62) · Capacity and load are different questions | 段落 | To encourage distributed assignments, code computes average routing probability \(\bar p_e\) and ass | E07.03 |
| 07EN-019 | [L64](../../docs/book/en/07_moe.md#L64) · Capacity and load are different questions | 公式 | \bar p_e=\frac1{T_{\rm tok}}\sum_t p_{t,e},\qquad | E07.03 |
| 07EN-020 | [L72](../../docs/book/en/07_moe.md#L72) · Capacity and load are different questions | 段落 | With a mask, numerator and denominator use valid-token weights. Our four-token example has \(\bar p= | E07.03 |
| 07EN-021 | [L74](../../docs/book/en/07_moe.md#L74) · Capacity and load are different questions | 段落 | Code also computes \(\mathcal L_z=T_{\rm tok}^{-1}\sum_t(\log\sum_e e^{r_{t,e}})^2\), again mask-wei | E07.03 |
| 07EN-022 | [L78](../../docs/book/en/07_moe.md#L78) · Why group the matrix multiplications | 段落 | Calling an MLP separately for every routed token creates many small matrix multiplications that may  | E07.04 |
| 07EN-023 | [L80](../../docs/book/en/07_moe.md#L80) · Why group the matrix multiplications | 公式 | \begin{bmatrix}h_0\\h_2\end{bmatrix}W_1,\qquad | E07.04 |
| 07EN-024 | [L85](../../docs/book/en/07_moe.md#L85) · Why group the matrix multiplications | 段落 | Grouped GEMM describes several input groups multiplied by their respective weights. It does not merg | E07.04 |
| 07EN-025 | [L87](../../docs/book/en/07_moe.md#L87) · Why group the matrix multiplications | 段落 | However, using grouped GEMM does not establish that higher-order training computes only selected exp | E07.04 |
| 07EN-026 | [L91](../../docs/book/en/07_moe.md#L91) · Where the experts live on two GPUs | 段落 | The tiny mesh validation has four experts with \(d=d_{\rm ff}=16\). Under code's weight storage conv | E07.05 |
| 07EN-027 | [L93](../../docs/book/en/07_moe.md#L93) · Where the experts live on two GPUs | 表格 | ／ Object ／ Global shape ／ GPU 0 local shape ／ GPU 1 local shape ／ | E07.05 |
| 07EN-028 | [L99](../../docs/book/en/07_moe.md#L99) · Where the experts live on two GPUs | 段落 | The mesh is \(\text{data}=1,\text{expert}=2\), so inputs are not split again along data. Five tokens | E07.05 |
| 07EN-029 | [L101](../../docs/book/en/07_moe.md#L101) · Where the experts live on two GPUs | 段落 | After installing the environment, run from the repository root with exactly two GPUs visible to the  | E07.05 |
| 07EN-030 | [L103](../../docs/book/en/07_moe.md#L103) · Where the experts live on two GPUs | 命令 | bash scripts/validate_mesh.sh \ | E07.05 |
| 07EN-031 | [L109](../../docs/book/en/07_moe.md#L109) · Where the experts live on two GPUs | 段落 | The expected report contains passed: true, scale: "tiny", mesh, expert_shards, and checkpoint_roundt | E07.05 |
| 07EN-032 | [L111](../../docs/book/en/07_moe.md#L111) · Where the experts live on two GPUs | 段落 | If the device count is not divisible by expert parallelism, inspect visible devices and mesh setting | E07.05 |
| 07EN-033 | [L115](../../docs/book/en/07_moe.md#L115) · Separate the research questions | 段落 | moe.py also allows fast-router corrections or Delta-memory reads to modify logits, and shared-expert | E07.06 |
| 07EN-034 | [L117](../../docs/book/en/07_moe.md#L117) · Separate the research questions | 段落 | These switches alter adaptation location or state mechanisms. Enabling all of them at once does not  | E07.06 |
| 07EN-035 | [L119](../../docs/book/en/07_moe.md#L119) · Separate the research questions | 段落 | The evidence chain is now concrete: scores determine selections, selections determine assignments, a | E07.06 |

### docs/book/08_experiments.md

| 单元 | 位置与小节 | 类型 | 首句/首式或命令 | 核对依据 |
|---|---|---|---|---|
| 08ZH-001 | [L5](../../docs/book/08_experiments.md#L5) · 从“跑出了结果”走到“回答了问题” | 段落 | 设两个训练作业都结束了，A 的验证 NLL 是 2.10，B 是 1.98。现在能说 B 的架构更好吗？还不能：B 可能读了更多数据、使用了不同 tokenizer，或者多花了一倍时间。一个实验首先需 | E08.01 |
| 08ZH-002 | [L7](../../docs/book/08_experiments.md#L7) · 从“跑出了结果”走到“回答了问题” | 段落 | 本章把前几章的实现放进一个比较协议。核心问题是：同一架构启用测试时更新后，质量变化有多大，为此付出了什么成本？我们会先定义控制变量，再手算一张质量—成本表，最后说明配置生成器、profile 与 sc | E08.01 |
| 08ZH-003 | [L9](../../docs/book/08_experiments.md#L9) · 从“跑出了结果”走到“回答了问题” | 图与图注 | 图 6　实验的闭环。 从问题开始，沿配置、运行、证据、判断走一圈。每一条结论都应能回到产生它的配置和测量，而不只是回到一个实验名称。 | E08.01 |
| 08ZH-004 | [L16](../../docs/book/08_experiments.md#L16) · 一次只改变哪件事 | 段落 | 前文的 C2/F2（或 C4/F4）对照中，两组保留相同的快 MLP 架构和初始参数，只改变内循环是否更新；名称中的数字表示 full-attention 层数。这样问的是“适应过程是否有效”。若把  | E08.02 |
| 08ZH-005 | [L18](../../docs/book/08_experiments.md#L18) · 一次只改变哪件事 | 段落 | 建议先写下这张协议表，再提交训练。这里的“不变”包括具体文件和哈希，而不只是“都用同一数据集”的口头说明。 | E08.02 |
| 08ZH-006 | [L20](../../docs/book/08_experiments.md#L20) · 一次只改变哪件事 | 表格 | ／ 项目 ／ 更新消融中的约束 ／ 为什么影响归因 ／ | E08.02 |
| 08ZH-007 | [L29](../../docs/book/08_experiments.md#L29) · 一次只改变哪件事 | 段落 | 初始化一致不意味着整条训练轨迹一致；启用更新会改变外循环梯度，后续静态参数当然可能分岔。控制的是起点与协议，不是强行让两组得到相同权重。评估也要独立说明快状态是否跨文档保留：重置状态的单文档性能与连续 | E08.02 |
| 08ZH-008 | [L31](../../docs/book/08_experiments.md#L31) · 一次只改变哪件事 | 段落 | 结构消融可以随后进行，例如去掉快 MLP、改变 suffix 深度、改变 attention 布局、开启 MoE 或重复块。每次变化都重新核算参数和成本；不能沿用最初那组配置的“125M”标签当作精确 | E08.02 |
| 08ZH-009 | [L35](../../docs/book/08_experiments.md#L35) · 用一张手算表同时看质量与成本 | 段落 | 以下数字是**教学假设，不是仓库实验结果**。两组具有相同的 128M 参数，处理同样的 \(D_{\rm valid}=1{,}000{,}000\) 个有效目标 token，均不计编译与数据准备时 | E08.03 |
| 08ZH-010 | [L37](../../docs/book/08_experiments.md#L37) · 用一张手算表同时看质量与成本 | 表格 | ／ 方案 ／ 内循环更新 ／ 验证 NLL ／ 稳态训练有效 token/s ／ 训练时间 ／ 峰值显存 ／ | E08.03 |
| 08ZH-011 | [L42](../../docs/book/08_experiments.md#L42) · 用一张手算表同时看质量与成本 | 段落 | 时间由 \(t=D_{\rm valid}/r\) 得到，例如 \(1{,}000{,}000/12{,}000\approx83.3\) 秒。相同 tokenizer、相同评估目标下，NLL 下降  | E08.03 |
| 08ZH-012 | [L44](../../docs/book/08_experiments.md#L44) · 用一张手算表同时看质量与成本 | 段落 | 如果改为相同 50 秒预算，C 能处理一百万有效 token，而 F 只能处理约 60 万。表中的 1.98 来自 F 处理一百万 token 后，不能挪给 60 万 token 的模型。要回答同时间 | E08.03 |
| 08ZH-013 | [L46](../../docs/book/08_experiments.md#L46) · 用一张手算表同时看质量与成本 | 段落 | 有效 token 数是 \(D_{\rm valid}=\sum_{k,b,t}m_{k,b,t}\)，不是简单的步数乘 batch 乘长度。padding、文档边界屏蔽或其他 mask 会让二者不同 | E08.03 |
| 08ZH-014 | [L48](../../docs/book/08_experiments.md#L48) · 用一张手算表同时看质量与成本 | 段落 | 若约束是训练 FLOPs，设两组每个有效 token 成本为 \(f_C,f_F\)，相同预算要求 \(D_Cf_C\approx D_Ff_F\)。这里的成本必须对应真实的内循环、高阶求导、长度和  | E08.03 |
| 08ZH-015 | [L52](../../docs/book/08_experiments.md#L52) · 为什么一个种子还不够 | 段落 | 训练初值、数据顺序和硬件执行都可能造成波动。为了减少无关变化，可以按种子配对运行 C 和 F，再分析每一对的差值。假设三个配对种子给出 NLL 差值 \(-0.10,-0.12,-0.08\)，它们仍 | E08.04 |
| 08ZH-016 | [L54](../../docs/book/08_experiments.md#L54) · 为什么一个种子还不够 | 公式 | \bar\delta=-0.10,\qquad | E08.04 |
| 08ZH-017 | [L60](../../docs/book/08_experiments.md#L60) · 为什么一个种子还不够 | 段落 | 平均差描述观察到的收益；样本标准差描述种子间波动；标准误描述平均差估计的不确定程度，依赖样本近似独立等假设。三个种子远不足以证明对所有数据域或规模都稳定。不要仅凭这组数画一个没有说明假设的“95% 置 | E08.04 |
| 08ZH-018 | [L62](../../docs/book/08_experiments.md#L62) · 为什么一个种子还不够 | 段落 | 实验记录应保留每个种子的原始结果、失败作业和排除理由。若先看结果再挑选最有利种子，任何随后计算的均值都不再回答最初的比较问题。 | E08.04 |
| 08ZH-019 | [L66](../../docs/book/08_experiments.md#L66) · 生成实验配置，先检查它生成了什么 | 段落 | ttt/experiments.py 生成 dense、hybrid、TTT 及可选 MoE/Loop 消融配置。在已安装环境的远程机器、仓库根目录运行： | E08.05 |
| 08ZH-020 | [L68](../../docs/book/08_experiments.md#L68) · 生成实验配置，先检查它生成了什么 | 命令 | .venv/bin/python -m ttt.experiments \ | E08.05 |
| 08ZH-021 | [L76](../../docs/book/08_experiments.md#L76) · 生成实验配置，先检查它生成了什么 | 段落 | 预期得到 outputs/ch20/configs/manifest.json 和 125m/ 下的各实验 JSON；标准输出也打印 manifest。查看 experiments 中的路径与 sch | E08.05 |
| 08ZH-022 | [L78](../../docs/book/08_experiments.md#L78) · 生成实验配置，先检查它生成了什么 | 段落 | 这个命令只生成配置，不训练模型、不下载 tokenizer、不准备数据。默认实验配方引用 Llama tokenizer，后续训练前仍需验证访问权限、实际 tokenizer 文件与词表匹配。若输出不 | E08.05 |
| 08ZH-023 | [L80](../../docs/book/08_experiments.md#L80) · 生成实验配置，先检查它生成了什么 | 段落 | 成本输入按实验名记录 training_flops_per_token，并注明 method 为 profiler 或 xla_estimate。生成器用两组每 token 成本的比值调整基线步数。测 | E08.05 |
| 08ZH-024 | [L84](../../docs/book/08_experiments.md#L84) · 推理 profile 与训练成本估计分别说明什么 | 段落 | 若上一章的 tiny mesh 验证已经成功，它会留下真实检查点 outputs/ch19/checkpoints/mesh-step-2.orbax。以下命令沿用该检查点，并把上下文与生成长度限制为 | E08.06 |
| 08ZH-025 | [L86](../../docs/book/08_experiments.md#L86) · 推理 profile 与训练成本估计分别说明什么 | 命令 | JAX_PLATFORMS=cuda .venv/bin/python -m ttt.profile \ | E08.06 |
| 08ZH-026 | [L94](../../docs/book/08_experiments.md#L94) · 推理 profile 与训练成本估计分别说明什么 | 段落 | 标准输出是 JSON，包括 parameters、prefill_seconds、decode_seconds、decode_tokens_per_second、state_bytes 和 inner | E08.06 |
| 08ZH-027 | [L96](../../docs/book/08_experiments.md#L96) · 推理 profile 与训练成本估计分别说明什么 | 段落 | profile_session 先完整预热一次，再计时一次，且在读取时间前等待设备结果。它测量单 session 的 prefill 和逐 token decode，不是训练 tokens/s，也不是 | E08.06 |
| 08ZH-028 | [L98](../../docs/book/08_experiments.md#L98) · 推理 profile 与训练成本估计分别说明什么 | 段落 | 训练 FLOPs 字段来自编译图成本分析，不是硬件计数器实测，也不含完整数据输入、优化器、检查点保存的端到端成本；若后端没有提供统计，值可能为 null。若检查点不存在，先完成上一章验证；若长度越界， | E08.06 |
| 08ZH-029 | [L102](../../docs/book/08_experiments.md#L102) · Scaling 是可以被留出规模反驳的假设 | 段落 | 单个规模的胜负不能自动外推到更大的模型。一个常用的经验模型把 loss 写成参数量 \(N\) 和训练目标数 \(D\) 的函数。本仓库 ttt/scaling.py 使用归一化变量： | E08.07 |
| 08ZH-030 | [L104](../../docs/book/08_experiments.md#L104) · Scaling 是可以被留出规模反驳的假设 | 公式 | \widehat{\mathcal L}(N,D) | E08.07 |
| 08ZH-031 | [L111](../../docs/book/08_experiments.md#L111) · Scaling 是可以被留出规模反驳的假设 | 段落 | \(L_\infty\) 是拟合中的损失下限项；\(A,B_s\) 控制两项尺度；\(\alpha,\beta\) 控制随规模与数据量增长的下降速度。下标用于避免把系数 \(B_s\) 与 batch | E08.07 |
| 08ZH-032 | [L113](../../docs/book/08_experiments.md#L113) · Scaling 是可以被留出规模反驳的假设 | 段落 | 拟合器要求至少八条训练结果，至少覆盖三个不同参数规模和三个不同 token 量。每行输入必须提供 parameters、tokens、loss，可用 held_out: true 标出不参与拟合的规模 | E08.07 |
| 08ZH-033 | [L115](../../docs/book/08_experiments.md#L115) · Scaling 是可以被留出规模反驳的假设 | 段落 | 输出包含训练 RMSE、留出规模的预测和 bootstrap 区间。训练误差很小不代表外推可靠；bootstrap 重新采样现有观测，也不会自动涵盖数据污染、实现错误或模型形式错误。应先预留规模，再拟 | E08.07 |
| 08ZH-034 | [L117](../../docs/book/08_experiments.md#L117) · Scaling 是可以被留出规模反驳的假设 | 段落 | 本章没有把一组配置文件当作研究结论。结论需要质量、成本、重复运行和留出规模共同支持。下一章将把“有足够证据启动更大训练”的条件明确写出来。 | E08.07 |

### docs/book/en/08_experiments.md

| 单元 | 位置与小节 | 类型 | 首句/首式或命令 | 核对依据 |
|---|---|---|---|---|
| 08EN-001 | [L5](../../docs/book/en/08_experiments.md#L5) · From completing a run to answering a question | 段落 | Suppose two jobs finish with validation NLLs of 2.10 for A and 1.98 for B. Is B's architecture bette | E08.01 |
| 08EN-002 | [L7](../../docs/book/en/08_experiments.md#L7) · From completing a run to answering a question | 段落 | We place the previous chapters' implementations inside a comparison protocol. The central question i | E08.01 |
| 08EN-003 | [L9](../../docs/book/en/08_experiments.md#L9) · From completing a run to answering a question | 图与图注 | Figure 6. Closing the experimental loop. Follow the question through configuration, execution, evide | E08.01 |
| 08EN-004 | [L16](../../docs/book/en/08_experiments.md#L16) · Decide which one thing changes | 段落 | In the C2/F2 (or C4/F4) comparison, both arms retain the same fast-MLP architecture and initial para | E08.02 |
| 08EN-005 | [L18](../../docs/book/en/08_experiments.md#L18) · Decide which one thing changes | 段落 | Write the protocol before launching training. “The same” should identify concrete files and hashes,  | E08.02 |
| 08EN-006 | [L20](../../docs/book/en/08_experiments.md#L20) · Decide which one thing changes | 表格 | ／ Item ／ Constraint for an update ablation ／ Why it matters ／ | E08.02 |
| 08EN-007 | [L29](../../docs/book/en/08_experiments.md#L29) · Decide which one thing changes | 段落 | Matching initialization does not imply matching the entire training trajectory. Updates change outer | E08.02 |
| 08EN-008 | [L31](../../docs/book/en/08_experiments.md#L31) · Decide which one thing changes | 段落 | Structural ablations can follow: removing the fast MLP, changing suffix depth, modifying attention l | E08.02 |
| 08EN-009 | [L35](../../docs/book/en/08_experiments.md#L35) · Calculate quality and cost together | 段落 | The following numbers are **hypothetical teaching values, not repository results**. Both arms have 1 | E08.03 |
| 08EN-010 | [L37](../../docs/book/en/08_experiments.md#L37) · Calculate quality and cost together | 表格 | ／ Arm ／ Inner updates ／ Validation NLL ／ Steady training valid tokens/s ／ Training time ／ Peak memor | E08.03 |
| 08EN-011 | [L42](../../docs/book/en/08_experiments.md#L42) · Calculate quality and cost together | 段落 | Time is \(t=D_{\rm valid}/r\), so \(1{,}000{,}000/12{,}000\approx83.3\) seconds. With the same token | E08.03 |
| 08EN-012 | [L44](../../docs/book/en/08_experiments.md#L44) · Calculate quality and cost together | 段落 | Under a fixed 50-second budget, C processes one million valid tokens while F processes about 600,000 | E08.03 |
| 08EN-013 | [L46](../../docs/book/en/08_experiments.md#L46) · Calculate quality and cost together | 段落 | Valid-token count is \(D_{\rm valid}=\sum_{k,b,t}m_{k,b,t}\), not simply steps times batch times len | E08.03 |
| 08EN-014 | [L48](../../docs/book/en/08_experiments.md#L48) · Calculate quality and cost together | 段落 | For a training-FLOP budget, let valid-token costs be \(f_C,f_F\). Matching compute requires \(D_Cf_C | E08.03 |
| 08EN-015 | [L52](../../docs/book/en/08_experiments.md#L52) · Why one seed is insufficient | 段落 | Initialization, data order, and execution can introduce variation. Pair C and F by seed to reduce un | E08.04 |
| 08EN-016 | [L54](../../docs/book/en/08_experiments.md#L54) · Why one seed is insufficient | 公式 | \bar\delta=-0.10,\qquad | E08.04 |
| 08EN-017 | [L60](../../docs/book/en/08_experiments.md#L60) · Why one seed is insufficient | 段落 | The mean describes observed improvement; sample standard deviation describes seed variation; standar | E08.04 |
| 08EN-018 | [L62](../../docs/book/en/08_experiments.md#L62) · Why one seed is insufficient | 段落 | Keep per-seed results, failed jobs, and exclusion reasons. Selecting favorable seeds after seeing ou | E08.04 |
| 08EN-019 | [L66](../../docs/book/en/08_experiments.md#L66) · Generate configurations and inspect the result | 段落 | ttt/experiments.py generates dense, hybrid, TTT, and optional MoE/Loop ablations. On a remote machin | E08.05 |
| 08EN-020 | [L68](../../docs/book/en/08_experiments.md#L68) · Generate configurations and inspect the result | 命令 | .venv/bin/python -m ttt.experiments \ | E08.05 |
| 08EN-021 | [L76](../../docs/book/en/08_experiments.md#L76) · Generate configurations and inspect the result | 段落 | Expect outputs/ch20/configs/manifest.json and experiment JSON files under 125m/. Standard output als | E08.05 |
| 08EN-022 | [L78](../../docs/book/en/08_experiments.md#L78) · Generate configurations and inspect the result | 段落 | This command generates configurations. It does not train, download a tokenizer, or prepare data. The | E08.05 |
| 08EN-023 | [L80](../../docs/book/en/08_experiments.md#L80) · Generate configurations and inspect the result | 段落 | Cost input records training_flops_per_token by experiment name and labels method as profiler or xla_ | E08.05 |
| 08EN-024 | [L84](../../docs/book/en/08_experiments.md#L84) · Distinguish inference profiling from training-cost estimates | 段落 | A successful tiny mesh validation in the preceding chapter leaves the real checkpoint outputs/ch19/c | E08.06 |
| 08EN-025 | [L86](../../docs/book/en/08_experiments.md#L86) · Distinguish inference profiling from training-cost estimates | 命令 | JAX_PLATFORMS=cuda .venv/bin/python -m ttt.profile \ | E08.06 |
| 08EN-026 | [L94](../../docs/book/en/08_experiments.md#L94) · Distinguish inference profiling from training-cost estimates | 段落 | Standard output is JSON containing parameters, prefill_seconds, decode_seconds, decode_tokens_per_se | E08.06 |
| 08EN-027 | [L96](../../docs/book/en/08_experiments.md#L96) · Distinguish inference profiling from training-cost estimates | 段落 | profile_session performs one full warmup run and then one measured run, waiting for device results b | E08.06 |
| 08EN-028 | [L98](../../docs/book/en/08_experiments.md#L98) · Distinguish inference profiling from training-cost estimates | 段落 | Training FLOPs come from compiled graph cost analysis, not hardware counters. They do not include th | E08.06 |
| 08EN-029 | [L102](../../docs/book/en/08_experiments.md#L102) · A scaling hypothesis must face held-out scales | 段落 | A result at one size does not automatically extrapolate. An empirical model can express loss as a fu | E08.07 |
| 08EN-030 | [L104](../../docs/book/en/08_experiments.md#L104) · A scaling hypothesis must face held-out scales | 公式 | \widehat{\mathcal L}(N,D) | E08.07 |
| 08EN-031 | [L111](../../docs/book/en/08_experiments.md#L111) · A scaling hypothesis must face held-out scales | 段落 | \(L_\infty\) is the fitted floor term; \(A,B_s\) set scales; \(\alpha,\beta\) describe rates of decr | E08.07 |
| 08EN-032 | [L113](../../docs/book/en/08_experiments.md#L113) · A scaling hypothesis must face held-out scales | 段落 | The fitter requires at least eight training results, three distinct parameter counts, and three dist | E08.07 |
| 08EN-033 | [L115](../../docs/book/en/08_experiments.md#L115) · A scaling hypothesis must face held-out scales | 段落 | Output includes training RMSE, held-out predictions, and bootstrap intervals. Small training error d | E08.07 |
| 08EN-034 | [L117](../../docs/book/en/08_experiments.md#L117) · A scaling hypothesis must face held-out scales | 段落 | A set of configuration files is not a research conclusion. Quality, cost, repeated runs, and held-ou | E08.07 |

### docs/book/09_gpu_validation.md

| 单元 | 位置与小节 | 类型 | 首句/首式或命令 | 核对依据 |
|---|---|---|---|---|
| 09ZH-001 | [L5](../../docs/book/09_gpu_validation.md#L5) · 三种成功，对应三种证据 | 段落 | 一个 tiny 模型在 GPU 上得到有限 loss，只说明这条小计算图可以运行。它没有证明参数按预期分片，没有证明中断后能接着训练，也没有证明目标长度和 batch 能装进显存。扩大实验前，必须先知 | E09.01 |
| 09ZH-002 | [L7](../../docs/book/09_gpu_validation.md#L7) · 三种成功，对应三种证据 | 段落 | 本章沿用前几章的 tiny 模型与双卡专家例子，把证据分成数值正确性、分布式状态正确性、目标配置的运行能力三个层次。读者需要理解张量形状、内外循环和 checkpoint 的用途；这里会从两张 GPU | E09.01 |
| 09ZH-003 | [L9](../../docs/book/09_gpu_validation.md#L9) · 三种成功，对应三种证据 | 表格 | ／ 层次 ／ 要回答的问题 ／ 当前主要入口 ／ | E09.01 |
| 09ZH-004 | [L15](../../docs/book/09_gpu_validation.md#L15) · 三种成功，对应三种证据 | 段落 | 最后一行是一套实验准入要求，不是仓库已经自动实现的“一键认证”。本章会明确列出哪些字段已存在、哪些测量仍需在目标硬件补齐。 | E09.01 |
| 09ZH-005 | [L19](../../docs/book/09_gpu_validation.md#L19) · 先完成单 GPU 功能验证 | 段落 | 在完成环境安装、已经分配 GPU 的远程节点，进入仓库根目录运行： | E09.02 |
| 09ZH-006 | [L21](../../docs/book/09_gpu_validation.md#L21) · 先完成单 GPU 功能验证 | 命令 | bash scripts/validate_gpu.sh outputs/ch21/gpu-validation.json | E09.02 |
| 09ZH-007 | [L25](../../docs/book/09_gpu_validation.md#L25) · 先完成单 GPU 功能验证 | 段落 | 脚本显式要求 CUDA 后端，先写 outputs/ch21/environment.json，再运行 ttt.validate。成功报告顶层包含 passed: true、environment、c | E09.02 |
| 09ZH-008 | [L27](../../docs/book/09_gpu_validation.md#L27) · 先完成单 GPU 功能验证 | 段落 | 每个 case 主要检查三件事。第一，序列 loss 与所有梯度叶子均为有限值；第二，流式 logits 与分块训练参照的最大绝对误差不大于 \(10^{-3}\)；第三，改变后半段输入后，前半段 l | E09.02 |
| 09ZH-009 | [L29](../../docs/book/09_gpu_validation.md#L29) · 先完成单 GPU 功能验证 | 段落 | 这些门槛是当前 float32 tiny case 的实现标准，不能直接套到所有精度和规模。若检查失败，报告会写 passed: false、error_type、error 与环境信息。GPU 未识 | E09.02 |
| 09ZH-010 | [L31](../../docs/book/09_gpu_validation.md#L31) · 先完成单 GPU 功能验证 | 段落 | 通过不代表 32K 或 128K 已验证：默认 context 长度只是 64 和 512。它也不执行 session 快照恢复、真实数据清洗检查或验证集质量评估。把这份报告写进实验记录时，要同时写清 | E09.02 |
| 09ZH-011 | [L35](../../docs/book/09_gpu_validation.md#L35) · 编译时间和执行时间不能混成一个吞吐 | 段落 | JAX 的第一次执行可能需要把 Python 表达的计算图编译成设备程序。之后同样的形状和类型通常复用该程序。若改变形状触发重编译，又会出现额外成本。因此至少应区分编译、稳定执行和包含输入输出的端到端 | E09.03 |
| 09ZH-012 | [L37](../../docs/book/09_gpu_validation.md#L37) · 编译时间和执行时间不能混成一个吞吐 | 段落 | 单卡验证明确调用 lower(...).compile()，把这段时间写入 compilation_seconds；随后执行一次已编译的 loss 与梯度计算，等待结果，再写入 training_se | E09.03 |
| 09ZH-013 | [L39](../../docs/book/09_gpu_validation.md#L39) · 编译时间和执行时间不能混成一个吞吐 | 段落 | 做一个教学计算：假设编译 35 秒，随后 10 步每步 1 秒，每步有 2048 个有效目标。若只看执行阶段，吞吐为 \(20{,}480/10=2048\) token/s；若计入编译，则这次短作业 | E09.03 |
| 09ZH-014 | [L41](../../docs/book/09_gpu_validation.md#L41) · 编译时间和执行时间不能混成一个吞吐 | 段落 | 正式测量应先让目标形状完成预热，再计时多个连续训练步；每段计时都必须等待异步设备执行完成。同时记录有效 token 数、步时间分布、是否含数据等待与 checkpoint、设备数和 peak memo | E09.03 |
| 09ZH-015 | [L45](../../docs/book/09_gpu_validation.md#L45) · 两张 GPU 只能组成相应大小的 mesh | 段落 | Mesh 是把设备排成带名字的轴，例如 data 轴和 expert 轴。create_mesh 将全局设备数组变成 [data, expert]，因此 | E09.04 |
| 09ZH-016 | [L47](../../docs/book/09_gpu_validation.md#L47) · 两张 GPU 只能组成相应大小的 mesh | 公式 | n_{\rm devices}=n_{\rm data}\,n_{\rm expert}. | E09.04 |
| 09ZH-017 | [L51](../../docs/book/09_gpu_validation.md#L51) · 两张 GPU 只能组成相应大小的 mesh | 段落 | 两张 GPU 可以是 \((2,1)\)，也可以是 \((1,2)\)；\((2,2)\) 需要四张，不能在两张卡的报告里同时写两路数据并行和两路专家并行。 | E09.04 |
| 09ZH-018 | [L53](../../docs/book/09_gpu_validation.md#L53) · 两张 GPU 只能组成相应大小的 mesh | 表格 | ／ 两 GPU 拓扑 ／ data ／ expert ／ 全局 hidden \([8,16,16]\) 的概念输入分布 ／ 四专家权重 \([4,16,16]\) 的实际参数分片 ／ | E09.04 |
| 09ZH-019 | [L58](../../docs/book/09_gpu_validation.md#L58) · 两张 GPU 只能组成相应大小的 mesh | 段落 | 第一行尤其值得注意：本仓库默认 shard_parameters=True，专家矩阵第二维还可沿 data 轴切分；它并非“data 轴上所有参数一律复制”的纯数据并行。非专家矩阵也可能沿 data  | E09.04 |
| 09ZH-020 | [L60](../../docs/book/09_gpu_validation.md#L60) · 两张 GPU 只能组成相应大小的 mesh | 段落 | 对 \(n_{\rm data}=2\) 的输入，全局 batch 必须能被 2 整除；对 \(n_{\rm expert}=2\) 的专家参数，专家数必须能被 2 整除。先把这些整数关系核对清楚，通 | E09.04 |
| 09ZH-021 | [L64](../../docs/book/09_gpu_validation.md#L64) · 双卡验证检查了哪一次恢复 | 段落 | 在恰好有两张可见 GPU 的远程作业中运行；每次复核使用新的输出目录，保留旧报告和检查点。默认 mesh 会使用所有可见设备，八张卡会产生 data=4、expert=2，与内置 batch=2 不兼 | E09.05 |
| 09ZH-022 | [L66](../../docs/book/09_gpu_validation.md#L66) · 双卡验证检查了哪一次恢复 | 命令 | bash scripts/validate_mesh.sh \ | E09.05 |
| 09ZH-023 | [L72](../../docs/book/09_gpu_validation.md#L72) · 双卡验证检查了哪一次恢复 | 段落 | 默认 expert_parallel_size=2。正好两张卡时，预期 mesh 为 data=1、expert=2；expert_shards 显示全局专家第一维 4、本地第一维 2。报告还应包含两 | E09.05 |
| 09ZH-024 | [L74](../../docs/book/09_gpu_validation.md#L74) · 双卡验证检查了哪一次恢复 | 段落 | 它先训练两步，保存 mesh-step-2.orbax，恢复模型、优化器和 step，逐个比较本地数组，再分别从原状态和恢复状态各执行下一步，比较参数、优化器状态与 loss。只比较“文件能打开”远弱 | E09.05 |
| 09ZH-025 | [L76](../../docs/book/09_gpu_validation.md#L76) · 双卡验证检查了哪一次恢复 | 段落 | 此处 batch 是脚本中固定的两条短序列，恢复检查不覆盖真实数据流游标、中途进程被杀、文件系统故障或改变设备拓扑。ttt/train.py 的训练恢复还要加载数据采样状态；session 恢复则属于 | E09.05 |
| 09ZH-026 | [L78](../../docs/book/09_gpu_validation.md#L78) · 双卡验证检查了哪一次恢复 | 段落 | 若设备数、专家数或 batch 不可整除，先检查 mesh 与形状；若写入失败，检查目录是否已包含同名检查点、空间与访问权限；若下一步不一致，查优化器状态、step 与精度，保留差异而不要只放宽相等条 | E09.05 |
| 09ZH-027 | [L82](../../docs/book/09_gpu_validation.md#L82) · 参数规模试跑也有明确边界 | 段落 | ttt.validate_mesh 另有 --scale 1b。下面的双卡示例仍保持恰好两张可见 GPU；内置 batch=2 也约束其他拓扑，例如四卡 data=2 可以整除，八卡 data=4 则 | E09.06 |
| 09ZH-028 | [L84](../../docs/book/09_gpu_validation.md#L84) · 参数规模试跑也有明确边界 | 命令 | JAX_PLATFORMS=cuda .venv/bin/python -m ttt.validate_mesh \ | E09.06 |
| 09ZH-029 | [L92](../../docs/book/09_gpu_validation.md#L92) · 参数规模试跑也有明确边界 | 段落 | 预期 JSON 的 scale 为 1b，parameters 给出真实参数数量，两个训练步的指标有限。此模式为避免额外写入大型检查点，**跳过恢复检查**，所以 checkpoint_roundtr | E09.06 |
| 09ZH-030 | [L94](../../docs/book/09_gpu_validation.md#L94) · 参数规模试跑也有明确边界 | 段落 | 该配置的位置上限只有 16，训练序列长度为 4；它探测较大参数和分片计算的可运行性，不是 1B 长上下文预训练复现。若 OOM，先确认实际参数、precision、设备分片与当前显存占用，不能把序列已 | E09.06 |
| 09ZH-031 | [L98](../../docs/book/09_gpu_validation.md#L98) · 正式训练前的准入表 | 段落 | 下面是应随实验保留的人工审查表。它不是当前脚本自动输出的额外字段；缺少证据的格子应写“未验证”，不能从 tiny 的 passed 推断。 | E09.07 |
| 09ZH-032 | [L100](../../docs/book/09_gpu_validation.md#L100) · 正式训练前的准入表 | 表格 | ／ 准入项 ／ 需要的证据 ／ 缺少证据时为什么不能扩大 ／ | E09.07 |
| 09ZH-033 | [L110](../../docs/book/09_gpu_validation.md#L110) · 正式训练前的准入表 | 段落 | 准入阈值应在作业前设定，例如可接受的显存余量、允许的数值误差和稳态测量步数；本书不为所有设备虚构一个通用数值。出现失败时，把原始错误、触发配置和对应报告放在一起：脚本已提供的错误 JSON 是起点，阶 | E09.07 |
| 09ZH-034 | [L112](../../docs/book/09_gpu_validation.md#L112) · 正式训练前的准入表 | 段落 | 到这里，“可以扩大训练”成为一组可检查的条件，而不是一次运行没有报错。下一阶段才应把数据章节构建的真实语料接入训练，用前一章的质量—成本协议回答模型是否值得继续扩大。 | E09.07 |

### docs/book/en/09_gpu_validation.md

| 单元 | 位置与小节 | 类型 | 首句/首式或命令 | 核对依据 |
|---|---|---|---|---|
| 09EN-001 | [L5](../../docs/book/en/09_gpu_validation.md#L5) · Three kinds of success require three kinds of evidence | 段落 | A tiny model producing finite GPU loss establishes that one small graph runs. It does not establish  | E09.01 |
| 09EN-002 | [L7](../../docs/book/en/09_gpu_validation.md#L7) · Three kinds of success require three kinds of evidence | 段落 | We retain the tiny model and two-device expert example, separating numerical correctness, distribute | E09.01 |
| 09EN-003 | [L9](../../docs/book/en/09_gpu_validation.md#L9) · Three kinds of success require three kinds of evidence | 表格 | ／ Level ／ Question ／ Current main entry point ／ | E09.01 |
| 09EN-004 | [L15](../../docs/book/en/09_gpu_validation.md#L15) · Three kinds of success require three kinds of evidence | 段落 | The final row is an experimental admission policy, not an existing one-command certification. We dis | E09.01 |
| 09EN-005 | [L19](../../docs/book/en/09_gpu_validation.md#L19) · Start with single-GPU functionality | 段落 | After environment installation and GPU allocation on a remote node, run from the repository root: | E09.02 |
| 09EN-006 | [L21](../../docs/book/en/09_gpu_validation.md#L21) · Start with single-GPU functionality | 命令 | bash scripts/validate_gpu.sh outputs/ch21/gpu-validation.json | E09.02 |
| 09EN-007 | [L25](../../docs/book/en/09_gpu_validation.md#L25) · Start with single-GPU functionality | 段落 | The script explicitly requires the CUDA backend, writes outputs/ch21/environment.json, and runs ttt. | E09.02 |
| 09EN-008 | [L27](../../docs/book/en/09_gpu_validation.md#L27) · Start with single-GPU functionality | 段落 | Each case mainly checks three properties. First, sequence loss and every gradient leaf are finite. S | E09.02 |
| 09EN-009 | [L29](../../docs/book/en/09_gpu_validation.md#L29) · Start with single-GPU functionality | 段落 | These thresholds belong to the current float32 tiny cases, not every precision and scale. Failure re | E09.02 |
| 09EN-010 | [L31](../../docs/book/en/09_gpu_validation.md#L31) · Start with single-GPU functionality | 段落 | Success does not validate 32K or 128K context: default context cases are only 64 and 512. It does no | E09.02 |
| 09EN-011 | [L35](../../docs/book/en/09_gpu_validation.md#L35) · Compilation and execution are different timing boundaries | 段落 | JAX's first execution can require compiling a Python-described graph into a device program. Later ca | E09.03 |
| 09EN-012 | [L37](../../docs/book/en/09_gpu_validation.md#L37) · Compilation and execution are different timing boundaries | 段落 | Single-GPU validation explicitly calls lower(...).compile() and records compilation_seconds. It then | E09.03 |
| 09EN-013 | [L39](../../docs/book/en/09_gpu_validation.md#L39) · Compilation and execution are different timing boundaries | 段落 | Consider hypothetical compilation of 35 seconds followed by ten one-second steps, each containing 20 | E09.03 |
| 09EN-014 | [L41](../../docs/book/en/09_gpu_validation.md#L41) · Compilation and execution are different timing boundaries | 段落 | A proper measurement warms up the target shapes and times several consecutive training steps, waitin | E09.03 |
| 09EN-015 | [L45](../../docs/book/en/09_gpu_validation.md#L45) · Two GPUs must fit the mesh dimensions | 段落 | A mesh arranges devices along named axes such as data and expert. create_mesh reshapes the global de | E09.04 |
| 09EN-016 | [L47](../../docs/book/en/09_gpu_validation.md#L47) · Two GPUs must fit the mesh dimensions | 公式 | n_{\rm devices}=n_{\rm data}\,n_{\rm expert}. | E09.04 |
| 09EN-017 | [L51](../../docs/book/en/09_gpu_validation.md#L51) · Two GPUs must fit the mesh dimensions | 段落 | Two GPUs can form \((2,1)\) or \((1,2)\). A \((2,2)\) mesh requires four GPUs; two devices cannot si | E09.04 |
| 09EN-018 | [L53](../../docs/book/en/09_gpu_validation.md#L53) · Two GPUs must fit the mesh dimensions | 表格 | ／ Two-GPU topology ／ data ／ expert ／ Conceptual input distribution for global hidden \([8,16,16]\) ／ | E09.04 |
| 09EN-019 | [L58](../../docs/book/en/09_gpu_validation.md#L58) · Two GPUs must fit the mesh dimensions | 段落 | The first row matters: default shard_parameters=True can partition expert matrices' second dimension | E09.04 |
| 09EN-020 | [L60](../../docs/book/en/09_gpu_validation.md#L60) · Two GPUs must fit the mesh dimensions | 段落 | Global batch must be divisible by \(n_{\rm data}\), and expert count must be divisible by \(n_{\rm e | E09.04 |
| 09EN-021 | [L64](../../docs/book/en/09_gpu_validation.md#L64) · What the two-device restore check actually covers | 段落 | Run the following in a remote job with exactly two visible GPUs. Use a new output directory for a re | E09.05 |
| 09EN-022 | [L66](../../docs/book/en/09_gpu_validation.md#L66) · What the two-device restore check actually covers | 命令 | bash scripts/validate_mesh.sh \ | E09.05 |
| 09EN-023 | [L72](../../docs/book/en/09_gpu_validation.md#L72) · What the two-device restore check actually covers | 段落 | The default is expert_parallel_size=2. With exactly two devices, expect data=1 and expert=2, and exp | E09.05 |
| 09EN-024 | [L74](../../docs/book/en/09_gpu_validation.md#L74) · What the two-device restore check actually covers | 段落 | The script trains two steps, saves mesh-step-2.orbax, restores model, optimizer, and step, and compa | E09.05 |
| 09EN-025 | [L76](../../docs/book/en/09_gpu_validation.md#L76) · What the two-device restore check actually covers | 段落 | The batch is two fixed short sequences. This restore check does not cover real data cursors, process | E09.05 |
| 09EN-026 | [L78](../../docs/book/en/09_gpu_validation.md#L78) · What the two-device restore check actually covers | 段落 | For divisibility failures, inspect mesh and shapes. For write failures, inspect existing checkpoint  | E09.05 |
| 09EN-027 | [L82](../../docs/book/en/09_gpu_validation.md#L82) · A parameter-scale trial has limits too | 段落 | ttt.validate_mesh also accepts --scale 1b. For this worked two-device example, keep exactly two visi | E09.06 |
| 09EN-028 | [L84](../../docs/book/en/09_gpu_validation.md#L84) · A parameter-scale trial has limits too | 命令 | JAX_PLATFORMS=cuda .venv/bin/python -m ttt.validate_mesh \ | E09.06 |
| 09EN-029 | [L92](../../docs/book/en/09_gpu_validation.md#L92) · A parameter-scale trial has limits too | 段落 | Expected JSON has scale: "1b", a measured parameters count, and finite metrics for two steps. To avo | E09.06 |
| 09EN-030 | [L94](../../docs/book/en/09_gpu_validation.md#L94) · A parameter-scale trial has limits too | 段落 | Its position limit is only 16, and training sequence length is 4. It probes larger parameter and sha | E09.06 |
| 09EN-031 | [L98](../../docs/book/en/09_gpu_validation.md#L98) · Admission criteria for a substantial training run | 段落 | Retain the following as a manual experiment review. These are not additional fields emitted by the c | E09.07 |
| 09EN-032 | [L100](../../docs/book/en/09_gpu_validation.md#L100) · Admission criteria for a substantial training run | 表格 | ／ Criterion ／ Evidence needed ／ Why scaling without it is premature ／ | E09.07 |
| 09EN-033 | [L110](../../docs/book/en/09_gpu_validation.md#L110) · Admission criteria for a substantial training run | 段落 | Choose thresholds before the run, including memory headroom, acceptable numerical error, and measure | E09.07 |
| 09EN-034 | [L112](../../docs/book/en/09_gpu_validation.md#L112) · Admission criteria for a substantial training run | 段落 | “Ready to scale” is now a set of inspectable conditions rather than an absence of exceptions. Only t | E09.07 |

## 静态检查记录

- `git diff --check --` 本轮10页及两份审稿文件：交付前重新执行；见工具输出，不能据此判断Python运行或网页渲染。
- 本索引记录 366 个正文单元；数量只说明覆盖范围，不能当作质量分数。
- 未提交：按根代理要求先交叉审稿，再统一处理提交。
