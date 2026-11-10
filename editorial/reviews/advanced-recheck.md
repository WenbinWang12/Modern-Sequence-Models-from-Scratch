# 第17–21章独立交叉复审

日期：2026-11-11。审稿代理：`paragraph_gate_rewritten_main`。正文作者与本审稿者不同；本审稿者没有修改05–09正文。

范围：`docs/book/05_inner_outer_loop.md` 至 `09_gpu_validation.md` 及其英文版，共10页。逐节通读正文、数学例子、代码命令、表格和图注，并与下面列出的实现静态对照。审稿时内容是工作区修订稿，不是原 `567fef3`；末表记录复审时文件摘要。

**结论：两项新发现的事实问题已由原作者修正并经本审稿者复核；本轮重点实现与数学复核未发现其他同级硬错误。出版审稿仍不通过。** 本轮不包含远端运行、插图本体审美、网页渲染、第三方链接全文验证，也不把“未发现硬错误”当作完整教学链达标。没有本地测试，没有GPU测量或恢复实验结果。

优先级沿用章节审稿约定：P0错误事实或不可解释实验，P1必要教学链缺失，P2措辞与记号改进。

## 发现—修复—独立确认

| ID | 级别 | 原位置与首句 | 原问题及证据 | 作者修复 | 独立结论 |
|---|---|---|---|---|---|
| A01 | P0 | 07双卡命令前“至少两张可见GPU”；09双卡与1b命令前，对应英文 `at least two visible GPUs` | 默认 `create_mesh` 使用全部可见卡，expert=2，`validate_mesh.py:50,68` 的batch固定2。8卡形成data4/expert2，batch无法分片。“至少2”不是充分前提。 | 双卡演算明确要求恰好两张可见设备，解释8卡反例，保留调度器映射；1b示例继承该演算前提。 | **正文事实关闭**。静态核对 `distributed.py:create_mesh/tree_shardings` 与固定batch；没有运行两卡验证。通用代码也可支持其他可整除拓扑，教程双卡限定应与代码必要条件区别。 |
| A02 | P0 | 07末段“分别比较更新专家MLP”；英文 `adapting expert MLPs` | `modeling_ttt_e2e.py:77–80` 的快权重来自 `layer.fast_mlp` 或 `fast_router`，不是 `layer.moe.experts`。`_block:155–168` 把额外MLP′放在静态MoE前。 | 明确mlp路线更新MoE之前额外稠密MLP′，router更新适配器，delta_router更新递归记忆；experts由外循环训练。 | **正文事实关闭**。中英文当前段落均匹配实际参数树，没有声称已运行消融。 |

## 核心数学与实现复核

| 范围/段落首句 | 读者问题 | 复核结果与具体证据 | 未核验/保留意见 |
|---|---|---|---|
| 05“用theta_k表示…” | 初值和序列状态是否不同？ | 保留。`initial_fast_weights(params)`提取可学习初值；`sequence_loss`每条序列重新取初值，不把上条样本终值直接复用。两版一致。 | 未运行跨样本探针。 |
| 05“先用一个刻意简化的二分类…” | 交叉熵如何产生快权重更新？ | 保留。logits=(0,phi)、目标B给导数sigmoid(phi)-1；phi0=0、eta1更新到0.5，p约0.6225、loss约0.4741；零梯度抵消例也正确。 | 这是标量教学模型，不是MLP实验结果；正文已说明。 |
| 05“Jacobian可以看成一张导数表…” | 梯度形状如何从词表回到参数？ | 保留。J为[V,n_phi]，J转置乘[V]误差给[n_phi]，无需显式构造J。 | 读者仍需要基础求导材料，见B01。 |
| 05“若所有快权重叶子的梯度…” | 裁剪是否与代码一致？ | 保留。`sgd_update`是整树范数sqrt(sum g²+1e-12)，scale=min(1,kappa/norm)，不是逐元素裁剪。标量-2、eta0.5、kappa0.5得到约1.25。 | 裁剪阈值处不可微/实现选定导数规则还可补一句；不影响所给非阈值算例。 |
| 05“四个输入位置需要五个token” | 标签何时足够更新？ | 保留。输入1..4的最后目标是第5个token；尾块由 `inner_loop.py:152–157` 整块保留旧fast，而非只忽略padding的梯度。 | “数学位置从1开始”与chunk从0计数应在统一notation中明确。 |
| 05“输出cache来自更新前的预测” | 历史缓存是否重编码？ | 保留。`inner_step`的output在objective(current_fast)内产生；sgd_update后返回同一output；runtime保留历史cache。 | 流式一致性仍待运行。 |
| 05“若样本b有n_b…” | batch是否按有效token加权？ | 保留。100×2+2×10=220，除102约2.157；匹配 `training.py` counts加权。 | 二值mask时是数目；允许分数权重时应称权重和。 |
| 05“设快权重初值为phi0(theta)” | full meta-gradient与一阶近似有何区别？ | 保留。`J1=J0-eta(g_theta+g_phi J0)`，eta不依赖theta、未裁剪条件清楚。忽略g导数保留J0；stop_gradient整个phi1消除J0。标量future梯度0.4669/0.6225/0三个结果正确。 | 这是推导与源码路径核对，不是有限差分验证；B01仍open。 |
| 05“实际更新若触发裁剪…” | checkpoint重算是否改变梯度？ | 保留。用a(g)g替代g且需对a求导正确；`jax.checkpoint`不等于stop_gradient；当前配置没有FOMAML开关。 | 未验证每种top-k、裁剪边界下的高阶数值行为。 |
| 05“远端最小验证…” | 验证脚本实际证明什么？ | 保留。字段与`validate.py:44–54`一致；因果性是assert而非独立JSON bool；case只验证finite/streaming/causality，正文没有把它当质量证据。 | 不能用finite_gradients证明二阶导数数值正确。 |
| 06“Attention可以理解为一次带权检索” | Q/K/V与softmax如何产生输出？ | 保留。给定二维矩阵QK转置除sqrt2、因果mask、权重0.6698/0.3302及输出6.698/3.302均一致。对角允许因输入t预测t+1，解释正确。 | 真实模型还有RoPE/GQA；当前例明确是投影后的简化单头。 |
| 06“除以sqrt(d_h)…” | 缩放从哪里来？ | 保留。在分量独立零均值单位方差假设下内积方差d_h；正文明确不是训练后分布保证。 | 可再说明query/key之间也需相应独立/不相关假设，避免把“各分量独立”只理解为单个向量内部。 |
| 06“本仓库的KVCache面向单session” | 概念shape和实际shape一致吗？ | 保留。实际capacity×kv_heads×head_dim，不是B轴数组；positions另存。2MiB/层、12层24MiB算例正确，排除项写清。 | 没有GPU显存测量，图本体未看。 |
| 06“还要区分三个长度” | sliding window是否等于实际显存？ | 保留。observed、capacity、visible三者区分正确；`grow_layer_cache`还按sliding_window限制对应cache，混合full层与临时开销仍须实测。 | 没有承诺allocator精确大小，边界合适。 |
| 06“收到x5时…”及状态表 | 满块是否立刻更新？ | 保留。runtime:97先在pending==C时用当前到达token补标签，`_finish_chunk:65`更新后 `_start_chunk`，再前向新token。x4后pending4/targets3/update0；x5后pending1/targets空/update1；x6后pending2/targets[x6]/update1均匹配。 | 需要可运行状态打印来供读者验证，B02。 |
| 06“历史suffix cache可能由较早…” | 什么才是正确参照路径？ | 保留。任意最新phi重跑全部历史不是该算法参照；应与同chunk/cache协议的sequence_loss比较。 | 当前runtime没有任意loss_mask输入；该一致性说明应保持在验证用例的监督协议范围内。 |
| 06“session_store把数组和元数据一起保存” | 恢复要包含哪些状态？ | 保留。格式2、model_revision/config/inner、shape/dtype/position/pending校验、第一本地设备恢复均可在session_store静态找到。 | 未执行session恢复测试，正文准确区分单卡validate不覆盖它。 |
| 07“当前实现用r+b选择top-k” | 选择分数与混合概率是否混用？ | 保留。`moe.py:79–82`原始logits softmax，top_k(logits+load_bias)，选中概率再归一化。四token排序[0,2,1,3]与top2输出(4/3,1)正确。 | 极小分母保护的数值退化不在演算例内。 |
| 07“top-1例子只用于解释重排” | router梯度来自哪里？ | 保留所给正常概率范围结论。归一化top1权重为1，选ID不提供通常梯度；aux可提供梯度。没有误推广所有top1模型。 | top2可导权重与离散边界的推导仍缺，B03。 |
| 07“容量与负载” | 是否真的丢token？ | 保留。容量公式是设计背景；当前group_sizes来自真实分配，无capacity配置/丢弃。aux=1及集中例1.8正确，没有误称1为全局下界。 | 代码的load_bias平衡更新没有充分解释，B03。 |
| 07“自定义JVP参考路径会循环专家…” | 稀疏激活是否自动带来稀疏训练FLOPs？ | 保留。`grouped_gemm.py:22–28`对每expert计算lhs@rhs[index]再mask；`_grouped_jvp:37–43`的primal和两项tangent均用该reference。正文准确限定普通primal才ragged_dot，不把高阶路径视作仅算活跃专家。 | 未检查XLA最终消去/融合效果；正文正确要求编译和硬件测量。 |
| 07“两张GPU上…” | 权重与输入到底如何分片？ | 保留。4专家×16×16，expert轴2时本地2×16×16；data轴1输入2×5及mask2×4复制。A01修后命令前提充分。 | 未执行实际shard检查，未声称all-to-all手写实现。 |
| 07“在同一框架区分几种研究问题” | MLP路线更新专家还是额外分支？ | A02修正后保留：额外dense MLP′适应，静态experts外循环。 | 三种路线质量还没有实验结论。 |
| 08“以下数字是教学假设” | 同token与同时间是否混为一谈？ | 保留。100万/12000=83.3s，PPL比exp(-.12)=.887，成本1.67倍；50秒F只有60万token，不能沿用终点NLL。演算与实测标注清楚。 | 当前训练日志名义token计数与有效目标另记的要求已明确。 |
| 08“三个配对种子…” | 波动量是什么？ | 保留。mean=-.10、sample sd=.02、SE=.0115正确，独立性与不外推说明适当。 | 未附真实种子结果，不应视为统计证据。 |
| 08“生成实验配置” | 命令及产物真实存在吗？ | 保留静态接口。experiments CLI的sizes/tokens/include-moe、manifest.experiments.scheduled_tokens与pending匹配源码，明确生成配置不等于训练。 | 未执行配置生成；Llama访问与正式数据前提未验证。 |
| 08“推理profile与训练成本估计” | 前章checkpoint可否用于profile？ | 保留静态接口。tiny mesh保存mesh-step-2.orbax，位置上限16，8+4=12合法；checkpoint恢复默认落首个本地设备，profile是单session；字段、预热、null、训练图不含optimizer的限制均正确。 | 未实际加载Orbax或计时；需要作者运行后存stdout以关闭实践验证。 |
| 08“Scaling是可以被留出规模反驳的假设” | 模型方程是否匹配代码，留出是否真实？ | 保留。N/1e6、D/1e9、E+A n^-alpha+B d^-beta与scaling.py一致；至少8训练行、3种N、3种D，held_out按行排除，不宣称自动交叉覆盖。 | 缺数据fixture/拟合命令及对bootstrap区间的区分，B04。 |
| 09“三种成功” | tiny是否等于生产准入？ | 保留。单卡数值、多卡restore、目标参数运行三个层次清楚，手工准入表未伪装成自动字段。 | 没有实际准入完成证据。 |
| 09“单GPU功能验证” | case和报告有哪些？ | 保留。dense命名实际含fast/TTT的限定正确；上下文64/512、finite、streaming<=1e-3、causality atol/rtol、异常字段匹配。 | GPU预检可能在validate之前失败，不保证所有失败都有validation.json，见B05。 |
| 09“编译时间和执行时间” | 一个timing是稳态吞吐吗？ | 保留。lower.compile计时与单次loss+grad执行清楚区分，455.1/s算例正确，未冒充多步steady统计。 | 未测GPU数值。 |
| 09“两张GPU只能组成…” | mesh是否对应真实参数分片？ | 保留。device=data×expert；default shard_parameters=True可沿data拆expert矩阵第二维，非“全复制纯DP”；表的本地形状符合tree_shardings。 | 中间hidden只标概念分布，未过度承诺编译后布局。 |
| 09“双卡验证检查哪次恢复” | restore到底比较了什么？ | 保留。先两步、保存step2、参数优化器state本地数组相等、各走下一步并比较loss；固定batch非实际loadercursor，局限写出。A01修后前提充分。 | 未实跑恢复。 |
| 09“参数规模试跑” | 1b是否等于长上下文复现？ | 保留。1b模式跳过保存/restore、checkpoint_roundtrip=false、序列4/max16，与源码一致。 | 4可见卡也可能满足代码整除条件；“恰好2”是本章演算限定，不是代码支持范围，已提醒作者措辞。 |

## 仍开放的出版问题

| ID | 等级 | 具体段落 | 建议与验收方式 |
|---|---|---|---|
| B01 | P1 | 05外循环二阶推导及远端验证之间 | 推导本身正确，但finite_gradients无法区分0.4669与错误的0.6225。补独立标量计算代码或中央差分对照，读者应同时看到三条梯度路径及输出，明确它只验证小例子的导数。主模型还需单独检查。 |
| B02 | P1 | 06 prefilling/decode状态表与session恢复段 | 当前可运行命令仍只是通用validate，不输出表内pending_count/targets/position。给一个使用真实TTTSession的六token打印与中间chunk保存/续接实例；把“读表理解”变成可以重现的状态证据。现引用测试文件并不说明读者如何执行与读结果。 |
| B03 | P1 | 07 top-1梯度提醒、aux与grouped JVP段 | 加一条top2选中集合固定时的权重导数及路由边界局限；解释load_bias的显式平衡更新和learned router梯度不同。给grouped乘法的JVP公式 d(LW)=dL·W+L·dW，并对应reference两次调用，避免JVP仍是未定义缩写。 |
| B04 | P1 | 08 scaling最后三段 | 给一个明确标为合成数据的JSONL fixture与ttt.scaling CLI、输出字段解释；不要把拟合摘要当作已做scaling实验。解释bootstrap预测的拟合均值区间不自动包含新训练运行噪声，不应无条件称未来观测的95%预测区间。 |
| B05 | P2 | 05/06/09失败诊断 | shell先运行check_environment；CUDA不可用时可能尚未进入ttt.validate。补一句“预检失败先看environment.json/终端，此时validation.json可能不存在”，使读者不去寻找必然不存在的error_type字段。 |
| B06 | P2 | 05–09数学记号 | 06矩阵Q/K/V/H未使用统一粗体，05原始token叙事用x而前章用s，需全书记号最后统一。chunk从0计数与数学位置从1计数是两种对象，应不再被误读为索引例外。 |
| B07 | P1 | 全部图和章节体验 | 本轮只看图注/引用，未打开SVG本体；不能给美术、导出、色盲可读性、数学渲染或字号通过意见。所有配图需要独立视觉审稿。当前数学链条比原稿完整，但出版验收仍应由零基础读者走完命令与推导。 |

## 版本与证据边界

复审实质关闭的是A01/A02的**正文事实错误**；运行状态仍为未验证。作者随后的小措辞修订可能改变以下摘要，重新投稿时应重新核对。摘要由文件字节静态计算，没有执行模型代码。

后续措辞复核：09两版1b段落已明确“双卡是本章演算设定；四卡data=2也可整除batch=2，八卡data=4不行”。该补充已重新阅读并确认，关闭了上表关于把双卡误读为代码唯一支持范围的意见。下列09摘要保留补充前的审稿快照，未将旧快照伪装成最终文件。

| 文件 | SHA256（复审时） |
|---|---|
| 05_inner_outer_loop.md | C087DF279D72239FDAE26F4F10684D96AA80F350D76FCCAF1411292DEE5649E9 |
| 06_attention_cache.md | 3C596B2B15305EB7AB363CFE0CB1A90CB5D9FDB505FEC67121CB504AF3A0999A |
| 07_moe.md | 4172372F91BA9BFDAF8D30742B537C6BF97377F732D30A3DD4F6D06A1C488188 |
| 08_experiments.md | 437B915B157F562085E65B3C38939D847D5F568464990B5368F9BF575A971D8E |
| 09_gpu_validation.md | 511CDF98340BCFA89D2FCF563561CF6010DD6356344A33421C396A01228CD370 |
| en/05_inner_outer_loop.md | 975AFD9D1D351ED3719448823D2AF84CCF1C022278B6C83341BA3E35164D4E7B |
| en/06_attention_cache.md | E1D1D59CC76484A1093E56632F42747DCAE43E6BA230CC51CEE669803EA4B225 |
| en/07_moe.md | CA27C34252C51B4947449697299BACFE2BE76028A80EE6F7773C8CD4F2E97CEF |
| en/08_experiments.md | 09D67707E43DE1C5FF71B99A25E0B5810669CA9E7E16D02F6EF038E16A85ADE6 |
| en/09_gpu_validation.md | 08446D62346B3AE42FEE30F9757D3689D5295F64F1A2F4C6C6933F04E1EB9518 |

未覆盖：05–09以外的修改、全仓数值正确性、论文官方结果复现、实际机器性能、远端CPU/GPU可用性、站点构建与部署。不能把本报告用于批准这些事项。
