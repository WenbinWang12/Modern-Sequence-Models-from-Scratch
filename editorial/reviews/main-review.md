# 早期章节逐段独立审稿记录

审稿基线：`567fef3`。审稿人：独立子代理 `paragraph_gate_rewritten_main`。日期：2026-11-11。

**结论：不通过出版审稿。** 这是对指定快照的静态审读，不是构建通过、运行通过或出版通过证明。审稿过程中根代理已经开始改正文，因此下面的行号、首句和 Paragraph ID 均指向 `567fef3`，不能把工作区的新内容倒填成该快照已通过。

范围：`notation`、`00_why_pretrain`、Part I 的五页，以及 `10_data_contract` 至 `20_baseline_model`，中英共 36 页。范围内正文已逐块阅读；公式、命令、表格和图注纳入相邻论证审查。插图本体、网页数学渲染、链接目标全文、第三方工具行为实测和服务器命令执行未验收。没有运行本地测试。根代理报告目标 CPU 节点 SSH 超时，不能把静态检查写成运行成功。

优先级沿用审稿时的 `chapter_template.md`：P0 为错误事实、错误数学或不可解释/不可执行的实验；P1 为缺失必要教学链；P2 为文字、图注、组织问题。此前即时消息使用通常代码审查里的 P1 表示硬错误，本记录将其统一映射到模板的 P0。下列“保留”只表示该块未发现独立硬错误，仍受所在节的前置知识、双语、实践及运行未验收项限制。

## 需要先修复的事实和实现问题

| 编号 | 级别 | 文件、小节与段首 | 问题、证据与具体修复 |
|---|---|---|---|
| F01 | P0 | 两版多页，命令 `--input examples/sample.txt ... --sharded` | `ttt/dataset.py:116–120` 对每个非空行无条件 `json.loads(line)`，分片入口只接受 JSONL，纯文本样例会解析失败。保留普通 NPY 的 TXT 路径；为分片提供真实 JSONL fixture 并明确 `text` schema。原命令的“成功时应出现 manifest”不是运行证据。 |
| F02 | P0 | notation，`如果只让前两个目标参与训练`/对应英文；`M=(1,1,0)` | 例子只有 `s=(17,42,31,2)`，最后目标是 EOS，不是跨文档目标。必须添加下一文档 token/BOS 后再举跨文档 mask，或把解释明确改为屏蔽 EOS。 |
| F03 | P0 | notation，`例如两个文件各有 100 个 token` | 等长两文件不能说明“长文件贡献更多窗口”。改成长度不同的例子，写清采文档后取一窗口还是穷尽窗口；英文解释与中文此例不齐。 |
| F04 | P0 | notation，末段 `d_model、num_heads`；03 英文 `Model size` 表 | 仓库字段是 `hidden_size`、`num_attention_heads`、`num_hidden_layers`，没有所称 `d_model`/`num_heads`/`num_layers`。概念简称必须明确不是 JSON 字段；代码映射使用真实名字。 |
| F05 | P0 | 01 contract 的运行例子；01 environment/02/20 的 `--config configs/tiny.json` 和 `...-C`/`baseline-...` | `configs/tiny.json` 的 `inner.enabled=true`、默认 `use_fast_mlp=true`。所谓 C 与 dense baseline 实际启动 F。C 使用同结构但 inner disabled 的独立配置；dense 还应关闭 fast 且 suffix=0。运行前打印 resolved 开关。 |
| F06 | P0 | 01 contract 指标表 `valid_tokens`；01 environment `compile_seconds`；02 `Code and artifacts` | `ttt/training.py` 的 metrics 无 `valid_tokens`；`ttt/train.py` 无 `compile_seconds`、`error.json`、`checkpoint/`；状态值为 `running/failed/complete/stopped`，不是 `completed/resumable`。`experiment.json` 是配置，不含承诺的 commit/data digest。`ttt/provenance.py` 无独立 backend/dtype 字段。将真实输出与需要另行记录的目标清单分开。 |
| F07 | P0 | 01 environment，bootstrap 后 `python3 scripts/check_environment.py` | bootstrap 创建 `.venv`，子 shell 不激活调用 shell。后续 `python3` 可能是未安装 JAX 的系统解释器。用 `.venv/bin/python`；CPU 分支不能只去掉 require-gpu 后仍使用 `run_gpu_train.sh`。 |
| F08 | P0 | 02，`用它从 dense baseline 初始化 TTT 是合理的`/`It is useful when initializing TTT from the dense baseline` | `ttt/train.py:77–85` 要求参数树、每叶形状和 dtype 一致；无 MLP′ 的 dense 不能直接初始化新增 fast 参数的模型。明确“同架构 C→F 可复用；真正 dense→新增 MLP′ 需要显式权重迁移工具，目前未实现”。 |
| F09 | P0 | 02 表格，`下一批索引 2→3` | `BatchStream.next_batch_with_masks` 抽取随机 token 起点，并非顺序样本索引。将 2→3 标成优化 step；下一批证据应是抽样 tokens/masks 或 RNG state 的比较。若用 sharded mask，固定 16 个有效 token 也不成立。 |
| F10 | P1 | 02，`中断 step 2 后重新执行` | 四步 tiny 命令常在人工中断前结束，完成后 auto-resume 只打开 step4，不构成恢复实验。提供相同配置、不同输出目录、明确 step2 checkpoint 的 `--resume` 路径及连续路径比较；给出需比较的字段和容差，不假造恢复数值。 |
| F11 | P0 | 03，`use_fast_mlp=false 却指定非零 suffix` | `configuration_ttt_e2e.py:161–174` 对 `adaptation_target='router'` 允许无 fast MLP 的非零 suffix。限制到 `adaptation_target='mlp'`，同时解释 C 是禁内更新，不能用 fast 模块不存在来定义。显式 head_dim 时 hidden_size 非整除也不能笼统判错。 |
| F12 | P0 | 04，`设 prefix 代码索引为 0、1`/`Let code indices 0 and 1 be the prefix` | 前文 prefix/suffix 指层栈，此处突然改指时间位置，混淆两个轴。改为“较早/较晚 token 位置”。改变 input index3 时，索引0、1、2均应不变；仅比较0:2漏掉紧前位置。 |
| F13 | P0 | 04，`report should contain ... restore results` | `ttt.validate.validation_case` 返回 loss、finite_gradients、logits_max_error、compilation_seconds、training_seconds、state_bytes、inner_updates，外层 environment/passed/cases。没有 restore、state shapes、配置快照或独立因果误差字段。因果校验是 assert。精确列真实字段；恢复指向02单独实验。 |
| F14 | P0 | 04，`如果最后一个 chunk 只有两个真实位置` | `inner_loop.py:152–157` 对不完整尾块保留整个旧 fast state；真实尾部目标仍评分，但整块不提交内更新。不能描述成仅 padding 位置无更新而真实尾部照常更新。 |
| F15 | P0 | 04，`Whether higher-order gradients are retained ... rematerialization` | `jax.checkpoint` 重算以省内存，并不自动截断高阶梯度。当前 `sequence_loss` 无 first-order 开关/stop_gradient；外层微分可穿过 value_and_grad 的内更新。区分截断近似与重计算。 |
| F16 | P0 | 10，`模型真正优化的是这个损失在窗口分布上的期望`及后续经验目标 | `E[sum NLL / count]` 与所有有效 token 汇总的 `sum NLL / sum count` 在窗口有效数不同的时候不是同一目标。代码 `training.py` 用 counts 加权。给 1-token loss=10 与9-token loss=0 的例子（窗口均值5，token均值1），定义 ratio-of-expectations 或显式 size-biased window distribution；分母零也须有定义。 |
| F17 | P0 | 10，`如果清洗、切分和来源混合构成变换 C` | 删除记录与重新采样权重不能不加条件当成普通确定性 pushforward。应先写保留指标和归一化后分布，再写 deterministic tokenizer 的推前；随机采样用转移核。当前数学比所需更抽象却省掉适用条件。 |
| F18 | P0 | 10 中文 `P_win=sum_i P_doc ...` | 此式只覆盖文档内取窗，不能表达跨两篇文档的流式窗。英文已补“document-local sampling”及拼接流说明，中文没有；中英文必须同步条件。 |
| F19 | P0 | 10，`均匀抽取文档则会提高书籍文档的采样概率` | 表中 books 占 token约9.09%，占文档约1.04%；均匀抽一文档再取固定窗会降低书籍来源概率。若每文档取所有token，比较对象又变了。写明采样单位和数值，不用没有分母的“提高”。 |
| F20 | P0 | 14 首段，`几乎可以肯定两边都会出现`/`almost certainly` | 未给切分概率无法断言。两个集合、验证概率p、10副本时，两边都有的概率为 `1-p^10-(1-p)^10`；p=0.01仅约9.56%。改成“存在泄漏风险”，或给计算后选择合理例子。 |
| F21 | P0 | 13 最小检查，`可加总字段 ... near_reviewed=50` | `kept890+exact80+near30=1000`；`near_reviewed50` 是检查活动数量，与删除数重叠，不能一起加总。两版后加段均如此，英文前文其实已正确说明50中30合并20保留。写清守恒式与独立活动计数。 |
| F22 | P0 | 15 API 示例，`tokenizer.pre_tokenizer = ...ByteLevel` | 直接 Python 示例缺 ByteLevel decoder；保存后 decode 不保证恢复文本。示例及实际脚本 trainer 都未明确全256 alphabet，因此“byte-level”不等于有完整 byte fallback。补 decoder，说明初始 alphabet/UNK覆盖并加入未见字符例；这是静态缺项，未执行库行为实验。 |
| F23 | P0 | 15 接入流程，`encode/decode round trip` | 示例及脚本使用 NFKC，它会不可逆合并兼容字形。期望不是无条件 decode(encode(u))==u；应明确等于规范化结果并展示原文与规范化文本差异。 |
| F24 | P0 | 15 两条 `--output data/pretrain` | `prepare_sharded_data` 拒绝覆盖现有 manifest，依次复制两行第二行必失败。两个 tokenizer 使用不同输出目录，并明确它们是替代实验。远端 tokenizer 模型还需访问条件及固定revision，不应当作无前提首跑。 |
| F25 | P1 | 15 `uv sync --extra tokenizer`、`uv run ...` | 安装路径没有沿用bootstrap固定uv/Python/全部extras，uv sync可能移除此前cuda/checkpoint extras。使用已准备解释器及明确组合的 frozen环境，说明额外依赖；不要把环境重建藏在tokenizer章。未执行uv验证。 |
| F26 | P0 | 16，`独立文档 ... attention 或 loss 不跨越边界` | loss屏蔽边界目标不等于后文不能attention到前文。真正文档隔离需要attention/state边界并明确位置重置；把“屏蔽一个目标”和“隔离上下文”作为两种不同性质展示。后文虽有提醒，定义仍错误。 |
| F27 | P0 | 17 中文最小诊断，`检查 manifest 的 start、end、count` | manifest shard只记录count及路径/hash；start/end由loader累计计算，不是JSON字段。用count累计ends并查 `searchsorted(..., side='right')`，另加恰好落到端点p=4的例子。现例p=2掩盖side参数的重要性。 |
| F28 | P0 | 18 首段，`900 GB网页 ... 大约90%的token` | 原始字节量不能直接变token比例，各源tokens/byte不同。改成明确的token数或加共同编码密度假设；应承接15的tokens/byte而不是反向误导。 |
| F29 | P0 | 18 中文 `温度小于1会提高小来源`，后段逆幂反例；两版recipe `weight` | 第一段未限定幂约定，与后文自己矛盾。先定义基础权重与最终概率，使用一个明确公式；`weight`不能既为最终目标概率又继续经过温度变换。英文公式较清晰但JSON含义仍不齐。 |
| F30 | P0 | 19 `一个真实的验收报告` | 没有该报告的运行产物，且正文说没有独立preflight命令。必须标“假设/演示报告”，不能称真实tiny运行。两版“file and window checks pass”示例只含file行，进一步表明不是证据。 |
| F31 | P0 | 17英文/19两版 `runs/example_shards/manifest.json`；18 `outputs/recipe.json`/`runs/recipe.json` | 未由本章真实样例流程产生的路径；中英有不同路径。明确上游准备依赖，统一fixture与输出路径，保存recipe的命令也应给出。不能把单纯JSON可解析称实验读取配方。 |
| F32 | P0 | 19 中文输出安全 `checkpoint的数据指纹与当前输入一致` | resume必须匹配数据，initialize-from允许新数据、新优化过程；不能把两者统一限制。英文输出安全已限resume，中文须同步。 |
| F33 | P0 | 20 中文softmax公式 `sum_{v=1}^{V}` | 全书vocab ID是0..V-1，此处漏0含V。改为0..V-1并核对其他分母。英文sum_v不犯此错。 |
| F34 | P0 | 20 英文，`Parameter count is the number of leaves in the model tree` | 参数量是叶张量元素数之和，非叶个数。`ttt/profile.py:parameter_counts` 明确 `sum(value.size for value in ...leaves)`，中文正确。用两个矩阵说明叶2但参数更多。 |
| F35 | P0 | 01–02/20 吞吐、有效token表述 | `ttt/train.py:139–140` 是名义B×L步数和吞吐，包含masked位置；不能称有效监督token或据此比较数据预算。表述名义输入位置数，另记录mask.sum；真实数据cost与effective supervision都需要分母。 |
| F36 | P0 | 00/notation/20，`正确类别梯度为负，所以...分数会上升` | 仅把logits本身作为自由变量做SGD时能保证逐分数方向；共享网络参数、batch、多步AdamW不保证每样本logit单调。20新手算段已正确限定，但前段仍无条件断言。把限定移到首次主张，说明梯度是局部误差信号。 |

## 尚未达到初学者教材标准的具体段落

| 编号 | 级别 | 位置 | 读者问题、缺口与可实施补法 |
|---|---|---|---|
| T01 | P1 | notation/00 首次softmax、log、导数、onehot | 已说“无ML背景”，却直接使用指数、自然对数、求导、链式法则。先用概率比和乘法转加法说明log，再由单变量斜率到梯度；链式法则至少算一层线性投影。符号页不应替代一章基础推导。 |
| T02 | P1 | 00 `p(x1:L)=prod`到NLL | 给出概率链式法则及负log展开的中间行，定义条件概率、空前缀/BOS；当前一句“根据链式法则”不是零基础推导。重复CE可移到后章，通过明确链接复用。 |
| T03 | P1 | 01 contract/environment 首次 optimizer、RNG、dtype、mesh | 先出现术语表后才解释CPU/JAX；用户需要知道参数、优化器状态、seed为何不同。将晚加环境解释移到首次出现附近，例子贯穿同一tiny配置，避免“合同”抽象术语堆叠。 |
| T04 | P1 | 02 `随机数状态`与恢复比较 | 补一个seed→前三次抽样→保存内部状态→恢复第四次的可读例子；本页目前要求读者比较state却没有读取checkpoint或产生输入证据的方法。不能以人工中断四步作业替代实践。 |
| T05 | P1 | 03 shape walk | “矩阵把一组数映射成另一组”仍跳过向量、矩阵、三维张量及最后轴乘法。增加2token×3维查表与3×2投影手算；明确教学B=2,L=4不同于tiny训练B=1,L=16。GQA/SwiGLU首次只命名未解释。 |
| T06 | P1 | 04 `_block`、prefix/suffix、cache | 需要一张层轴×token轴图区分静态前层和时间chunk；当前箭头图不解释层/时间两种prefix。逐行说明输入shape、缓存owner、phi0来源及真实尾块不提交更新。 |
| T07 | P1 | 10 pushforward/expectation整节 | 读者还未学概率测度就用 `C_#`，且条件遗漏。以2文档、2来源、有限离散表推导采样概率，后给高级紧凑记号；用两种mask-count反例建立窗口/位置分布差别。 |
| T08 | P1 | 11 `来源记录`、最小运行 | 只运行tokenizer不能演示本章“获取并记录来源”。在原创fixture上实际计算bytes/hash、生成来源记录并做一次修改前后摘要比较；数据下载可以不做，来源身份练习仍应有可检查产物。 |
| T09 | P1 | 12 NFC/NFKC、`inspect_document` | NFC/NFKC只描述，缺真实Unicode例（组合音标、全角字符）和代码输出；`.strip()`会移除首尾空白，技术文本影响需说明。给3–5条记录跑函数并核对主拒绝原因守恒；现在章末重复编码不检查清洗。 |
| T10 | P1 | 13 MinHash/LSH段 | 只有工具名，读者无法知道怎样产生签名、为什么估计Jaccard、候选不等于真重复。补最小集合排列例和命中概率解释，手算非传递三文档例并运行exact key代表表。 |
| T11 | P1 | 14 hash split段 | SHA256为何稳定与“近似均匀”不是一回事；补固定group表、分桶算例和组数vs token数表。实际命令只是将未知上游JSONL编码，没有运行choose_split或生成可核对报告。 |
| T12 | P1 | 15 BPE/WordPiece/Unigram | BPE只列步骤和`lo→low`，无频次表、合并后计数、编码时merge rank；Unigram无分割概率或剪枝目标。正文不能把这些命名段作为严谨算法讲解已完成。先完整手算BPE，两种其余方法给概率例和限定范围。 |
| T13 | P1 | 16 `固定窗口检查` | 要求读者检查X/Y/M，却不给打印窗口、目标mask与attention矩阵的实际代码。补真实BOS/EOS的跨文档例，展示mask0目标仍可作为后续上下文。必须与ByteTokenizer实际ID一致或明确教学ID。 |
| T14 | P1 | 17 `searchsorted`、mmap | 跨分片数值例有进步；仍需端点例和mmap分块读取为何节省堆内存、hash全扫描仍需IO。章末应实际调用rows而非仅打印JSON，否则没有验证本章核心。 |
| T15 | P1 | 18 数据混合、rank、恢复 | 只有设计宣言和JSON，不包含确定性sample-plan实现或来源抽样手算；rank、global index、with replacement未解释。先实现独立教学sample plan，明确不是训练已接入，展示恢复前后索引与消费比例。 |
| T16 | P1 | 19 preflight | 八项列表有价值，但缺一个完整可运行的成功/失败对照；TokenDataset构造无法证明其余七类。给真实loader错vocab示例和异常，再把未实现的项逐个pending，不能把假报告放在“真实”标题下。 |
| T17 | P1 | 20 learning、held-out loss、AdamW | 本章有logits手算，但无真实dense配置、独立验证数据、eval命令、最小结果解析；AdamW直接作为黑盒。至少给控制流和SGD→动量→二阶矩直觉及指向后续算法推导；先完成一个训练/评估/恢复证据链再称“证明能学习”。 |
| T18 | P1 | 两版全部，章节实践的总结构 | 新增末尾Minimal checkpoint经常重复前面的命令，并在原“下一章”后继续讲新内容；11–14多数只调用同一个ttt.data。按章的因果问题设计不同证据，不用每章相同模板段落代替实作。 |
| T19 | P1 | 双语对应：03/10/17/18/19/20 | 英文03缺中文版新增完整shape walkthrough；中文10缺英文文档局部采样限定；英文17有累计offset公式及sidecar说明，中文章末却要求不存在的start/end；18recipe路径不同；19中文把示例称真实；20英文参数量定义错误。逐项对齐同一事实与命令，不要求逐字翻译。 |
| T20 | P1 | 所有图与无图章节 | 本次只审图引用与图注，未打开SVG本体或网页，无法声称“配图精美”。数据分布、hash分组、BPE、mask、offset、温度采样均适合有学习任务的矢量图；需要另一轮逐图视觉审稿与数学标记检查。 |
| T21 | P2 | notation/15/16等记号 | 原始流s又写x/z；E同时expert数和embedding；B同时batch和bytes；t同时token与更新步。补全维度记号表并用粗体矩阵、N_bytes、chunk c减少冲突。中文“四个词”实际列五个。 |
| T22 | P2 | 11/15/多页结尾 | 已有Transition/下一章段后再新增Summary又重复下一章。合并为一次真实问题过渡；不能仅删除小结标题而保留重复正文。 |

## 本轮覆盖及没有批准的事项

下面的逐块附录记录每页全部非空内容块，使用 `文件@567fef3:Pnnn` 和原快照行号。每个内容块保留其节名和首句，便于下一轮针对性复审。标题用于定位，正文、列表、公式、代码、表格、图注分别计块；同一代码/公式块内部空行不另分段。块内缺陷可通过 F/T 编号追溯上面的证据和修复要求。

“保留/静态”表示本轮阅读没有给该块新增独立错误项，不表示该章符合出版要求；核心教学链未补全的章节仍因 T 类事项不通过。代码块统一待远端执行，插图统一待视觉检查，外链参考的全文证据统一待来源核验。未审高级章节05–09、frontier、网页导航、站点部署和全仓实现，不能把本记录用作它们的审稿证据。

## 修改后复审状态

根代理授权本子代理在写完基线报告后修订Part I十页；数据10–20、notation、00由根代理处理。**本文不批准自己的修订。** 新版本需要另一名审稿者核对F项实际关闭情况，P1教学深度仍需后续重写；远端运行、SVG美术与网页渲染仍单独待验收。

2026-11-11 实施记录：已修改Part I十页的C配置、真实输出字段、解释器、初始化兼容条件、恢复路径、配置约束、层轴/时间轴、尾块不提交更新、高阶梯度及验证报告字段。02两版新增相同的连续/step2恢复命令与数据/状态比较代码；03两版新增相同的四token二维查表/投影手算。上述代码只写入文稿，未执行。静态 `git diff --check` 未报告空白错误；这不检查 Python 语法、库调用、网页渲染或运行行为。F项修复是否充分由交叉审稿者复核，T项不因新增代码块自动关闭。

## 逐块附录

### docs/book/notation.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| zh-notation-P001 / L1 | 全书统一记号 / # 全书统一记号 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-notation-P002 / L3 | 本章导读 / ## 本章导读 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-notation-P003 / L5 | 本章导读 / 后面的章节会同时讨论数据、张量、概率、优化器和运行时状态。如果同一个字母在不同章节中表示不同东西，读者会很难判断公式和代码是否在描述同一个对象。本章先用一条很短的中文句子走完一次语言模型输入，再固定全书会反复使用的记号。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-notation-P004 / L7 | 本章导读 / 读完本章，你应该能完成三件事：把文本写成 token ID；把 token 流变成输入 \(X\)、目标 \(Y\) 和损失掩码 \(M\)；看到一个 logits 向量时，知道它如何变成概率和 loss。数学位置从 1… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-notation-P005 / L9 | 一个贯穿例子：猫坐下 / ## 一个贯穿例子：猫坐下 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-notation-P006 / L11 | 一个贯穿例子：猫坐下 / 为了手算，我们假设 tokenizer 把“猫坐下”以及句末标记分成四个 token，并给出一组**教学用** ID： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-notation-P007 / L13 | 一个贯穿例子：猫坐下 / ~~~text 文本 token：猫 坐 下 &lt;EOS&gt; token ID： 17 42 31 2 数学位置： 1 2 3 4 代码索引： 0 1 2 3 ~~~ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| zh-notation-P008 / L20 | 一个贯穿例子：猫坐下 / 真实 tokenizer 的 ID 不一定是 17、42、31、2；ID 只是词表中的整数坐标。模型不会看到“猫”这个汉字，而会看到整数序列 \(s=(17,42,31,2)\)。这个例子中的 \(s\) 表示 toke… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-notation-P009 / L22 | 一个贯穿例子：猫坐下 / 数学位置 \(t=1\) 对应代码里的 &lt;code&gt;tokens[0]&lt;/code&gt;，一般关系是： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-notation-P010 / L24 | 一个贯穿例子：猫坐下 / $$ \text{数学位置 }t \quad\longleftrightarrow\quad \text{代码索引 }t-1. $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| zh-notation-P011 / L30 | 一个贯穿例子：猫坐下 / 如果代码打印了 &lt;code&gt;tokens[2]&lt;/code&gt;，它表示数学位置 \(t=3\)，不是位置 2。我们保留这种差异，因为数学推导用从 1 开始的求和更自然，而 Python、NumPy 和 JAX 都用从 0 … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-notation-P012 / L32 | 从 token 流到 \(X,Y,M\) / ## 从 token 流到 \(X,Y,M\) | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-notation-P013 / L34 | 从 token 流到 \(X,Y,M\) / 因果语言模型的任务是：给定当前位置以前看到的 token，预测下一个 token。对上面的四个 token，取前三个作为输入： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-notation-P014 / L36 | 从 token 流到 \(X,Y,M\) / $$ X=(x_1,x_2,x_3)=(17,42,31), \qquad Y=(y_1,y_2,y_3)=(42,31,2). $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| zh-notation-P015 / L42 | 从 token 流到 \(X,Y,M\) / 逐项看，位置 1 输入“猫”并预测“坐”，位置 2 输入“猫、坐”并预测“下”，位置 3 输入“猫、坐、下”并预测 EOS。代码中的数组就是： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-notation-P016 / L44 | 从 token 流到 \(X,Y,M\) / ~~~python input_ids = [17, 42, 31] labels = [42, 31, 2] loss_mask = [1, 1, 1] ~~~ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| zh-notation-P017 / L50 | 从 token 流到 \(X,Y,M\) / 这里的 &lt;code&gt;input_ids&lt;/code&gt; 对应数学 \(X\)，&lt;code&gt;labels&lt;/code&gt; 对应 \(Y\)，&lt;code&gt;loss_mask&lt;/code&gt; 对应 \(M\)。如果把两篇文档直接拼接… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-notation-P018 / L52 | 从 token 流到 \(X,Y,M\) / $$ M=(1,1,0) $$ | 改写/P0 | F02；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-notation-P019 / L56 | 从 token 流到 \(X,Y,M\) / 可以屏蔽跨文档的最后一个目标；而连续语料训练可能使用 \(M=(1,1,1)\)。mask 不是装饰字段，它定义哪些预测真正进入优化目标。 | 改写/P0 | F02；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-notation-P020 / L58 | 索引、形状和 batch / ## 索引、形状和 batch | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-notation-P021 / L60 | 索引、形状和 batch / 全书使用以下基本符号： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-notation-P022 / L62 | 索引、形状和 batch / &#124; 对象 &#124; 记号 &#124; 约定 &#124; &#124;---&#124;---&#124;---&#124; &#124; batch 样本 &#124; \(b\) &#124; \(b\in\{1,\ldots,B\}\) &#124; &#124; token 位置 &#124; \(t\) &#124; \(t\in\{1,\l… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-notation-P023 / L74 | 索引、形状和 batch / 一个 batch 的输入张量写成： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-notation-P024 / L76 | 索引、形状和 batch / $$ X\in\{0,\ldots,V-1\}^{B\times L}. $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| zh-notation-P025 / L80 | 索引、形状和 batch / 例如 \(B=1,L=3\) 时，&lt;code&gt;input_ids&lt;/code&gt; 的形状是 &lt;code&gt;[1, 3]&lt;/code&gt;；模型把每个 ID 查成 \(d\) 维向量后，hidden 的形状变成 &lt;code&gt;[1,… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-notation-P026 / L82 | embedding、hidden、logits 和概率 / ## embedding、hidden、logits 和概率 | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-notation-P027 / L84 | embedding、hidden、logits 和概率 / 四个词容易混在一起： | 补写/P1 | T01、T21；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-notation-P028 / L86 | embedding、hidden、logits 和概率 / 1. **token ID** 是词表中的整数坐标，例如 17； 2. **embedding** 是查表得到的 \(d\) 维向量； 3. **hidden state** 是经过模型层变换后的 \(d\) 维残差流；… | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-notation-P029 / L92 | embedding、hidden、logits 和概率 / 设 embedding 矩阵为 \(E\in\mathbb{R}^{V\times d}\)。token 17 的向量是 \(E_{17,:}\)。某个位置经过模型后得到 hidden \(h\in\mathbb{R}^… | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-notation-P030 / L94 | embedding、hidden、logits 和概率 / $$ z=hW_{\mathrm{out}}\in\mathbb{R}^{V}. $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-notation-P031 / L98 | embedding、hidden、logits 和概率 / 第 \(v\) 个分量 \(z_v\) 只是分数，不是概率。softmax 把它变成： | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-notation-P032 / L100 | embedding、hidden、logits 和概率 / $$ p_\theta(v\mid x_{1:t}) = \frac{\exp z_v}{\sum_{u=0}^{V-1}\exp z_u}. $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-notation-P033 / L106 | embedding、hidden、logits 和概率 / 分母把所有候选的指数分数加起来，所以 \(p_\theta(v\mid x_{1:t})\) 在所有 \(v\) 上相加等于 1。词表大小 \(V\) 因而同时决定输出头的最后一维和概率分布的类别数。 | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-notation-P034 / L108 | 损失、梯度和两个时间尺度 / ## 损失、梯度和两个时间尺度 | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-notation-P035 / L110 | 损失、梯度和两个时间尺度 / 如果目标 token 是 \(y_t\)，这个位置的负对数似然为： | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-notation-P036 / L112 | 损失、梯度和两个时间尺度 / $$ \ell_t^{\mathrm{tok}}(\theta) = -\log p_\theta(y_t\mid x_{1:t}). $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-notation-P037 / L118 | 损失、梯度和两个时间尺度 / 有 \(m_t\in\{0,1\}\) 时，一个窗口的平均 loss 是： | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-notation-P038 / L120 | 损失、梯度和两个时间尺度 / $$ \mathcal{L} = \frac{\sum_{t=1}^{L}m_t\ell_t^{\mathrm{tok}}} {\max(\sum_{t=1}^{L}m_t,1)}. $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-notation-P039 / L127 | 损失、梯度和两个时间尺度 / 为什么梯度会出现 \(p-\text{one-hot}\)？设 logits 向量为 \(z\)，正确类别为 \(y\)： | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-notation-P040 / L129 | 损失、梯度和两个时间尺度 / $$ \ell=-\log\frac{\exp z_y}{\sum_v\exp z_v} =-z_y+\log\sum_v\exp z_v. $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-notation-P041 / L134 | 损失、梯度和两个时间尺度 / 对第 \(j\) 个分量求导，第一项贡献 \(-\mathbf{1}[j=y]\)，第二项贡献 softmax 概率 \(p_j\)，因此： | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-notation-P042 / L136 | 损失、梯度和两个时间尺度 / $$ \frac{\partial\ell}{\partial z_j} =p_j-\mathbf{1}[j=y]. $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-notation-P043 / L141 | 损失、梯度和两个时间尺度 / 正确类别的梯度为负，梯度下降会提高它的分数；错误类别的梯度为正，梯度下降会降低它们的分数。这个局部推导会在 baseline 和 TTT 内循环中再次使用。 | 改写/P0 | T01、F36；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-notation-P044 / L143 | 损失、梯度和两个时间尺度 / 静态参数跨训练 step 更新： | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-notation-P045 / L145 | 损失、梯度和两个时间尺度 / $$ \theta_{k+1} = \operatorname{Opt}_{\mathrm{outer}} \left(\theta_k,\nabla_\theta\mathcal{L}_{\mathrm{outer}}… | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-notation-P046 / L152 | 损失、梯度和两个时间尺度 / TTT 的快权重在序列或 chunk 内更新： | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-notation-P047 / L154 | 损失、梯度和两个时间尺度 / $$ \phi_{t+1} = \phi_t-\eta_{\mathrm{inner}} \nabla_\phi\mathcal{L}_{\mathrm{inner}}. $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-notation-P048 / L161 | 损失、梯度和两个时间尺度 / \(k\) 和 \(t\) 不能混为同一个下标：前者描述训练过程的长期状态，后者描述当前序列的短期适应。 | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-notation-P049 / L163 | 数据集、分布和状态 owner / ## 数据集、分布和状态 owner | 修订/P2 | T21；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-notation-P050 / L165 | 数据集、分布和状态 owner / 数据集 \(\mathcal{D}\) 是可以检查文件、manifest 和 hash 的对象；分布 \(D\) 是采样器实际让模型看到的概率分布。例如两个文件各有 100 个 token，按文件数均匀采样得到的分布不一… | 改写/P0 | F03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-notation-P051 / L167 | 数据集、分布和状态 owner / 状态也必须有 owner 和 lifetime： | 修订/P2 | T21；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-notation-P052 / L169 | 数据集、分布和状态 owner / &#124; 状态 &#124; 记号 &#124; owner &#124; 生命周期 &#124; &#124;---&#124;---&#124;---&#124;---&#124; &#124; 静态参数 &#124; \(\theta\) &#124; model/checkpoint &#124; 多个 batch &#124; &#124; 优化器状态 &#124; \(O… | 修订/P2 | T21；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-notation-P053 / L177 | 数据集、分布和状态 owner / “模型状态”不是一个足够精确的词。只恢复参数而不恢复 optimizer 和 sampler，可能加载成功，却不能称为同一个训练过程的 resume。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-notation-P054 / L179 | 代码检查与证据 / ## 代码检查与证据 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-notation-P055 / L181 | 代码检查与证据 / 记号本身不需要 GPU。远端 CPU 节点上可以先生成 tiny token 文件： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-notation-P056 / L183 | 代码检查与证据 / ~~~bash bash scripts/prepare_sample.sh outputs/sample-tokens.npy python3 -m ttt.data --input examples/sample.t… | 改写/P0 | F01；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-notation-P057 / L188 | 代码检查与证据 / 成功时应看到 token 文件、分片文件和 &lt;code&gt;manifest.json&lt;/code&gt;；检查 &lt;code&gt;vocab_size&lt;/code&gt;、shard token 数和输入摘要是否存在。这个检查只能证明数据对… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-notation-P058 / L190 | 小结与下一问 / ## 小结与下一问 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-notation-P059 / L192 | 小结与下一问 / 本章固定了位置、形状、\(X/Y/M\)、embedding/hidden/logits/probability、loss/gradient，以及 \(\theta\) 和 \(\phi\) 的状态边界。下一章用这些记号… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-notation-P060 / L194 | 书写规则 / ## 书写规则 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-notation-P061 / L196 | 书写规则 / 标量使用斜体小写或大写字母，向量使用粗体小写，矩阵使用粗体大写，集合或损失使用花体。&lt;code&gt;d_model&lt;/code&gt;、&lt;code&gt;num_heads&lt;/code&gt;、&lt;code&gt;vocab_size&lt;/code&gt; … | 改写/P0 | F04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |

### docs/book/00_why_pretrain.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| zh-00_why_pretrain-P001 / L1 | 序章　为什么从零训练一个基模 / # 序章 为什么从零训练一个基模 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P002 / L3 | 本章导读 / ## 本章导读 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P003 / L5 | 本章导读 / 如果只想调用一个已经训练好的模型，最短路径是下载权重、准备提示词，然后调用生成接口。本书选择更长的路线：从数据文件开始，亲手构造 tokenizer、输入窗口、loss、优化器和 checkpoint，再研究 TTT、线… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P004 / L7 | 本章导读 / 这条路线回答的不是“怎样尽快得到一个会说话的模型”，而是一个更基础的问题： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P005 / L9 | 本章导读 / &gt; 在数据、表示、训练预算和设备都可审计的条件下，序列模型怎样保存、更新和使用上下文？ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P006 / L11 | 本章导读 / 读完本章，读者应该能够解释一个 token 预测任务、手算一个位置的概率和 loss、说出一次预训练由哪些部分组成，并在远端 CPU 机器上生成第一份 tiny 数据产物。本章不声称 tiny 模型具备有用的语言能力；它… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P007 / L13 | 一个具体场景：模型如何完成“猫坐下” / ## 一个具体场景：模型如何完成“猫坐下” | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P008 / L15 | 一个具体场景：模型如何完成“猫坐下” / 人看到“猫”以后，可能会预测接下来是“坐”。模型并不直接处理汉字，而是先把文本交给 tokenizer。为了手算，我们使用一组教学 ID： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P009 / L17 | 一个具体场景：模型如何完成“猫坐下” / ~~~text 文本： 猫 坐 下 &lt;EOS&gt; token ID： 17 42 31 2 ~~~ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| zh-00_why_pretrain-P010 / L22 | 一个具体场景：模型如何完成“猫坐下” / 模型在第一个位置收到 17，要在词表中的所有 ID 里给出“下一个 token”的分数。它并不是只输出“坐”或“不坐”，而是输出长度为 \(V\) 的 logits 向量。softmax 把这组分数变成概率；交叉熵再根据… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P011 / L24 | 一个具体场景：模型如何完成“猫坐下” / 这个小场景包含了本书的主线：数据决定模型看到哪句话，tokenizer 决定这句话如何变成整数，架构决定上下文怎样影响 hidden state，loss 和优化器决定参数怎样改变，系统决定计算能否完成，评估决定我们能否… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P012 / L26 | 预备知识与完成标准 / ## 预备知识与完成标准 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P013 / L28 | 预备知识与完成标准 / 读者不需要先学完完整的深度学习课程，但需要知道数组、函数和命令行的基本用法。线性代数、概率和梯度会在首次使用处补齐。读者完成本章后应能： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P014 / L30 | 预备知识与完成标准 / 1. 区分 token ID、embedding、hidden、logits 和 probability； 2. 从一条 token 流构造 \(X\)、\(Y\) 和 \(M\)； 3. 从 softmax 和负对数似… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P015 / L36 | 基模究竟在学习什么 / ## 基模究竟在学习什么 | 补写/P1 | T02；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P016 / L38 | 基模究竟在学习什么 / 设 tokenizer 输出 token 流 \(x_1,\ldots,x_L\)。自回归模型把整段序列的概率分解成每个位置的条件概率： | 补写/P1 | T02；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P017 / L40 | 基模究竟在学习什么 / $$ p_\theta(x_{1:L}) = \prod_{t=1}^{L}p_\theta(x_t\mid x_{&lt;t}). $$ | 补写/P1 | T02；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-00_why_pretrain-P018 / L46 | 基模究竟在学习什么 / 这里 \(x_{&lt;t}\) 表示位置 \(t\) 之前的 token，\(\theta\) 是模型参数。训练时，我们希望真实 token 获得更高概率，因此最小化平均负对数似然： | 补写/P1 | T02；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P019 / L48 | 基模究竟在学习什么 / $$ \mathcal{L}(\theta) = -\frac{1}{L} \sum_{t=1}^{L} \log p_\theta(x_t\mid x_{&lt;t}). $$ | 补写/P1 | T02；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-00_why_pretrain-P020 / L56 | 基模究竟在学习什么 / 这个目标是“局部”的：每次只评价下一个 token。但要预测代码中的变量、长句中的指代或题目中的答案，模型可能需要使用很远的上下文。于是，减少局部预测误差会推动模型学习语法、格式和部分事实结构。 | 补写/P1 | T02；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P021 / L58 | 基模究竟在学习什么 / 这不等于低 loss 自动保证推理、事实正确或指令遵循。若训练集有大量重复，记忆重复片段也能降低 loss；若验证集被训练数据污染，验证 loss 会过于乐观；若 tokenizer 改变了 token 数，固定字符量对… | 补写/P1 | T02；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P022 / L60 | 从 logits 到概率、loss 和梯度 / ## 从 logits 到概率、loss 和梯度 | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P023 / L62 | 从 logits 到概率、loss 和梯度 / 假设上下文是“猫坐”，正确的下一个 token 是“下”。模型对三个候选 token 给出 logits： | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P024 / L64 | 从 logits 到概率、loss 和梯度 / $$ z=(2,1,0). $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-00_why_pretrain-P025 / L68 | 从 logits 到概率、loss 和梯度 / softmax 定义为： | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P026 / L70 | 从 logits 到概率、loss 和梯度 / $$ p_j=\frac{\exp z_j}{\sum_u\exp z_u}. $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-00_why_pretrain-P027 / L74 | 从 logits 到概率、loss 和梯度 / 因此三个概率约为 \((0.665,0.245,0.090)\)。如果正确 token 是第二个，负对数似然为： | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P028 / L76 | 从 logits 到概率、loss 和梯度 / $$ \ell=-\log(0.245)\approx 1.407. $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-00_why_pretrain-P029 / L80 | 从 logits 到概率、loss 和梯度 / 当模型把正确 token 的概率提高到 \(0.8\) 时，loss 变成 \(-\log(0.8)\approx0.223\)。所以“loss 下降”首先表示模型把更多概率放到了正确 token 上，而不是直接表示模型… | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P030 / L82 | 从 logits 到概率、loss 和梯度 / 把 loss 展开： | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P031 / L84 | 从 logits 到概率、loss 和梯度 / $$ \ell = -\log\frac{\exp z_y}{\sum_v\exp z_v} = -z_y+\log\sum_v\exp z_v. $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-00_why_pretrain-P032 / L92 | 从 logits 到概率、loss 和梯度 / 对第 \(j\) 个 logits 求导： | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P033 / L94 | 从 logits 到概率、loss 和梯度 / $$ \frac{\partial\ell}{\partial z_j} = p_j-\mathbf{1}[j=y]. $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-00_why_pretrain-P034 / L100 | 从 logits 到概率、loss 和梯度 / 正确类别的梯度为 \(p_y-1&lt;0\)，梯度下降会提高它的分数；错误类别的梯度为 \(p_j&gt;0\)，梯度下降会降低它们的分数。这个推导以后会连接到 dense baseline、inner SGD 和 meta-gr… | 改写/P0 | T01、F36；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P035 / L102 | 从 logits 到概率、loss 和梯度 / 有 padding 或 prompt mask 时，只对 \(m_t=1\) 的位置求平均： | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P036 / L104 | 从 logits 到概率、loss 和梯度 / $$ \mathcal{L} = \frac{\sum_{t=1}^{L} m_t[-\log p_\theta(y_t\mid x_{\le t})]} {\max(\sum_{t=1}^{L}m_t,1)}. $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-00_why_pretrain-P037 / L112 | 从 logits 到概率、loss 和梯度 / 若把 input 和 label 错开一位，模型训练的就不是同一个 next-token 任务；若把 padding 计入 loss，模型会被迫学习填充符号。这是数据管线问题，不是模型结构问题。 | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P038 / L114 | 为什么要从零训练 / ## 为什么要从零训练 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P039 / L116 | 为什么要从零训练 / 已有权重对应用开发通常更合适。从零训练的研究价值在于，我们可以控制数据、词表、架构、优化器、token 预算和设备，然后问一个可归因的问题：在相同条件下，新增的状态更新是否改善未见上下文的预测？ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P040 / L118 | 为什么要从零训练 / 例如两个模型在长文问答上相差两分。如果它们使用不同 tokenizer、不同训练 token 数和不同后训练配方，我们无法判断差异来自 TTT 还是数据。固定实验契约后，控制变量才有意义。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P041 / L120 | 为什么要从零训练 / 代价也同样清楚：数据许可、清洗、存储、训练稳定性、GPU 调度、通信和恢复都需要自己负责。因此本书先运行 tiny 规模，依次证明“计算图能运行”“目标能学习”“对照能解释”，最后才考虑更大的模型和上下文。tiny 结果… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P042 / L122 | 一次预训练包含哪些部分 / ## 一次预训练包含哪些部分 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P043 / L124 | 一次预训练包含哪些部分 / &#124; 部分 &#124; 要回答的问题 &#124; 典型失败 &#124; &#124;---&#124;---&#124;---&#124; &#124; 数据 &#124; 模型实际看到了什么分布？ &#124; 训练/验证近重复，loss 虚低 &#124; &#124; 表示 &#124; 文本如何变成 token、位置和 mask？ &#124; … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P044 / L133 | 一次预训练包含哪些部分 / 这六部分相互耦合。词表大小 \(V\) 同时改变 embedding 和输出头；序列长度 \(L\) 改变激活和注意力成本；TTT 改变状态与梯度边界；MoE 改变通信和专家负载。书中不把它们当成可以随意交换的独立按钮。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P045 / L135 | 预算：参数、token、吞吐和墙钟 / ## 预算：参数、token、吞吐和墙钟 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P046 / L137 | 预算：参数、token、吞吐和墙钟 / 设模型有 \(N_{\mathrm{param}}\) 个参数，训练消费 \(N_{\mathrm{tok}}\) 个 token，实测稳态吞吐为 \(r_{\mathrm{tok}}\)。常见的稠密 Transform… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P047 / L139 | 预算：参数、token、吞吐和墙钟 / $$ F_{\mathrm{rough}} \approx 6N_{\mathrm{param}}N_{\mathrm{tok}}. $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| zh-00_why_pretrain-P048 / L145 | 预算：参数、token、吞吐和墙钟 / 它没有完整覆盖长序列注意力、特殊 kernel、通信和 TTT 内循环，不能直接当作价格或完成时间。工程估计应该使用实际吞吐： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P049 / L147 | 预算：参数、token、吞吐和墙钟 / &#124; 量 &#124; 示例值 &#124; 解释 &#124; &#124;---&#124;---:&#124;---&#124; &#124; 训练 token &#124; \(10^9\) &#124; 预算，不是文件字节数 &#124; &#124; 稳态吞吐 &#124; \(50{,}000\) token/s &#124; 需要在目标配置测量… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P050 / L154 | 预算：参数、token、吞吐和墙钟 / 参数和数据也要一起报告。对 \(d=768\)，词表从 32K 变成 128K 时，单个 embedding 的参数量从约 \(24.6\)M 变成 \(98.3\)M；同样的模型宽度并不代表同样的参数预算。Scalin… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P051 / L156 | 本章的第一个可运行检查 / ## 本章的第一个可运行检查 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P052 / L158 | 本章的第一个可运行检查 / 先在约定的远端 CPU 机器上生成 tiny 数据： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P053 / L160 | 本章的第一个可运行检查 / ~~~bash bash scripts/prepare_sample.sh outputs/sample-tokens.npy python3 -m ttt.data \ --input examples/sample… | 改写/P0 | F01；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-00_why_pretrain-P054 / L170 | 本章的第一个可运行检查 / 预期产物是 &lt;code&gt;outputs/sample-tokens.npy&lt;/code&gt;、若干 shard 和 &lt;code&gt;manifest.json&lt;/code&gt;。先检查 manifest 中的 &lt;code&gt;vocab… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P055 / L172 | 本章的第一个可运行检查 / 全书使用三层完成标准：**可运行**表示入口和依赖成立；**可学习**表示 loss、有效 token 和参数更新呈现合理变化；**可解释**表示控制变量、评估和报告足以支持结论。完成上一层不能跳过下一层。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P056 / L174 | 主线与前沿专题 / ## 主线与前沿专题 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P057 / L176 | 主线与前沿专题 / 主线按依赖关系推进：统一记号、实验契约、数据、dense baseline、状态与计算、消融和验证。Frontier Lab 研究 TTT 与线性注意力、共享层循环、长上下文等快速变化的问题，但每个专题都必须标注论文版本… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P058 / L178 | 主线与前沿专题 / 前沿页面可以提出假设，不能把论文数字当成本项目结果。只有当它有可运行的 baseline、控制变量、指标和失败判据时，才进入主线实验。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P059 / L180 | 小结与下一问 / ## 小结与下一问 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-00_why_pretrain-P060 / L182 | 小结与下一问 / 本章建立了“为什么训练、训练什么、怎样解释证据”的总问题，并手算了 token、softmax、NLL、梯度和预算。下一章把这些对象写成实验契约：一次运行必须固定什么、保存什么，以及中断后怎样证明它真的恢复了同一个过程。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/01_experiment_contract.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| zh-01_experiment_contract-P001 / L1 | 1　实验契约：让一次运行成为一个可以解释的实验 / # 1 实验契约：让一次运行成为一个可以解释的实验 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P002 / L3 | 本章导读：先把“比较”写成一个问题 / ## 本章导读：先把“比较”写成一个问题 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P003 / L5 | 本章导读：先把“比较”写成一个问题 / 假设我们训练同一个 tiny 模型两次：第一次关闭 TTT 内循环，第二次打开它。如果第二次的 loss 更低，我们还不能立刻说“TTT 有效”。两次运行可能读了不同的 token、用了不同的学习率，甚至第二次从一个已经… | 改写/P0 | F35；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P004 / L7 | 本章导读：先把“比较”写成一个问题 / 读完本章，你应该能完成一件具体的事：给一个 control/variant 对照填写研究变量、固定变量、预算和证据文件，并判断一次运行属于“可运行”“可学习”还是“可解释”。本章只讨论实验身份；模型结构和数据处理分别在后… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P005 / L9 | 贯穿例子：tiny-contract-001 / ## 贯穿例子：tiny-contract-001 | 改写/P0 | F05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P006 / L11 | 贯穿例子：tiny-contract-001 / 后文使用 &lt;code&gt;tiny-contract-001&lt;/code&gt; 作为示例。它读取 &lt;code&gt;configs/tiny.json&lt;/code&gt; 和由 &lt;code&gt;examples/sample.txt&lt;/code… | 改写/P0 | F05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P007 / L13 | 贯穿例子：tiny-contract-001 / &#124; 字段 &#124; control &#124; variant &#124; &#124;---&#124;---&#124;---&#124; &#124; 实验身份 &#124; &lt;code&gt;tiny-contract-001-C&lt;/code&gt; &#124; &lt;code&gt;tiny-contract-001-F… | 改写/P0 | F06、F35、F05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P008 / L21 | 贯穿例子：tiny-contract-001 / 这张表先于任何结果存在。它防止我们在看到结果以后才修改“比较条件”。 | 改写/P0 | F05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P009 / L23 | 先约定我们要证明什么 / ## 先约定我们要证明什么 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P010 / L25 | 先约定我们要证明什么 / 这本书不把“代码能够执行”当作研究结论。我们要回答的是：在相同的数据、训练预算和系统条件下，一个新的序列模块是否改变了模型保存、更新和使用上下文的方式，并且这种改变是否带来可重复的质量或效率收益。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P011 / L27 | 先约定我们要证明什么 / 因此，每次实验开始前必须写出一个具体主张： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P012 / L29 | 先约定我们要证明什么 / &gt; 在固定 ______ 的条件下，只改变 ______，观察 ______ 是否发生变化。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P013 / L31 | 先约定我们要证明什么 / 例如：在固定 tokenizer、有效训练 token、序列长度、优化器和硬件的条件下，只打开 TTT 的内循环，观察 held-out loss、快权重状态和训练成本是否变化。如果没有这句话，实验很容易同时改变多个因素… | 改写/P0 | F35；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P014 / L33 | 实验的身份 / ## 实验的身份 | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P015 / L35 | 实验的身份 / 一次运行不是一条命令，而是一个有身份的对象。可以把它写成： | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P016 / L37 | 实验的身份 / $$ R=(S,C,I_{\mathrm{data}},E,K,Q_{\mathrm{budget}},O), $$ | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-01_experiment_contract-P017 / L41 | 实验的身份 / 其中： | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P018 / L43 | 实验的身份 / - \(S\)：源码 commit； - \(C\)：解析后的模型和训练配置； - \(I_{\mathrm{data}}\)：数据、Tokenizer 和 manifest 的指纹； - \(E\)：操作系统、Pyth… | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P019 / L51 | 实验的身份 / 复现的目标不一定是让每个浮点数完全相同，而是让这些定义实验的对象一致，并把剩余的数值差异记录下来。 | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P020 / L53 | 实验的身份 / 命令只是入口。两个运行即使命令相同，只要读取了不同的数据、从不同 checkpoint 开始，或使用了不同的默认配置，就不是同一个实验。 | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P021 / L55 | 哪些变量必须固定 / ## 哪些变量必须固定 | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P022 / L57 | 哪些变量必须固定 / 比较架构时，至少要固定或明确报告以下变量： | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P023 / L59 | 哪些变量必须固定 / &#124; 类别 &#124; 内容 &#124; &#124;---&#124;---&#124; &#124; 数据 &#124; 来源、清洗版本、去重规则、train/validation manifest、有效 token 数 &#124; &#124; 表示 &#124; tokenizer 文件、词表大小、特殊 t… | 改写/P0 | F35、T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P024 / L69 | 哪些变量必须固定 / 一次实验可以改变其中一项，但必须把改变写成研究变量。若同时改变架构、Tokenizer 和训练 token，就应把它称为新的配方，而不是架构消融。 | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P025 / L71 | 哪些变量必须固定 / 一个常见的不可归因例子是：control 使用 byte tokenizer、sequence length 16，variant 换成子词 tokenizer、sequence length 32，同时 loss 下降… | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P026 / L73 | 三层完成标准 / ## 三层完成标准 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P027 / L75 | 三层完成标准 / 每次运行都按三个层次报告。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P028 / L77 | 三层完成标准 / **可运行**：依赖、设备、配置和数据入口成立，程序可以启动并产生输出。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P029 / L79 | 三层完成标准 / **可学习**：loss 和梯度是有限值，训练状态能够更新，checkpoint 可以读取，恢复后的下一步与连续运行一致或在预先定义的容忍范围内一致。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P030 / L81 | 三层完成标准 / **可解释**：对照组和实验组只有声明的变量不同，评测使用独立且可信的 manifest，结果包含质量、成本和状态证据，能够支持一个边界清楚的结论。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P031 / L83 | 三层完成标准 / 第一层不是第二层，第二层也不是第三层。tiny 模型通过四步训练，只能说明接口和优化路径基本连通；它不能证明模型有语言能力，更不能证明某个架构优于另一个架构。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P032 / L85 | 运行目录必须自己解释自己 / ## 运行目录必须自己解释自己 | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P033 / L87 | 运行目录必须自己解释自己 / 正式运行至少保存： | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P034 / L89 | 运行目录必须自己解释自己 / ~~~text experiment.json # 输入、解析后的配置和预算 source.json # commit 和工作区状态 environment.json # Python、JAX、设备、驱动和 dtype … | 改写/P0 | F06、T03；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-01_experiment_contract-P035 / L100 | 运行目录必须自己解释自己 / JSON、JSONL 和 checkpoint 是事实来源，Markdown 报告是阅读入口。报告中的数字必须能够回到运行目录中的原始记录；手工抄写的数字不能替代机器生成的证据。 | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P036 / L102 | 运行目录必须自己解释自己 / 对 &lt;code&gt;tiny-contract-001&lt;/code&gt;，读者应能按下面的映射找到证据： | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P037 / L104 | 运行目录必须自己解释自己 / &#124; 论断 &#124; 首要证据 &#124; 成功判据 &#124; &#124;---&#124;---&#124;---&#124; &#124; 运行使用了预期配置 &#124; &lt;code&gt;experiment.json&lt;/code&gt; &#124; 与 &lt;code&gt;configs/tiny.json&lt;/cod… | 改写/P0 | F06、T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P038 / L112 | 运行目录必须自己解释自己 / 训练 checkpoint、服务 session snapshot 和实验报告不是同一种对象。训练 checkpoint 需要恢复模型、优化器、global step 和数据状态；session snapshot 需要恢… | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P039 / L114 | resume 不等于 initialize-from / ## resume 不等于 initialize-from | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P040 / L116 | resume 不等于 initialize-from / `resume` 继续同一个优化过程，必须恢复模型参数、optimizer state、global step、采样器状态、数据指纹和学习率进度。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P041 / L118 | resume 不等于 initialize-from / `initialize-from` 使用旧参数开启一个新实验，应该有新的运行身份；它可以重新建立 optimizer、step 和数据顺序。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P042 / L120 | resume 不等于 initialize-from / 如果只载入参数却把它称为 resume，后面的 loss 曲线和训练预算就无法解释。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P043 / L122 | 失败也属于实验结果 / ## 失败也属于实验结果 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P044 / L124 | 失败也属于实验结果 / 失败可能发生在依赖安装、设备发现、XLA 编译、数据读取、梯度更新、checkpoint 保存或恢复阶段。失败目录至少保留命令、作业 ID、最后成功解析的配置、环境摘要、已有 metrics、异常阶段和是否可以安全恢复。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P045 / L126 | 失败也属于实验结果 / 我们不把所有失败都归结为“随机性”。如果恢复后下一批数据不同，应先检查采样器状态和数据指纹；如果设备 OOM，应记录 batch、sequence length、dtype 和编译策略；如果 loss 为 NaN，应保留… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P046 / L128 | 失败也属于实验结果 / 失败记录的作用不是证明项目顺利，而是缩小下一次实验的未知范围。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P047 / L130 | 结论的边界 / ## 结论的边界 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P048 / L132 | 结论的边界 / 实验报告最后必须分别回答： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P049 / L134 | 结论的边界 / 1. 这次运行实际改变了什么？ 2. 哪些指标发生了变化？ 3. 这些变化能否归因于声明的变量？ 4. 结果是否在其他 seed、数据或规模上复现？ 5. 哪些更强的结论仍然没有证据？ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P050 / L140 | 结论的边界 / 可以使用下面的句式： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P051 / L142 | 结论的边界 / &gt; 在固定的 tiny 配置和已声明控制变量下，只改变一个研究变量，我们观察到某个指标发生变化。该结果支持一个边界清楚的结论，但不能直接推广到尚未控制或验证的条件。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P052 / L144 | 结论的边界 / 本书后面的每个章节都遵守这份契约。环境章节回答“在哪里运行”，数据与表示章节回答“模型实际看到了什么”，baseline 章节回答“系统是否真的会学习”，架构章节回答“状态如何改变”，实验章节回答“收益是否来自这个改变”… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P053 / L146 | 本章小结 / ## 本章小结 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-01_experiment_contract-P054 / L148 | 本章小结 / - 一个实验身份至少包含源码、解析配置、数据指纹、环境、随机与优化状态、预算和输出证据； - control/variant 必须先写固定项和唯一改变，再启动训练； - 可运行、可学习、可解释是三个不同门槛； - che… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/01_environment.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| zh-01_environment-P001 / L1 | 2　环境：第一次训练之前先固定什么 / # 2 环境：第一次训练之前先固定什么 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-01_environment-P002 / L3 | 本章导读：先证明机器能运行 / ## 本章导读：先证明机器能运行 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-01_environment-P003 / L5 | 本章导读：先证明机器能运行 / 本章不追求一次把大模型训练起来。我们先用 &lt;code&gt;tiny-contract-001&lt;/code&gt; 做一件更小但更可靠的事：让机器报告自己是谁，生成一份短 token 文件，启动一次 tiny 训练，并留下可以解释的… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P004 / L7 | 本章导读：先证明机器能运行 / 读完本章，你应该能在目标 Linux 节点完成三项检查：看到 JAX backend 和设备列表，看到样例 token 的长度与词表范围，看到 tiny 训练的 &lt;code&gt;experiment.json&lt;/code&gt;、… | 改写/P0 | F06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_environment-P005 / L9 | 预备知识与贯穿输入 / ## 预备知识与贯穿输入 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-01_environment-P006 / L11 | 预备知识与贯穿输入 / 你只需要会打开终端、运行 shell 命令，并理解“文件夹中的 JSON 是机器写出的记录”。本章使用 &lt;code&gt;examples/sample.txt&lt;/code&gt;；它不是质量足够高的预训练语料，而是用于验证路径的最… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P007 / L13 | 环境不是安装步骤的附录 / ## 环境不是安装步骤的附录 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-01_environment-P008 / L15 | 环境不是安装步骤的附录 / 从 scratch 项目经常出现一个奇怪现象：代码在作者机器上运行，换一台机器后不是依赖找不到，就是 loss 曲线完全不同。问题未必来自模型，可能来自 Python 小版本、默认 dtype、JAX 后端、CPU/GP… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P009 / L17 | 环境不是安装步骤的附录 / 所以环境是实验定义的一部分。我们希望别人拿到仓库后能回答：使用什么代码、什么依赖、什么设备、什么数据、什么命令，才能得到这个结果。只写“安装 JAX”远远不够，JAX 的 CPU 与 CUDA wheel、驱动兼容性和默… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P010 / L19 | 三层环境合同 / ## 三层环境合同 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-01_environment-P011 / L21 | 三层环境合同 / 把运行环境拆成三层，排错会更容易。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P012 / L23 | 三层环境合同 / **系统层**包括操作系统、内核、GPU 型号、驱动、CUDA、磁盘和网络文件系统。它决定设备能否被发现、编译缓存放在哪里，以及多个进程能否访问同一目录。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P013 / L25 | 三层环境合同 / **Python 层**包括 Python 版本、JAX、Optax、Orbax 和 tokenizer 扩展。它决定 API 和数值实现。`uv.lock` 固定了依赖版本，但不替代驱动与 CUDA 运行时检查。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P014 / L27 | 三层环境合同 / **实验层**包括 Git commit、解析后的模型配置、tokenizer、数据摘要、batch、sequence length、seed 和输出路径。它决定这一次究竟训练了什么。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P015 / L29 | 三层环境合同 / &#124; 层 &#124; 典型失败 &#124; 应保存的证据 &#124; &#124;---&#124;---&#124;---&#124; &#124; 系统 &#124; 找不到 GPU、OOM、通信失败 &#124; device、driver、backend、文件系统 &#124; &#124; Python &#124; import 失… | 改写/P0 | F06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_environment-P016 / L35 | 三层环境合同 / 这三层不能互相替代。一个正确的环境不保证配置正确，一个正确的配置也不保证 GPU 有足够显存。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P017 / L37 | 从仓库入口开始 / ## 从仓库入口开始 | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_environment-P018 / L39 | 从仓库入口开始 / `scripts/bootstrap_linux.sh` 在 Linux 节点下载 uv，安装指定 Python，并按照 extra 同步依赖。它默认包含开发、checkpoint 和 serving 依赖；GPU 作业… | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_environment-P019 / L41 | 从仓库入口开始 / `scripts/check_environment.py` 使用 JAX 报告当前后端和设备数量。`--require-gpu` 只在至少有目标 GPU 数量时通过，它不验证模型能否在目标显存中编译；那是后面的功能和规… | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_environment-P020 / L43 | 从仓库入口开始 / `scripts/record_environment.sh` 将运行时信息保存为 JSON。训练入口还会在输出目录写入 `environment.json`，这样报告不依赖终端日志。 | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_environment-P021 / L45 | 从仓库入口开始 / 在目标 Linux 环境中，第一次准备可以这样做： | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_environment-P022 / L47 | 从仓库入口开始 / ~~~bash TTT_INSTALL_CUDA=1 bash scripts/bootstrap_linux.sh python3 scripts/check_environment.py \ --require-gp… | 改写/P0 | F07、T03；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-01_environment-P023 / L55 | 从仓库入口开始 / 成功时，JSON 至少应包含 Python 版本、JAX 版本、backend、可见设备数量和设备名称。失败时先看 backend 与设备列表：若 CPU 节点被要求 &lt;code&gt;--require-gpu&lt;/code&gt;… | 改写/P0 | F07、T03、F06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_environment-P024 / L57 | 从仓库入口开始 / 没有 GPU 的 CPU 节点使用 `TTT_INSTALL_CUDA=0`。本项目约定 CPU 只用于功能与数据路径验证，GPU 吞吐不能用 CPU 结果代替。 | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_environment-P025 / L59 | 为什么先生成样例数据 / ## 为什么先生成样例数据 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-01_environment-P026 / L61 | 为什么先生成样例数据 / 直接拿正式语料启动，会把数据问题和模型问题混在一起。`scripts/prepare_sample.sh` 生成一个很小的一维 token 文件，让我们先检查： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P027 / L63 | 为什么先生成样例数据 / - token 是否位于 vocab_size 范围； - 数据长度是否能组成 sequence_length+1； - BatchStream 是否能取窗口； - loss mask 是否有有效位置； - checkp… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P028 / L69 | 为什么先生成样例数据 / 它不能证明语言能力，也不能证明清洗策略。它的作用类似电路板上的通断测试：先确认接口连接，再接入复杂输入。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P029 / L71 | 为什么先生成样例数据 / 样例输入的第一条检查结果应能回答三个问题：token 数是否至少为 &lt;code&gt;sequence_length + 1 = 17&lt;/code&gt;，所有 ID 是否位于 &lt;code&gt;[0, 259)&lt;/code&gt;，以及 sh… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P030 / L73 | 第一次训练的四个阶段 / ## 第一次训练的四个阶段 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-01_environment-P031 / L75 | 第一次训练的四个阶段 / 一次训练的第一步不等于模型已经开始稳态计算。通常包含： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P032 / L77 | 第一次训练的四个阶段 / 1. 进程启动与配置解析； 2. 参数初始化和随机 key 派生； 3. JAX tracing 与 XLA 编译； 4. 编译后的执行和指标写入。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P033 / L82 | 第一次训练的四个阶段 / 因此 `metrics.jsonl` 中的第一条耗时可能很大。报告吞吐时要区分 compile time 与 steady-state tokens/s，否则一次编译时间会被平均到短实验里，得出没有意义的性能数字。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P034 / L84 | 第一次训练的四个阶段 / 训练目录还应出现： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P035 / L86 | 第一次训练的四个阶段 / ~~~text experiment.json # 完整输入和解析后的配置 environment.json # Python、JAX、设备和源码摘要 metrics.jsonl # 每步 loss、梯度、速度 stat… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| zh-01_environment-P036 / L95 | 第一次训练的四个阶段 / 不同 checkpoint 格式的文件名可能不同，但身份和状态的含义不能改变。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P037 / L97 | 运行失败时先看哪一层 / ## 运行失败时先看哪一层 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-01_environment-P038 / L99 | 运行失败时先看哪一层 / 如果 `check_environment.py` 找不到 GPU，先看后端和设备，而不是改模型。若设备存在但初始化 OOM，检查 dtype、预分配比例、batch、sequence length 和模型配置。若初始化… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P039 / L101 | 运行失败时先看哪一层 / 若训练跑完但恢复结果不同，先对比 data fingerprint、采样器状态、optimizer state 和 global step。最后才检查随机浮点差异。把所有失败都归结为“随机性”会让真正的状态错误长期存在。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P040 / L103 | CPU、GPU、JAX 和 mesh 各自负责什么 / ## CPU、GPU、JAX 和 mesh 各自负责什么 | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_environment-P041 / L105 | CPU、GPU、JAX 和 mesh 各自负责什么 / 如果你第一次接触训练环境，可以先把它们分成四个角色。CPU 是负责启动程序、准备数据和调度工作的通用处理器；GPU 有大量适合矩阵乘法的并行计算单元，负责模型的大部分张量计算；JAX 是把 Python 函数转换成可自动… | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_environment-P042 / L107 | CPU、GPU、JAX 和 mesh 各自负责什么 / 例如，&lt;code&gt;hidden&lt;/code&gt; 的形状是 &lt;code&gt;[B, L, d]&lt;/code&gt;，其中 &lt;code&gt;B=2&lt;/code&gt;、&lt;code&gt;L=4&lt;/code&gt;、&lt;code&gt;d=8&lt;/code&gt;。单卡运行时… | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_environment-P043 / L109 | CPU、GPU、JAX 和 mesh 各自负责什么 / JAX 还会把第一次调用分成 tracing、编译和执行。可以把它想成先把菜谱翻译成某种 GPU 能执行的机器指令，再真正做菜。第一次调用可能花 40 秒编译，后续每步只花 2 秒执行。如果把两者混在一起，短实验的吞吐会… | 改写/P0 | F06、F35、T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_environment-P044 / L111 | 本章的完成标准 / ## 本章的完成标准 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-01_environment-P045 / L113 | 本章的完成标准 / 完成环境章需要一个可审计的目录，而不是一条绿色命令： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P046 / L115 | 本章的完成标准 / - 代码 commit 可以从输出中找到； - 设备和依赖版本已经保存； - 样例 token 有明确摘要； - tiny 训练产生有限 loss； - checkpoint 能被读取； - auto-resume 识别… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P047 / L123 | 本章的完成标准 / 下一章把这些文件组织成实验身份。环境解决“在哪里跑”，可复现运行解决“这一次跑的是什么”。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P048 / L125 | 本章的最小 checkpoint 与证据 / ## 本章的最小 checkpoint 与证据 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-01_environment-P049 / L127 | 本章的最小 checkpoint 与证据 / 在已安装依赖的目标节点上，按顺序执行： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P050 / L129 | 本章的最小 checkpoint 与证据 / ~~~bash bash scripts/prepare_sample.sh outputs/sample-tokens.npy JAX_PLATFORMS=cpu bash scripts/run_train.sh \… | 改写/P0 | F05；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-01_environment-P051 / L137 | 本章的最小 checkpoint 与证据 / 如果目标节点有 GPU，把第二条命令换成 &lt;code&gt;bash scripts/run_gpu_train.sh&lt;/code&gt;，并先完成 GPU 检查。成功目录应有 &lt;code&gt;experiment.json&lt;/code… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-01_environment-P052 / L139 | 本章的最小 checkpoint 与证据 / 若命令失败，先检查：&lt;code&gt;outputs/sample-tokens.npy&lt;/code&gt; 是否存在、&lt;code&gt;configs/tiny.json&lt;/code&gt; 是否能解析、&lt;code&gt;environment.j… | 改写/P0 | F06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-01_environment-P053 / L141 | 本章小结 / ## 本章小结 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-01_environment-P054 / L143 | 本章小结 / - 系统层、Python 层和实验层必须分别记录； - 样例 token 先验证词表范围、长度、shift 和 mask； - tracing、编译和稳态执行是不同阶段； - 预期输出和失败诊断与命令同样重要； - 下一… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/02_reproducible_runs.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| zh-02_reproducible_runs-P001 / L1 | 3　可复现实验：把一次运行变成一个有身份的对象 / # 3 可复现实验：把一次运行变成一个有身份的对象 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P002 / L3 | 本章导读：中断一次 tiny run，再证明它真的接上了 / ## 本章导读：中断一次 tiny run，再证明它真的接上了 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P003 / L5 | 本章导读：中断一次 tiny run，再证明它真的接上了 / “能打开 checkpoint”不是恢复实验。我们要证明的是：连续运行和中断后恢复运行，在恢复点之后读取相同的输入、使用相同的 step 和 optimizer state，并在数值容忍范围内得到相同的下一步结果。本章使… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P004 / L7 | 本章导读：中断一次 tiny run，再证明它真的接上了 / 读完本章，你应该能区分 &lt;code&gt;--resume&lt;/code&gt; 与 &lt;code&gt;--initialize-from&lt;/code&gt;，解释 RNG 为什么是实验状态，并从 &lt;code&gt;status.json&lt;/code&gt;… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P005 / L9 | 预备知识和结果表 / ## 预备知识和结果表 | 补写/P1 | T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P006 / L11 | 预备知识和结果表 / 假设 tiny run 在 step 2 保存 checkpoint，连续路径继续到 step 4，中断路径从 step 2 恢复后也继续到 step 4： | 补写/P1 | T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P007 / L13 | 预备知识和结果表 / &#124; 检查项 &#124; 连续路径 &#124; 恢复路径 &#124; 成功判据 &#124; &#124;---&#124;---:&#124;---:&#124;---&#124; &#124; 起点 &#124; step 0 &#124; step 2 &#124; 恢复路径加载同一 checkpoint &#124; &#124; 下一批索引 &#124; 2→3 … | 改写/P0 | F35、F09、T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P008 / L21 | 预备知识和结果表 / 这张表说明“恢复成功”是一个状态比较，而不是文件存在检查。 | 补写/P1 | T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P009 / L23 | “同样的命令”并不代表同样的实验 / ## “同样的命令”并不代表同样的实验 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P010 / L25 | “同样的命令”并不代表同样的实验 / 两台机器执行同一条命令，可能读取不同的数据文件、使用不同默认值、从不同 checkpoint 开始，甚至在不同 step 记录第一条 loss。命令是入口，不是实验身份。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P011 / L27 | “同样的命令”并不代表同样的实验 / 我们把一个运行写成： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P012 / L29 | “同样的命令”并不代表同样的实验 / $$ R=(S,C,I_{\mathrm{data}},E,K,Q_{\mathrm{budget}},O), $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| zh-02_reproducible_runs-P013 / L33 | “同样的命令”并不代表同样的实验 / 其中 \(S\) 是源码 commit，\(C\) 是解析后的配置，\(I_{\mathrm{data}}\) 是数据、tokenizer 和 manifest 的指纹，\(E\) 是环境，\(K\) 是随机与优化状态，… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P014 / L35 | resume 和 initialize-from 是两个动词 / ## resume 和 initialize-from 是两个动词 | 补写/P1 | T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P015 / L37 | resume 和 initialize-from 是两个动词 / `--resume` 表示继续同一个优化过程。它必须恢复： | 补写/P1 | T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P016 / L39 | resume 和 initialize-from 是两个动词 / - 模型参数； - AdamW 的一阶、二阶统计； - 当前 global step； - BatchStream 的 RNG state； - 数据指纹、sequence length、batch size 和 see… | 补写/P1 | T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P017 / L46 | resume 和 initialize-from 是两个动词 / 如果只载入参数，优化器会从零开始，学习率 schedule 也可能回到 warmup。这样得到的是新实验，即使输出目录名字叫 resume。 | 补写/P1 | T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P018 / L48 | resume 和 initialize-from 是两个动词 / `--initialize-from` 表示使用已有参数作为新实验的初始点。新实验应该有新的输出目录和 experiment identity，optimizer、step、数据顺序可以重新建立。用它从 dense ba… | 改写/P0 | F08、T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P019 / L50 | 一个最小的失败案例 / ## 一个最小的失败案例 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P020 / L52 | 一个最小的失败案例 / 假设训练在 global step 100 保存 checkpoint。这个 checkpoint 中的参数已经看过 100 个 batch，AdamW 也积累了对应的动量。若只复制 params 到新目录，再把 ste… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P021 / L54 | 一个最小的失败案例 / 1. 优化器的一阶、二阶统计被清零，更新方向会突然改变； 2. 学习率 schedule 可能从 step 0 开始，或从 step 100 开始但使用了不匹配的 optimizer state； 3. BatchStr… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P022 / L59 | 一个最小的失败案例 / 恢复检查不能只问“文件能否打开”。至少要用同一个初始 checkpoint 做一次短实验：一条路径连续运行，另一条路径在中间保存并 resume，然后比较恢复点之后的下一批输入、step、有效 token 和 loss。… | 改写/P0 | F35；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P023 / L61 | 一个最小的失败案例 / 相反，从 baseline 初始化 TTT 时，应明确产生新的身份： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P024 / L63 | 一个最小的失败案例 / ~~~text baseline: source=S1, data=D1, step=1000, optimizer=O_dense TTT init: source=S2, data=D1, step=0, optim… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| zh-02_reproducible_runs-P025 / L68 | 一个最小的失败案例 / 这两行共享参数文件，但不是同一次运行；报告不能把第二行的 loss 曲线接在第一行后面。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P026 / L70 | 当前 checkpoint 保存什么 / ## 当前 checkpoint 保存什么 | 补写/P1 | T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P027 / L72 | 当前 checkpoint 保存什么 / `ttt/checkpoint.py` 将 Experiment、TrainState 和数据状态组合起来。实验对象包含模型和训练配置；TrainState 包含 params、opt_state 和 step；数据状态… | 补写/P1 | T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P028 / L74 | 当前 checkpoint 保存什么 / 检查点不是只有模型权重。对 TTT 来说，快权重有两种不同角色： | 补写/P1 | T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P029 / L76 | 当前 checkpoint 保存什么 / - 训练时的 fast weights 是一个序列内部的临时状态，外循环更新静态模型； - 服务或流式推理时的 fast weights 是 session 状态，需要随会话保存或回滚。 | 补写/P1 | T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P030 / L79 | 当前 checkpoint 保存什么 / 把 session snapshot 当成训练 checkpoint，会让优化器和数据采样状态缺失；把训练 checkpoint 当成会话 snapshot，又会携带不该暴露给请求的全局状态。两者需要不同的 schema… | 补写/P1 | T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P031 / L81 | 运行目录应能自己解释自己 / ## 运行目录应能自己解释自己 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P032 / L83 | 运行目录应能自己解释自己 / 失败目录也要有固定字段，哪怕进程只完成了环境初始化： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P033 / L85 | 运行目录应能自己解释自己 / &#124; 字段 &#124; 作用 &#124; &#124;---&#124;---&#124; &#124; status.json &#124; running、failed、completed、resumable 等生命周期状态 &#124; &#124; error.json &#124; 异常类型、阶段、最后一条… | 改写/P0 | F06、F35；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P034 / L94 | 运行目录应能自己解释自己 / 这样即使编译阶段失败，也能区分“环境不可用”“配置不合法”和“训练运行后发散”。缺少 status.json 的空目录不能证明作业从未启动，只能说明记录协议没有覆盖初始化失败。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P035 / L96 | 运行目录应能自己解释自己 / 一份可审计的输出目录至少能回答： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P036 / L98 | 运行目录应能自己解释自己 / ~~~text 这次用哪个 commit？ 加载了哪个 config？ 数据文件和 tokenizer 的摘要是什么？ 从哪个 step 开始，消耗了多少有效 token？ 机器、后端和 dtype 是什么？ 最后一次保… | 改写/P0 | F35；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-02_reproducible_runs-P037 / L108 | 运行目录应能自己解释自己 / `experiment.json` 写输入，`environment.json` 写机器，`metrics.jsonl` 写过程，`status.json` 写生命周期，checkpoint 写状态。Markdown 报… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P038 / L110 | 运行目录应能自己解释自己 / 可以把运行信息组织成一个 manifest： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P039 / L112 | 运行目录应能自己解释自己 / ~~~json { "source": {"commit": "&lt;git-sha&gt;"}, "inputs": { "config": "&lt;config-sha256&gt;", "data": "&lt;manifest-sha25… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| zh-02_reproducible_runs-P040 / L125 | 运行目录应能自己解释自己 / 这段结构是教学示意；仓库的实际字段以训练输出为准。重要的是把输入、预算和状态分开。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P041 / L127 | 失败也应成为记录 / ## 失败也应成为记录 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P042 / L129 | 失败也应成为记录 / 作业可能在下载依赖、编译、设备初始化、数据读取、梯度更新或保存时失败。失败目录应该保留： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P043 / L131 | 失败也应成为记录 / - 运行命令和调度作业 ID； - 最后解析成功的配置； - 环境摘要； - 已写入的 metrics； - 异常类型与阶段； - 是否可以安全 resume。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P044 / L138 | 失败也应成为记录 / “没有输出”通常是最难恢复的状态。即使进程在初始化阶段失败，也应由外层脚本创建目录并写出 status。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P045 / L140 | 如何比较两个结果 / ## 如何比较两个结果 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P046 / L142 | 如何比较两个结果 / 比较模型质量时，优先固定有效 token 预算和数据 manifest。比较系统效率时，还要固定设备、精度、batch、序列长度和编译策略。比较流式状态时，固定初始 cache、位置和 session seed。 | 改写/P0 | F35；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P047 / L144 | 如何比较两个结果 / 可以把差异分成三类： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P048 / L146 | 如何比较两个结果 / **定义差异**：词表、模型结构、训练 token 不同。它们构成新实验，不能用“同配置”描述。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P049 / L148 | 如何比较两个结果 / **环境差异**：GPU 型号、驱动或 XLA 版本不同。质量可能仍可比较，但性能必须分开。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P050 / L150 | 如何比较两个结果 / **数值差异**：同一输入下并行归约、低精度或非确定性带来的小差异。需要给出容忍范围，而不是把所有差异都叫失败。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P051 / L152 | 本章的实际复现路径 / ## 本章的实际复现路径 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P052 / L154 | 本章的实际复现路径 / 在远端环境中执行： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P053 / L156 | 本章的实际复现路径 / ~~~bash git rev-parse HEAD bash scripts/record_environment.sh --output outputs/run/environment.json bash scrip… | 改写/P0 | F31；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-02_reproducible_runs-P054 / L166 | 本章的实际复现路径 / 随后检查 step、metrics 和 latest checkpoint。对一个中断实验，先用 resume 继续；对一个新架构实验，复制输入记录并使用 initialize-from。书后面的每个架构消融都应遵循这个… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P055 / L168 | 本章的实际复现路径 / 下一章进入配置。我们要把词表、隐藏维度、MLP、快权重和训练选项写成一份能被代码校验的架构合同。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P056 / L170 | 本章的最小恢复 checkpoint / ## 本章的最小恢复 checkpoint | 补写/P1 | F10、T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P057 / L172 | 本章的最小恢复 checkpoint / 先生成样例数据并启动一个会保存中间状态的 tiny 运行： | 补写/P1 | F10、T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P058 / L174 | 本章的最小恢复 checkpoint / ~~~bash bash scripts/prepare_sample.sh outputs/sample-tokens.npy bash scripts/run_train.sh \ --config configs/… | 改写/P0 | F05、F10、T04；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-02_reproducible_runs-P059 / L183 | 本章的最小恢复 checkpoint / 成功时，检查 &lt;code&gt;outputs/tiny-contract-001-C/status.json&lt;/code&gt;、&lt;code&gt;metrics.jsonl&lt;/code&gt;、&lt;code&gt;latest.json&lt;/code… | 补写/P1 | F10、T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P060 / L185 | 本章小结 / ## 本章小结 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-02_reproducible_runs-P061 / L187 | 本章小结 / - resume 恢复同一个优化过程，initialize-from 创建新的实验身份； - RNG、采样位置、optimizer state 和 global step 都是恢复语义的一部分； - 运行目录要能描述成功… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/03_configuration.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| zh-03_configuration-P001 / L1 | 14　配置：用一份架构合同约束所有张量 / # 14 配置：用一份架构合同约束所有张量 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-03_configuration-P002 / L3 | 本章导读：让每个数字都有来源 / ## 本章导读：让每个数字都有来源 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-03_configuration-P003 / L5 | 本章导读：让每个数字都有来源 / 在 &lt;code&gt;tiny-contract-001&lt;/code&gt; 中，词表大小是 259、隐藏宽度是 16、attention 有 2 个 query head。读者应该能够从这三个数字一路算到 embedding、每个… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-03_configuration-P004 / L7 | 本章导读：让每个数字都有来源 / 读完本章，你应该能加载一份配置，列出解析后的默认值，手算一个 batch 的 shape，并解释为什么 tokenizer 的词表大小会固定分类头最后一维。本章不讨论训练是否有效，下一章才会把这些 shape 接到 lo… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-03_configuration-P005 / L9 | 预备知识和 tiny 目标 / ## 预备知识和 tiny 目标 | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-03_configuration-P006 / L11 | 预备知识和 tiny 目标 / 你只需要知道矩阵可以把一组数字映射成另一组数字。贯穿输入为 &lt;code&gt;input_ids&lt;/code&gt;，形状是 &lt;code&gt;[B,L]=[2,4]&lt;/code&gt;；本章的完成标准是写出它经过 embedding、atte… | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-03_configuration-P007 / L13 | 为什么模型不应该“猜”参数 / ## 为什么模型不应该“猜”参数 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-03_configuration-P008 / L15 | 为什么模型不应该“猜”参数 / 如果模型代码在 forward 里根据张量形状猜 head 数、词表或快权重层位置，实验配置就会变成隐藏状态。一个 JSON 看起来写了 hidden_size，实际模型却在别处覆盖它，最后得到的是无法审计的结构。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-03_configuration-P009 / L17 | 为什么模型不应该“猜”参数 / 配置对象应该是架构合同：它写明所有会影响参数形状、计算图、状态和比较公平性的字段；加载时立即校验；保存时包含补全后的默认值。模型类只消费这份合同。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-03_configuration-P010 / L19 | 从词表到分类头 / ## 从词表到分类头 | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-03_configuration-P011 / L21 | 从词表到分类头 / 设： | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-03_configuration-P012 / L23 | 从词表到分类头 / - \(V\)：词表大小； - \(d\)：hidden size； - \(n_h\)：attention heads； - \(d_{\mathrm{ff}}\)：普通 MLP 中间维度； - \(d'_{\math… | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-03_configuration-P013 / L29 | 从词表到分类头 / 输入 embedding： | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-03_configuration-P014 / L31 | 从词表到分类头 / $$ E\in\mathbb R^{V\times d}. $$ | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-03_configuration-P015 / L35 | 从词表到分类头 / 每个位置的隐藏状态位于 $\mathbb R^d$。SwiGLU 的中间计算可以使用 $d_{\mathrm{ff}}$，但残差输出必须回到 $d$。最后： | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-03_configuration-P016 / L37 | 从词表到分类头 / $$ \mathbf z_{b,t}=\mathbf h_{b,t}W_{\mathrm{out}}, \quad W_{\mathrm{out}}\in\mathbb R^{d\times V}. $$ | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-03_configuration-P017 / L42 | 从词表到分类头 / 当前实现的 `lm_head` 使用 `bias=False`；如果以后显式启用输出偏置，可以在公式中再加 \(\mathbf b_{\mathrm{out}}\in\mathbb R^V\)。如果 tied embed… | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-03_configuration-P018 / L44 | 从词表到分类头 / 这也解释了为什么不能只把 MLP 中间层扩大来“增加词表能力”。中间层的宽度改变计算和表示容量，词表投影的输出空间仍由 tokenizer 决定。 | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-03_configuration-P019 / L46 | 四类字段和它们的边界 / ## 四类字段和它们的边界 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-03_configuration-P020 / L48 | 四类字段和它们的边界 / **模型尺寸字段**包括 vocab_size、hidden_size、intermediate_size、layers、attention heads 和 KV heads。它们决定初始化树的 shape。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-03_configuration-P021 / L50 | 四类字段和它们的边界 / **序列字段**包括 max_position_embeddings、sliding_window、ttt_chunk_size 和 layer_types。它们决定 cache 容量、分块边界和合法位置。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-03_configuration-P022 / L52 | 四类字段和它们的边界 / **自适应字段**包括 use_fast_mlp、ttt_suffix_length、adaptation_target、inner.enabled 和 inner learning rate。它们决定是否有可更新状态，… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-03_configuration-P023 / L54 | 四类字段和它们的边界 / **系统字段**包括 dtype、attention_backend、loss block size、parallelism 和 expert_parallel_size。它们影响运行方式，部分字段也会改变可比较的成本。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-03_configuration-P024 / L56 | 四类字段和它们的边界 / 仓库的 TTTE2EConfig 将这些字段集中在 ttt/models/ttt_e2e/configuration_ttt_e2e.py；训练配置则由 ttt/config.py 和 ttt/training.py 管… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-03_configuration-P025 / L58 | 默认值也属于实验输入 / ## 默认值也属于实验输入 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-03_configuration-P026 / L60 | 默认值也属于实验输入 / `ttt_suffix_length=None` 会根据层数和 use_fast_mlp 推导后缀长度；`num_key_value_heads=None` 默认等于 query heads；`head_dim=None… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-03_configuration-P027 / L62 | 默认值也属于实验输入 / 因此 `to_dict()` 将补全后的字段写入实验记录。配置审计要问两个问题：原始输入是什么，解析后的实际值是什么。迁移到新版本时，默认规则改变也会改变模型；配置文件不变不代表实验不变。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-03_configuration-P028 / L64 | 校验为什么要早 / ## 校验为什么要早 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-03_configuration-P029 / L66 | 校验为什么要早 / 配置类检查正整数、head 整除性、旋转位置编码的偶数维度、层类型数量、suffix 范围、dtype、特殊 token ID 和 attention 后端。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-03_configuration-P030 / L68 | 校验为什么要早 / 例如： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-03_configuration-P031 / L70 | 校验为什么要早 / - hidden_size 不能被 num_attention_heads 整除，无法推导默认 head_dim； - num_attention_heads 不能被 num_key_value_heads 整除，GQA… | 改写/P0 | F11；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-03_configuration-P032 / L76 | 校验为什么要早 / 这些错误在 Python 初始化阶段失败，远比编译十分钟后才发现好。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-03_configuration-P033 / L78 | AutoConfig 与 AutoModel / ## AutoConfig 与 AutoModel | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-03_configuration-P034 / L80 | AutoConfig 与 AutoModel / `AutoConfig.from_pretrained` 读取配置，并兼容早期没有 model_type 的训练配置；`AutoModelForCausalLM` 根据 mode 选择 base、train 或 infe… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-03_configuration-P035 / L82 | AutoConfig 与 AutoModel / 但自动接口也需要边界。它不能从一个 ttt_e2e checkpoint 自动推断你想用 training 还是 inference 语义；调用方必须显式选择。加载参数不等于加载 optimizer 或 session … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-03_configuration-P036 / L84 | 一次配置审计 / ## 一次配置审计 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-03_configuration-P037 / L86 | 一次配置审计 / 审计 tiny 配置时，依次核对： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-03_configuration-P038 / L88 | 一次配置审计 / ~~~text vocab_size = 259 hidden_size = 16 heads = 2, kv_heads = 1 layers = 2 ttt_suffix_length = 1 fast_interm… | 改写/P0 | F04；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-03_configuration-P039 / L100 | 一次配置审计 / 再画出 embedding、每层 attention、MLP/MLP′ 和 lm_head 的 shape。只要其中一个数字没有来源，就说明配置还没有成为真正的实验合同。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-03_configuration-P040 / L102 | 用一个 tiny shape 走完整条路径 / ### 用一个 tiny shape 走完整条路径 | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-03_configuration-P041 / L104 | 用一个 tiny shape 走完整条路径 / 取 \(V=259\)、\(d=16\)、\(d_{\mathrm{ff}}=32\)、batch \(B=2\)、序列长度 \(L=4\)。输入是整数张量 \[ \texttt{input\_ids}\in\{0,\l… | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-03_configuration-P042 / L118 | 用一个 tiny shape 走完整条路径 / 下一章沿着这张 shape 图追踪一个 token，理解 Attention、MLP′、普通 FFN 和残差的先后顺序。 | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-03_configuration-P043 / L120 | 最小配置审计 / ## 最小配置审计 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-03_configuration-P044 / L122 | 最小配置审计 / 在远端节点上，先确认配置可以被解析，再启动训练： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-03_configuration-P045 / L124 | 最小配置审计 / ~~~bash python3 -c "from ttt.config import load_experiment; import json; print(json.dumps(load_experiment('con… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| zh-03_configuration-P046 / L128 | 最小配置审计 / 预期输出中应同时出现 &lt;code&gt;model&lt;/code&gt;、&lt;code&gt;inner&lt;/code&gt;、&lt;code&gt;training&lt;/code&gt; 和 &lt;code&gt;tokenizer&lt;/code&gt;，并包含补全后的派生字段。若出… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-03_configuration-P047 / L130 | 本章小结 / ## 本章小结 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-03_configuration-P048 / L132 | 本章小结 / - 配置合同同时约束参数 shape、状态边界、成本和比较公平性； - \(V\) 决定 embedding 和 logits 的词表轴，\(d\) 决定残差流宽度； - 默认值必须写入解析后的实验记录； - tiny … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/04_forward_and_ttt.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| zh-04_forward_and_ttt-P001 / L1 | 16　前向顺序：一个 token 如何穿过 TTT-E2E / # 16 前向顺序：一个 token 如何穿过 TTT-E2E | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P002 / L3 | 先不要从 kernel 读模型 / ## 先不要从 kernel 读模型 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P003 / L5 | 先不要从 kernel 读模型 / TTT-E2E 看起来包含很多新名词：prefix、suffix、fast weights、inner loss、MLP′、KV cache。最稳妥的读法不是先看 JAX scan 或 grouped GEMM，而是拿一… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P004 / L7 | 先不要从 kernel 读模型 / 模型首先把 token ID 查成 embedding。随后每一层将隐藏状态送进 attention 或 gated delta，再送入 MLP。后段层可能多一条 fast MLP′。最后的 hidden state 经… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P005 / L9 | 先不要从 kernel 读模型 / ~~~text token ids ↓ embedding ↓ attention / gated delta + residual ↓ fast MLP′ + residual （选择的后段层） ↓ static ML… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| zh-04_forward_and_ttt-P006 / L23 | 先不要从 kernel 读模型 / 每个箭头都是形状、mask、cache 和梯度的接口。将 MLP′ 放在普通 MLP 前后，得到的算法并不相同；将更新发生在 token 内、chunk 末尾或下一个 chunk，状态语义也不同。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P007 / L25 | 先不要从 kernel 读模型 / &lt;figure class="book-figure"&gt; &lt;img src="../../assets/figures/ttt-timeline-zh.svg" alt="TTT 中 token chunk、fast w… | 补写/P1 | T20；具体修复、证据及双语问题见上表 | SVG本体与网页未看 |
| zh-04_forward_and_ttt-P008 / L30 | 当前代码的前向顺序 / ## 当前代码的前向顺序 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P009 / L32 | 当前代码的前向顺序 / `TTTE2EForCausalLM._block` 先执行 attention branch。若层类型是 full_attention 或 sliding_attention，则调用 attention；若是 gate… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P010 / L34 | 当前代码的前向顺序 / 若参数中包含 fast_mlp，代码取当前 fast weight 初始化的 MLP′，对 hidden 做 norm、SwiGLU 和可选 post norm，再加回 residual。之后才执行普通 MLP 或 Mo… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P011 / L36 | 当前代码的前向顺序 / 这意味着当前实现的概念顺序是： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P012 / L38 | 当前代码的前向顺序 / ~~~text Attention / Delta → Fast MLP′ → Static MLP or MoE ~~~ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| zh-04_forward_and_ttt-P013 / L44 | 当前代码的前向顺序 / 这与“用 MLP′ 替换原 FFN”不同。普通 FFN 仍然存在，MLP′ 是附加的适应路径。若 fast weights 不存在，配置和参数树也不会凭空生成一条快路径。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P014 / L46 | Prefix 与 suffix 的作用 / ## Prefix 与 suffix 的作用 | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P015 / L48 | Prefix 与 suffix 的作用 / 配置将层分成 prefix 和 suffix。prefix 一次性运行静态层，产生每个位置的 hidden；suffix 使用这些 hidden 和可更新快权重继续计算。 | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P016 / L50 | Prefix 与 suffix 的作用 / 这种划分减少了内循环需要重复的计算，但也带来一个约束：prefix 的 cache 和 hidden 必须与 suffix 的位置、mask 对齐。后缀不是一段独立文本，它接收的是同一序列中对应位置的表示。 | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P017 / L52 | Prefix 与 suffix 的作用 / `sequence_loss` 先把 token 序列按 ttt_chunk_size 补齐并 reshape，再运行 model.prefix。suffix 在 scan 中逐 chunk 处理。快权重从 initia… | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P018 / L54 | 为什么快权重不是普通参数 / ## 为什么快权重不是普通参数 | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P019 / L56 | 为什么快权重不是普通参数 / 普通参数由外循环 AdamW 更新，跨 batch、跨训练 step 保存。快权重在一个序列内由 inner loss 更新，可以在下一个 chunk 变化；是否跨 request 保留，则由 runtime sessi… | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P020 / L58 | 为什么快权重不是普通参数 / 可以用两个时间索引区分： | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P021 / L60 | 为什么快权重不是普通参数 / $$ \theta_{k+1}=\mathrm{AdamW}(\theta_k,\nabla_\theta \mathcal{L}_{\mathrm{outer}}), $$ | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-04_forward_and_ttt-P022 / L64 | 为什么快权重不是普通参数 / $$ \phi_{t+1}=\phi_t-\eta_{\mathrm{inner}} \nabla_\phi \mathcal{L}_{\mathrm{inner}}(x_t;\theta,\phi_t). $$ | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-04_forward_and_ttt-P023 / L69 | 为什么快权重不是普通参数 / $\theta$ 是静态参数，$\phi$ 是快权重。外循环学习的是“怎样初始化和使用 $\phi$”，内循环执行的是“当前上下文如何改变 $\phi$”。如果把 $\phi$ 直接放进 optimizer state，… | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P024 / L71 | 梯度边界和因果性 / ## 梯度边界和因果性 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P025 / L73 | 梯度边界和因果性 / 一个安全的内循环顺序是： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P026 / L75 | 梯度边界和因果性 / ~~~text 使用 fast_t 预测 y_t → 用已观测 label 计算 inner loss → 更新得到 fast_{t+1} → 后续 chunk 使用 fast_{t+1} ~~~ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| zh-04_forward_and_ttt-P027 / L82 | 梯度边界和因果性 / 如果 suffix 的 label 在同一次预测前已经用于更新，模型可以通过标签泄漏获得不应有的信息。因果性测试应修改后半段 token，再比较前半段 logits；前半段不能因为未来标签改变。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P028 / L84 | 梯度边界和因果性 / 当前实现的 inner_step 对 fast_weights 求 value_and_grad，再调用 sgd_update。外循环对 sequence_loss 的输出求导，因此是否保留高阶梯度由这条计算图和 che… | 改写/P0 | F15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P029 / L86 | 一个手算的 chunk 例子 / ## 一个手算的 chunk 例子 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P030 / L88 | 一个手算的 chunk 例子 / 设 sequence length 为 8，chunk size 为 4。下面的 0–7 是代码数组索引；按全书记号，数学位置对应 \(1\)–\(8\)。代码中的输入索引 0–7 被分成两块： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P031 / L90 | 一个手算的 chunk 例子 / ~~~text chunk 0: positions 0,1,2,3 chunk 1: positions 4,5,6,7 ~~~ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| zh-04_forward_and_ttt-P032 / L95 | 一个手算的 chunk 例子 / chunk 0 使用 $\phi_0$ 产生 $\mathcal{L}_0$，并更新成 $\phi_1$。chunk 1 使用 $\phi_1$ 产生 $\mathcal{L}_1$。若最后一个块只有 2 个真实位置，另… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P033 / L97 | 一个手算的 chunk 例子 / 这个例子解释了为什么 chunk size 不是单纯的 batch 参数。它改变了快权重观察上下文的频率，也改变了内循环反向传播和 cache 的边界。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P034 / L99 | 如何验证前向不是“形状正确但语义错误” / ## 如何验证前向不是“形状正确但语义错误” | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P035 / L101 | 如何验证前向不是“形状正确但语义错误” / 至少做四种检查： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P036 / L103 | 如何验证前向不是“形状正确但语义错误” / 1. dense C 路径与 fast MLP 关闭时的输出形状和数值； 2. 修改 suffix token 不影响 prefix logits 的因果性； 3. 完整序列前向与 prefill/decode 的输出在… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P037 / L108 | 如何验证前向不是“形状正确但语义错误” / GPU 验证脚本会生成架构和运行时报告，但报告中的数值必须关联到 config 与初始参数。一次通过不说明所有长度和所有 layer_types 都通过。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P038 / L110 | 如何验证前向不是“形状正确但语义错误” / 下一章把前向中的更新写成内循环与外循环，解释损失如何在两个时间尺度之间流动。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P039 / L112 | 一个 token 的 shape walk / ## 一个 token 的 shape walk | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P040 / L114 | 一个 token 的 shape walk / 用 tiny 配置的 \(d=16\)、\(V=259\)、batch \(B=1\) 和位置数 \(L=4\) 走一遍： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P041 / L116 | 一个 token 的 shape walk / &#124; 阶段 &#124; 代码对象 &#124; 形状 &#124; 读者要检查的事实 &#124; &#124;---&#124;---&#124;---&#124;---&#124; &#124; 输入 &#124; &lt;code&gt;input_ids&lt;/code&gt; &#124; \([1,4]\) &#124; 每个整数在 \([0,259)\) … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P042 / L124 | 一个 token 的 shape walk / 这张表的最后一列比“shape 能对上”更重要：它说明每个接口为什么需要这个形状。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P043 / L126 | 一个可计算的因果性断言 / ## 一个可计算的因果性断言 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P044 / L128 | 一个可计算的因果性断言 / 设 prefix 代码索引为 0、1，suffix 为 2、3。用同一初始参数和 mask 跑两次，只把索引 3 的 token 从 18 改为 19。记录 prefix logits 的最大绝对差： | 改写/P0 | F12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P045 / L130 | 一个可计算的因果性断言 / ~~~text max_abs_diff(logits_a[:, 0:2, :], logits_b[:, 0:2, :]) &lt;= 1e-6 max_abs_diff(logits_a[:, 2:, :], logits… | 改写/P0 | F12；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-04_forward_and_ttt-P046 / L135 | 一个可计算的因果性断言 / 第一行失败说明未来信息泄漏到 prefix；第二行允许变化，因为 suffix 看到了不同输入。这个断言比只检查输出 shape 更能验证 causal mask 和 fast-state 更新边界。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P047 / L137 | 本章最小验证 / ## 本章最小验证 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P048 / L139 | 本章最小验证 / 在远端 GPU 节点运行 &lt;code&gt;bash scripts/validate_gpu.sh outputs/gpu-validation.json&lt;/code&gt;。预期报告至少包含架构摘要、整段与流式 logits 比… | 改写/P0 | F13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P049 / L141 | 本章小结 / ## 本章小结 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-04_forward_and_ttt-P050 / L143 | 本章小结 / - 前向顺序决定每条残差路径看到的输入和状态； - fast weights 是序列/会话状态，不是普通 AdamW 参数； - prefix/suffix、chunk 和 mask 必须使用同一位置坐标； - 因果性要… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/10_data_contract.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| zh-10_data_contract-P001 / L1 | 4　预训练数据：分布、质量与可复现性 / # 4 预训练数据：分布、质量与可复现性 | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P002 / L3 | 为什么数据是基模训练的第一项设计 / ## 为什么数据是基模训练的第一项设计 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P003 / L5 | 为什么数据是基模训练的第一项设计 / 训练一个基模，不是把一批文本交给 GPU 后等待参数自动吸收“互联网知识”。模型只会看到数据管线最终产生的训练窗口；来源选择、过滤、去重、Tokenizer、packing 和 mask 共同决定这些窗口长什么样。因此，… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P004 / L7 | 为什么数据是基模训练的第一项设计 / 数据至少从三个方面改变训练结果： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P005 / L9 | 为什么数据是基模训练的第一项设计 / - **能力覆盖。** 网页、代码、书籍、论文和对话包含的词汇、结构与任务不同。来源混合比例改变模型接触到的知识类型，也会改变长上下文、代码语法和多语言能力的覆盖范围。 - **训练成本。** 同一段文本经过不同 tok… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P006 / L13 | 为什么数据是基模训练的第一项设计 / 下面的简化例子说明为什么“文件数量”不能代表训练分布： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P007 / L15 | 为什么数据是基模训练的第一项设计 / &#124; 来源 &#124; 文档数 &#124; 平均 token 数 &#124; 直接按 token 拼接后的占比 &#124; 可能带来的偏向 &#124; &#124;---&#124;---:&#124;---:&#124;---:&#124;---&#124; &#124; 网页 &#124; 900,000 &#124; 1,000 &#124; 约 82%… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P008 / L21 | 本章导读：先追踪一条短文本 / ## 本章导读：先追踪一条短文本 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P009 / L23 | 本章导读：先追踪一条短文本 / 本章先不讨论大规模下载。我们用短文本“猫坐下。”作为贯穿例子，依次追踪原始记录、清洗后的文档、token 流、训练窗口和 loss。读者读完应能解释一条 token 为什么会进入分母、为什么一个 EOS mask 会改变… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P010 / L25 | 本章导读：先追踪一条短文本 / 假设原始记录有两条：记录 \(r_1\) 的正文是“猫坐下。”，记录 \(r_2\) 的正文是“猫坐下。”的网页镜像。若清洗和去重后只保留 \(r_1\)，均匀采样的概率从 \(P(r_1)=P(r_2)=1/2\) 变… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P011 / L27 | 本章导读：先追踪一条短文本 / 在 byte tokenizer 的教学配置中，字符会先变成整数流；假设为了说明而得到 \(s=(10,11,12,2)\)，其中 2 是 EOS。长度为 3 的窗口变成 \(X=(10,11,12)\)、\(Y=(11… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P012 / L29 | 本章导读：先追踪一条短文本 / 本章的最小检查是运行下面的命令并读取输出目录中的 manifest。它证明的是输入对象可被 loader 识别，不证明清洗质量、污染检测或模型泛化。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P013 / L31 | 本章导读：先追踪一条短文本 / ~~~bash .venv/bin/python -m ttt.data \ --input examples/sample.txt \ --output outputs/book-data-contract \ --s… | 改写/P0 | F01；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-10_data_contract-P014 / L38 | 本章导读：先追踪一条短文本 / 成功时应出现 &lt;code&gt;outputs/book-data-contract/manifest.json&lt;/code&gt; 以及 token、mask 分片；若失败，先检查输入路径、词表大小和输出目录是否可写。下一节开始分… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P015 / L40 | 本章导读：先追踪一条短文本 / 直接拼接会让网页占据大多数训练位置；按来源均匀抽取又会得到另一种分布。两种方案都可以实现，但它们训练的目标不同。若网页中有 30% 是重复模板，清洗后有效 token 下降，代码和书籍的相对权重还会自动上升，除非重新声明… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P016 / L42 | 本章导读：先追踪一条短文本 / &lt;figure class="book-figure"&gt; &lt;img src="../../assets/figures/data-funnel-zh.svg" alt="从原始记录、清洗切分、Tokenizer、pack… | 补写/P1 | T20；具体修复、证据及双语问题见上表 | SVG本体与网页未看 |
| zh-10_data_contract-P017 / L47 | Tokenizer 是什么：把文本变成模型能读的 token / ## Tokenizer 是什么：把文本变成模型能读的 token | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P018 / L49 | Tokenizer 是什么：把文本变成模型能读的 token / 模型不能直接接收 Unicode 文本。Tokenizer 是一个把文本编码成整数序列的规则或模型：它把一段文本 $u$ 映射为 token 序列 $s$，再把每个 token 映射到词表中的一个 ID。这里的 toke… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P019 / L51 | Tokenizer 是什么：把文本变成模型能读的 token / 可以把这条转换链写成 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P020 / L53 | Tokenizer 是什么：把文本变成模型能读的 token / $$ u \xrightarrow{\ \tau\ } s=(s_1,\ldots,s_L), \qquad s_t\in\{0,\ldots,V-1\}, $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| zh-10_data_contract-P021 / L58 | Tokenizer 是什么：把文本变成模型能读的 token / 其中 $\tau$ 是 tokenizer，$V$ 是词表大小，$L$ 是这段文本被编码后的 token 数。模型的 embedding 从 $V$ 行中取出输入表示，最后的分类头也把隐藏向量映射回这 $V$ 个可能的下… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P022 / L60 | Tokenizer 是什么：把文本变成模型能读的 token / Tokenizer 还要定义特殊 token 的语义，例如文档结束的 EOS、序列开始的 BOS、补齐用的 PAD，以及未知字符或字节回退规则。它们会影响文档边界、packing、loss mask 和评测结果，所以 t… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P023 / L62 | 把动机写成可计算的分布 / ## 把动机写成可计算的分布 | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P024 / L64 | 把动机写成可计算的分布 / 下面正式定义两个容易被一句话带过的对象：**训练数据分布**和**预测损失**。只有定义清楚它们，清洗、Tokenizer、packing 和 manifest 才能放在同一条技术链上。按照全书记号，tokenizer … | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P025 / L66 | 把动机写成可计算的分布 / 设原始数据空间中的一条记录为 | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P026 / L68 | 把动机写成可计算的分布 / $$ r_i=(u_i,m_i), $$ | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-10_data_contract-P027 / L72 | 把动机写成可计算的分布 / 其中 $u_i$ 是文本，$m_i$ 是来源、许可证、语言、时间和文档簇等元数据。用 $P_{\mathrm{raw}}$ 表示原始记录的分布。对于一组固定的原始文件，最简单的均匀抽样情况是： | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P028 / L74 | 把动机写成可计算的分布 / 如果均匀抽取 \(N\) 条原始记录，那么每条记录的概率都是 | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P029 / L76 | 把动机写成可计算的分布 / $$ P_{\mathrm{raw}}(r_i)=\frac{1}{N}. $$ | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-10_data_contract-P030 / L80 | 把动机写成可计算的分布 / 例如只有两条记录 \(r_1\) 和 \(r_2\) 时，抽到它们的概率分别是 \(1/2\)。原始文件本身还不是最终的训练分布。我们需要经过来源选择、清洗、去重和切分，得到训练文档集合 | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P031 / L82 | 把动机写成可计算的分布 / $$ C_{\mathrm{train}}=\{r_1,\ldots,r_n\}. $$ | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-10_data_contract-P032 / L86 | 把动机写成可计算的分布 / 如果清洗、切分和来源混合构成变换 $C$，那么文档分布可以写成 | 改写/P0 | F17、T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P033 / L88 | 把动机写成可计算的分布 / $$ P_{\mathrm{doc}}=C_{\#}P_{\mathrm{raw}}, $$ | 改写/P0 | F17、T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-10_data_contract-P034 / L92 | 把动机写成可计算的分布 / 这里的 $\#$ 表示把分布沿着变换推过去。对于固定的文档集合，如果采样器以权重 $w_i$ 选择文档，那么文档层面的经验分布是 | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P035 / L94 | 把动机写成可计算的分布 / $$ P_{\mathrm{doc}}(r_i)= \frac{w_i}{\sum_{j=1}^{n}w_j}. $$ | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-10_data_contract-P036 / L99 | 把动机写成可计算的分布 / 均匀抽取文档、按文档长度抽取、按来源比例抽取，都会产生不同的 $P_{\mathrm{doc}}$。所以“训练集有多少文件”并不能单独说明模型实际看到了什么。 | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P037 / L101 | 把动机写成可计算的分布 / 还要区分文档分布和训练位置分布。即使先均匀抽取文档，较长文档也可能在后续窗口中贡献更多 token；如果改为均匀抽取 token 起点，短文档、长文档和边界位置的权重又会不同。因此文档数、token 总数和真正参与 lo… | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P038 / L103 | 把动机写成可计算的分布 / Tokenizer $\tau$ 把一条文档的文本映射到 token 序列，并把文档分布进一步推到 token 序列分布： | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P039 / L105 | 把动机写成可计算的分布 / $$ \tau(u_i)=s^{(i)}=(s^{(i)}_1,\ldots,s^{(i)}_{L_i}), \qquad s^{(i)}_t\in\{0,\ldots,V-1\}, $$ | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-10_data_contract-P040 / L110 | 把动机写成可计算的分布 / 其中 $V$ 是词表大小。采样器再按照 sequence length、packing、EOS 和文档边界规则，从这些 token 序列中构造训练窗口。记这个过程为 $\Pi$，它诱导出窗口分布： | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P041 / L112 | 把动机写成可计算的分布 / $$ P_{\mathrm{tok}}=\tau_{\#}P_{\mathrm{doc}}, \qquad P_{\mathrm{win}}=\Pi_{\#}P_{\mathrm{tok}}. $$ | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-10_data_contract-P042 / L118 | 把动机写成可计算的分布 / $$ P_{\mathrm{win}}(X,Y,M) = \sum_i P_{\mathrm{doc}}(r_i) \,P_{\Pi}(X,Y,M\mid \tau(u_i)). $$ | 改写/P0 | F18、T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-10_data_contract-P043 / L125 | 把动机写成可计算的分布 / 整条链可以压缩成： | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P044 / L127 | 把动机写成可计算的分布 / $$ P_{\mathrm{raw}} \xrightarrow{\,C\,}P_{\mathrm{doc}} \xrightarrow{\,\tau\,}P_{\mathrm{tok}} \xrightarrow{\,… | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-10_data_contract-P045 / L135 | 把动机写成可计算的分布 / 在实际项目中，$P_{\mathrm{win}}$ 才是更接近“训练数据分布”的对象。它不仅由原始文本决定，还由来源权重、清洗规则、去重规则、Tokenizer、packing、窗口长度和 mask 共同决定。 | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P046 / L137 | 预测损失究竟是什么 / ## 预测损失究竟是什么 | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P047 / L139 | 预测损失究竟是什么 / 设一个训练窗口由连续 token 流 | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P048 / L141 | 预测损失究竟是什么 / $$ s=(s_1,s_2,\ldots,s_{L+1}). $$ | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-10_data_contract-P049 / L145 | 预测损失究竟是什么 / 构造出 \(L\) 个输入位置。因果语言模型把当前位置的 token 作为输入，把下一个 token 作为目标： | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P050 / L147 | 预测损失究竟是什么 / $$ x_t=s_t,\qquad y_t=s_{t+1}, \qquad X=(x_1,\ldots,x_L),\quad Y=(y_1,\ldots,y_L). $$ | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-10_data_contract-P051 / L153 | 预测损失究竟是什么 / 模型参数为 $\theta$ 时，第 $t$ 个位置的预测损失是观察到的目标 token 的负对数概率： | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P052 / L155 | 预测损失究竟是什么 / $$ \ell_t(\theta)=-\log p_\theta(y_t\mid x_{1:t}). $$ | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-10_data_contract-P053 / L159 | 预测损失究竟是什么 / 如果模型在位置 $t$ 输出 logits 向量 $\mathbf{z}_\theta(x_{1:t})$，那么 | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P054 / L161 | 预测损失究竟是什么 / $$ p_\theta(v\mid x_{1:t}) = \operatorname{softmax}(\mathbf{z}_\theta(x_{1:t}))_v. $$ | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-10_data_contract-P055 / L167 | 预测损失究竟是什么 / 如果 $M=(m_1,\ldots,m_L)$ 是 loss mask，$m_t=1$ 表示该位置参与训练，$m_t=0$ 表示忽略，那么一个窗口的平均预测损失为 | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P056 / L169 | 预测损失究竟是什么 / $$ \ell(\theta;X,Y,M) = -\frac{\sum_{t=1}^{L}m_t\log p_\theta(y_t\mid x_{1:t})} {\sum_{t=1}^{L}m_t}. $$ | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-10_data_contract-P057 / L176 | 预测损失究竟是什么 / 模型真正优化的是这个损失在窗口分布上的期望： | 改写/P0 | F16、T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P058 / L178 | 预测损失究竟是什么 / $$ \mathcal{L}(\theta;P_{\mathrm{win}}) = \mathbb{E}_{(X,Y,M)\sim P_{\mathrm{win}}} \left[\ell(\theta;X,Y,M)\r… | 改写/P0 | F16、T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-10_data_contract-P059 / L185 | 预测损失究竟是什么 / 有限训练运行只能看到采样得到的 $N$ 个窗口。若按所有未被 mask 的 token 做平均，经验损失为： | 改写/P0 | F16、T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P060 / L187 | 预测损失究竟是什么 / $$ \widehat{\mathcal{L}}_N(\theta) = \frac{\sum_{k=1}^{N}\sum_{t=1}^{L_k} m^{(k)}_t\ell^{(k)}_t(\theta)} {\sum… | 改写/P0 | F16、T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-10_data_contract-P061 / L195 | 预测损失究竟是什么 / 优化器通过 minibatch 梯度近似最小化这个经验目标。于是，“模型从数据中学习”可以更精确地表述为：模型参数被更新，使采样器产生的窗口上的平均 next-token 负对数似然下降。 | 改写/P0 | F16、T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P062 / L197 | 预测损失究竟是什么 / 验证集损失使用同一个定义，但把 $P_{\mathrm{win}}$ 换成独立的 $P_{\mathrm{val}}$。因此训练损失回答“模型是否适应了训练分布”，验证损失回答“模型在另一份声明过的分布上预测得如何”。如… | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P063 / L199 | 预测损失究竟是什么 / 这也解释了本章的核心判断：我们优化的不是一份抽象的“语言知识”，而是由数据处理和采样协议共同定义的预测目标。改变 $P_{\mathrm{doc}}$、$\tau$、$\Pi$ 或 $M$，即使模型结构和优化器不变，目标… | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P064 / L201 | 一份原始文档如何变成训练信号 / ## 一份原始文档如何变成训练信号 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P065 / L203 | 一份原始文档如何变成训练信号 / 数据管线不是把文件搬到 GPU，而是连续执行一组会改变数学对象的变换： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P066 / L205 | 一份原始文档如何变成训练信号 / ~~~mermaid flowchart LR A[原始文件] --&gt; B[文档记录与元数据] B --&gt; C[清洗、去重、切分] C --&gt; D[Tokenizer: text → token IDs] D --&gt; E… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| zh-10_data_contract-P067 / L216 | 一份原始文档如何变成训练信号 / 用一组仅用于说明的 token ID 表示一条短文档： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P068 / L218 | 一份原始文档如何变成训练信号 / &#124; 对象 &#124; 内容 &#124; &#124;---&#124;---&#124; &#124; 连续 token 流 \(s\) &#124; \((10,11,12,2)\) &#124; &#124; 代码字段 `input_ids`（数学记号 \(X\)） &#124; \((10,11,12)\) … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P069 / L225 | 一份原始文档如何变成训练信号 / 这一个窗口产生的损失是 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P070 / L227 | 一份原始文档如何变成训练信号 / $$ -\frac{1}{3} \left[ \log p_\theta(11\mid 10) +\log p_\theta(12\mid 10,11) +\log p_\theta(2\mid 10,11,12) \r… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| zh-10_data_contract-P071 / L236 | 一份原始文档如何变成训练信号 / 如果最后一个 EOS 位置被 mask 掉，分母和求和项都会改变；如果窗口跨越两篇文档，模型学习到的条件上下文也会改变；如果 tokenizer 把同一段文本拆成更多 token，预测位置数和成本也会改变。训练信号不是原… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P072 / L238 | 每一步改变了什么 / ## 每一步改变了什么 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P073 / L240 | 每一步改变了什么 / **来源选择改变支持范围。** 网页、代码、书籍和论文拥有不同的词汇、长度和结构。模型只能在它实际被采样到的区域内获得训练信号。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P074 / L242 | 每一步改变了什么 / **清洗改变样本集合。** 删除乱码、导航栏、截断页面和异常记录会改变哪些文档保留在 $C_{\mathrm{train}}$ 中。过滤过强又可能删除代码、数学、少数语言或长文档。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P075 / L244 | 每一步改变了什么 / **去重改变样本权重。** 同一篇文档出现一百次，相当于提高它在 $P_{\mathrm{doc}}$ 中的概率。训练集和验证集近重复，还会让验证损失失去独立性。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P076 / L246 | 每一步改变了什么 / **切分改变评估对象。** 去重和按文档簇切分决定验证窗口是否真正代表未见内容。评测集污染会让低损失或高分更接近记忆，而不是泛化。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P077 / L248 | 每一步改变了什么 / **Tokenizer 改变样本空间和成本。** 同一段字符经过不同 tokenizer 会产生不同 token 序列。词表大小同时影响 embedding、LM head、序列长度、attention、KV cache… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P078 / L250 | 每一步改变了什么 / **Packing 和 mask 改变条件概率。** 是否跨文档、是否保留 EOS、哪些位置参与 loss，都会改变 $P_{\mathrm{win}}$ 或窗口损失。不能把 packing 当成只影响存储的实现细节。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P079 / L252 | 一个具体例子：文件大小不是训练分布 / ## 一个具体例子：文件大小不是训练分布 | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P080 / L254 | 一个具体例子：文件大小不是训练分布 / 假设有三个来源： | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P081 / L256 | 一个具体例子：文件大小不是训练分布 / &#124; 来源 &#124; 文档数 &#124; 平均长度 &#124; token 总量 &#124; &#124;---&#124;---:&#124;---:&#124;---:&#124; &#124; 网页 &#124; 900,000 &#124; 1,000 &#124; 900M &#124; &#124; 代码 &#124; 50,000 &#124; 2,000 &#124; 10… | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P082 / L262 | 一个具体例子：文件大小不是训练分布 / 直接拼接会让网页约占 82% 的 token。均匀抽取文档则会提高书籍文档的采样概率。两种方案都可以合理，但它们定义的是不同的 $P_{\mathrm{doc}}$，因此也定义了不同的训练目标。 | 改写/P0 | F19、T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P083 / L264 | 一个具体例子：文件大小不是训练分布 / 如果网页中 30% 是重复模板，清洗后网页有效 token 可能降到 630M；若不重新计算混合比例，代码和书籍的实际权重就会自动上升。每个阶段都必须报告自己的分母，不能只在最后写一个总 token 数。 | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P084 / L266 | 一个具体例子：文件大小不是训练分布 / 当前仓库的 `BatchStream` 从一个逻辑 token 流中随机采样窗口，还没有实现按来源调度的 mixture sampler。后面的 mixture 章节会定义这种能力，但本章不把未来设计写成当前功能。 | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P085 / L268 | 训练前的数据契约 / ## 训练前的数据契约 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P086 / L270 | 训练前的数据契约 / 进入长时间训练前，数据管线至少要回答五个问题： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P087 / L272 | 训练前的数据契约 / 1. **来源是否清楚？** 能否定位原始发布版本、获取时间和许可证？ 2. **质量是否可解释？** 每条过滤规则是否有版本、阈值、统计和抽样审计？ 3. **切分是否可信？** train、validation 和 … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P088 / L278 | 训练前的数据契约 / 数据可以被程序读取，不代表它满足这些条件。每个条件都对应后续章节和一个可保存的证据对象。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P089 / L280 | 数据对象和版本边界 / ## 数据对象和版本边界 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P090 / L282 | 数据对象和版本边界 / **原始文件**是下载或采集得到的不可变输入，例如压缩包、HTML、JSONL 或代码仓库快照。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P091 / L284 | 数据对象和版本边界 / **文档记录**是抽取出的正文和元数据，支持清洗、去重、切分和文档边界管理。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P092 / L286 | 数据对象和版本边界 / **Token 分片**是 tokenizer 生成的整数序列和监督 mask，附带词表、tokenizer 版本、dtype、分片数量和 SHA256。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P093 / L288 | 数据对象和版本边界 / **训练窗口**是采样器从 token 流中取得的一段固定长度序列，进一步变成 `input_ids`、`labels`、`loss_mask` 和位置状态。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P094 / L290 | 数据对象和版本边界 / 这四种对象不能共用一个含糊的“数据版本”。原始文件不变，清洗规则可以产生新的文档版本；文档不变，Tokenizer 可以产生新的 token 版本；分片不变，sequence length 和 packing 也可以产生… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P095 / L292 | 当前仓库已经实现什么 / ## 当前仓库已经实现什么 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P096 / L294 | 当前仓库已经实现什么 / `ttt/data.py` 提供 byte tokenizer 和 JSONL 到 token 文件的转换。普通输入可以写成一维 NPY；sharded 路径会生成 token 分片、mask 分片和 manifest。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P097 / L296 | 当前仓库已经实现什么 / `ttt/dataset.py` 的 `TokenDataset` 会检查 manifest 版本、`vocab_size`、文件路径、SHA256、token 范围、mask shape 和 mask 数值，再用 me… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P098 / L298 | 当前仓库已经实现什么 / ~~~bash .venv/bin/python -m ttt.data \ --input examples/sample.txt \ --output outputs/book-data \ --sharded \ … | 改写/P0 | F01；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-10_data_contract-P099 / L307 | 当前仓库已经实现什么 / `shard-tokens` 是存储分片大小，不是模型一次处理的 sequence length。当前窗口可以跨越 EOS 和文档边界，模型没有通用的“遇到 EOS 就重置 attention”接口，因此应把现有 loa… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P100 / L309 | 当前仓库已经实现什么 / 当前 manifest 也没有完整的 document-offset sidecar，所以任意 token 到原始文档的反查仍是大规模数据管线需要补齐的能力。把已实现行为和设计要求分开，是本书数据章节的基本写法。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P101 / L311 | 后续章节如何推进 / ## 后续章节如何推进 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P102 / L313 | 后续章节如何推进 / - `11_data_acquisition.md`：来源版本、许可证、下载和溯源； - `12_data_cleaning.md`：规范化、格式、语言、长度和质量过滤； - `13_deduplication.md`：… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-10_data_contract-P103 / L324 | 后续章节如何推进 / 从下一章开始，“数据”不再是一个笼统的输入名词。我们会逐步追踪每个对象如何产生、如何改变训练分布，以及如何留下能够被下一次运行复核的证据。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/11_data_acquisition.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| zh-11_data_acquisition-P001 / L1 | 5　获取语料与记录来源 / # 5 获取语料与记录来源 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P002 / L3 | 为什么不从下载命令开始 / ## 为什么不从下载命令开始 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P003 / L5 | 为什么不从下载命令开始 / 假设我们用同一个 URL 下载了两次数据。文件名没变，但发布方补充了一批文档，重新打包并修正了元数据。两次训练都写着“使用 dataset-a”，loss 却不同。没有固定版本和摘要，我们无法判断差异来自模型还是语料。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P004 / L7 | 为什么不从下载命令开始 / 获取数据首先要回答“我们在保存什么对象”。下载地址是位置，发布版本是身份，文件摘要是内容检查。这三者各自解决一个问题，不应把其中一个当成另外两个。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P005 / L9 | 为什么不从下载命令开始 / 本书不在这里推荐一个必须购买或下载的大语料。我们先用仓库内原创教学文本演示接口；正式数据选择应服从目标语言、领域、预算和发布范围。这样读者可以先走通方法，再替换输入，而不是在开篇就受制于大规模存储。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P006 / L11 | 来源清单和文件清单分开 / ## 来源清单和文件清单分开 | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P007 / L13 | 来源清单和文件清单分开 / 来源记录描述发布方和获取方式，文件清单描述这次实际拿到的对象。一个来源可以有多个文件，一个压缩包也可能包含多个子来源。把二者塞进一行字符串，后面按来源统计时就会出现歧义。 | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P008 / L15 | 来源清单和文件清单分开 / 一个来源记录可以这样组织： | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P009 / L17 | 来源清单和文件清单分开 / ~~~json { "source_id": "source-a", "publisher": "&lt;publisher&gt;", "source_url": "&lt;canonical-page&gt;", "release": "&lt;… | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-11_data_acquisition-P010 / L31 | 来源清单和文件清单分开 / 这个结构是治理示例，不是当前 `TokenDataset` 的输入格式。训练 loader 需要的是前一章介绍的 token manifest；来源清单留在数据处理记录中，通过版本关系与它关联。 | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P011 / L33 | 来源清单和文件清单分开 / 对于会更新的远端资源，发布方提供的 commit、revision 或 snapshot 优先于“latest”。ETag 可以辅助缓存判断，但不能不加说明地当成 SHA256；不同服务的 ETag 语义并不相同。 | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P012 / L35 | 下载完成不等于内容可用 / ## 下载完成不等于内容可用 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P013 / L37 | 下载完成不等于内容可用 / 文件完整性检查和文本抽取检查是不同阶段。压缩包 hash 正确，仍可能因为抽取器版本不同而生成不同文本。例如 HTML 抽取器是否保留标题、表格和代码块，会改变模型实际看到的上下文。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P014 / L39 | 下载完成不等于内容可用 / 因此我们保存原始对象而不直接覆盖它。清洗后的文件写到新目录，记录抽取器版本、输入摘要和输出摘要。抽取失败时可以定位原文件和文档，而不需要重新访问可能已经变化的网站。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P015 / L41 | 下载完成不等于内容可用 / 失败重试也应记录清楚。一次下载超时可以重试，但不能在报告中把没下载成功的分片当成“被质量过滤删除”。下载失败、解析失败和内容过滤属于不同损耗，它们会导致不同偏差。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P016 / L43 | 来源、许可与可公开范围 / ## 来源、许可与可公开范围 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P017 / L45 | 来源、许可与可公开范围 / 每个来源需要记录可核对的条款依据，包括数据集页面、附带许可证以及特殊约束。这里的工程要求是保留事实和审核结论，不是用一个模板自动作法律判断。未知条款应保留为未知，而不是为了让流水线继续运行而填上宽松许可。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P018 / L47 | 来源、许可与可公开范围 / 还要区分计划发布的对象：原始文件、清洗文本、token 分片、处理脚本和模型权重并不是同一份产物。一个“可以使用但不能再分发原文”的来源，可能影响复现方式；项目需要解释别人如何在有权限的环境重建相同输入。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P019 / L49 | 来源、许可与可公开范围 / 同样，删除个人信息的规则、对第三方代码的处理和 benchmark 污染检查都要分开描述。许可字段存在，并不证明这些审查已经完成；后面章节也不会用去重代替它们。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P020 / L51 | 用数量把获取阶段讲清楚 / ## 用数量把获取阶段讲清楚 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P021 / L53 | 用数量把获取阶段讲清楚 / 考虑一个纯教学例子：来源列表宣称有 1,000 个文件，我们成功下载 990 个，其中 980 个能解析。解析得到 10 万条记录，有 500 条缺少正文。最终交给清洗阶段的是 99,500 条文档。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P022 / L55 | 用数量把获取阶段讲清楚 / 如果只报告“获得了 99,500 篇文档”，我们看不出 10 个未下载文件和 10 个无法解析文件是否集中在某种语言或格式。一个小比例的损耗，也可能对某个小来源形成很大偏差。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P023 / L57 | 用数量把获取阶段讲清楚 / 来源报告应保留从文件到文档的数量链，并按来源、语言和格式分组。数量对不上时，先修报告，不要继续调模型。否则后面的保留率是在一个已经不完整的分母上计算的。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P024 / L59 | 仓库中的实践边界 / ## 仓库中的实践边界 | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P025 / L61 | 仓库中的实践边界 / 仓库内可直接运行的教学输入是 `examples/sample.txt`；JSONL 语料应写成需要读者替换的 `&lt;prepared-jsonl&gt;`，不能伪装成仓库已有文件。`ttt.data` 只负责把支持的文本字段编… | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P026 / L63 | 仓库中的实践边界 / 因此，替换正式语料时应在 `ttt.data` 之前完成获取、抽取和来源记录。保持训练入口简单是合理的，但上游必须有自己的版本和报告，不能因为训练模块没读那些字段就把它们丢掉。 | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P027 / L65 | 本章最小运行、证据与失败诊断 / ## 本章最小运行、证据与失败诊断 | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P028 / L67 | 本章最小运行、证据与失败诊断 / 先用仓库中真实存在的教学文本跑通编码接口： | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P029 / L69 | 本章最小运行、证据与失败诊断 / ~~~bash python -m ttt.data \ --input examples/sample.txt \ --output outputs/acquisition-sample.npy \ --tokeniz… | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-11_data_acquisition-P030 / L76 | 本章最小运行、证据与失败诊断 / 成功产物是一个一维 token 文件；它只能证明输入可被 tokenizer 读取，不能证明来源许可、正文抽取或数据质量已经通过。若要检查分片和 manifest，再运行： | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P031 / L78 | 本章最小运行、证据与失败诊断 / ~~~bash python -m ttt.data \ --input examples/sample.txt \ --output outputs/acquisition-shards \ --sharded --t… | 改写/P0 | F01、T08；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-11_data_acquisition-P032 / L85 | 本章最小运行、证据与失败诊断 / 预期看到 &lt;code&gt;outputs/acquisition-shards/manifest.json&lt;/code&gt;，其中有 &lt;code&gt;vocab_size&lt;/code&gt;、输入摘要和每个 shard 的 token/m… | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P033 / L87 | 本章最小运行、证据与失败诊断 / 正式 JSONL 输入应写成 &lt;code&gt;&lt;prepared-jsonl&gt;&lt;/code&gt; 占位符，并在执行前替换为实际路径。报告至少保存 source record、每个文件的字节数和 SHA256、下载/解… | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P034 / L89 | 本章小结与下一问 / ## 本章小结与下一问 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P035 / L91 | 本章小结与下一问 / 本章建立了三个区分：URL 是位置、release 是版本、digest 是字节身份；下载失败、解析失败和内容拒绝是不同损耗；原始文件、文档记录、token 分片和模型权重是不同发布对象。下一章在固定输入版本上定义清洗规… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-11_data_acquisition-P036 / L93 | 本章小结与下一问 / 下一章进入清洗。我们将从同一份文本为什么可能被不同规则判成不同质量开始，建立一条可解释的规则链。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/12_data_cleaning.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| zh-12_data_cleaning-P001 / L1 | 6　清洗文本：每一条规则都在改变分布 / # 6 清洗文本：每一条规则都在改变分布 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P002 / L3 | “干净”并不是一个天然标签 / ## “干净”并不是一个天然标签 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P003 / L5 | “干净”并不是一个天然标签 / 下面三段文本，哪一段应该被删除？ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P004 / L7 | “干净”并不是一个天然标签 / ~~~text A. 首页 &#124; 登录 &#124; 相关推荐 &#124; 下一页 B. def normalize(text): return text.strip() C. 证明分为三步。首先设 ε &gt; 0…… ~~~ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| zh-12_data_cleaning-P005 / L14 | “干净”并不是一个天然标签 / A 很像网页模板，B 的标点和短行很多，C 混合数学符号。如果只按非字母比例、句号数量或短行比例过滤，可能连 B 和 C 一起删除。质量规则通常是针对某种语料设计的启发式，移到新领域后需要重新检查。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P006 / L16 | “干净”并不是一个天然标签 / 清洗真正要做的，是降低明显噪声在训练中的权重，同时控制误删。我们不能只看总体保留率，还要看代码、数学、中文、英文和长文档分别发生了什么。[FineWeb](https://arxiv.org/abs/2406.17557… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P007 / L18 | 三种不同的处理 / ## 三种不同的处理 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P008 / L20 | 三种不同的处理 / 第一种是**格式修复**。例如统一换行或删除正文开头的字节顺序标记。修复应该尽量保留内容含义，并允许追踪到原文。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P009 / L22 | 三种不同的处理 / 第二种是**规则过滤**。例如正文为空、解析器失败或长度超出处理上限。它需要明确原因和统计分母。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P010 / L24 | 三种不同的处理 / 第三种是**内容评估**。例如语言识别、质量分类、个人信息与安全策略。它通常存在误判，应该保留分数、阈值和抽样审计，而不只保留最终布尔值。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P011 / L26 | 三种不同的处理 / 把这三种操作拆开有一个实际好处：当语料的保留率突然下降时，我们可以判断是解析器坏了、阈值变了，还是输入来源发生变化。否则一个笼统的 `bad_text` 标签无法帮助排错。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P012 / L28 | 先做保守的规范化 / ## 先做保守的规范化 | 补写/P1 | T09；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P013 / L30 | 先做保守的规范化 / 以 Unicode 为例，NFC 和 NFKC 不是“弱清洗”和“强清洗”的简单关系。NFKC 会合并某些兼容字符，这可能适合搜索或去重 key，却不一定适合保留原始技术文本。代码缩进、全角符号、数学字形都可能受到影响。 | 补写/P1 | T09；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P014 / L32 | 先做保守的规范化 / 因此可以保存原始 `text`，同时构造一个仅用于比较的规范化视图。模型训练使用哪个版本，需要另行声明。不要为了方便 hash，就把清洗视图不可逆地覆盖到原文上。 | 补写/P1 | T09；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P015 / L34 | 先做保守的规范化 / 下面是保守教学函数，展示决策如何返回原因；它不是仓库已经集成的生产清洗命令： | 补写/P1 | T09；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P016 / L36 | 先做保守的规范化 / ~~~python import unicodedata def inspect_document(record, *, min_chars=1, max_chars=100_000): text = record.ge… | 补写/P1 | T09；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-12_data_cleaning-P017 / L58 | 先做保守的规范化 / 代码只有几个分支，但它表达了重要约定：类型错误不等于空文本；修复发生在长度统计前；每条记录产生一个主状态。若以后添加多标签诊断，还应区分“第一个拒绝原因”和“命中的所有规则”，以免计数重复。 | 补写/P1 | T09；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P018 / L60 | 规则顺序会改变报告 / ## 规则顺序会改变报告 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P019 / L62 | 规则顺序会改变报告 / 假设一篇空字符串同时满足“太短”和“无字母”。如果按空文本检查优先，它记为 `empty`；若先看字母比例，可能记为 `bad_character_ratio`。两种流水线删除相同文本，却生成不同原因分布。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P020 / L64 | 规则顺序会改变报告 / 为使统计能对账，可以让主状态互斥： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P021 / L66 | 规则顺序会改变报告 / $$ n_{\mathrm{input}}= n_{\mathrm{kept}}+\sum_r n_{\mathrm{rejected},r}. $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| zh-12_data_cleaning-P022 / L71 | 规则顺序会改变报告 / 所有数必须来自同一阶段和同一计数单位。若一份 HTML 被拆成三段，输入文档数与输出段落数不能直接放进这个公式；要另记父文档和子段关系。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P023 / L73 | 规则顺序会改变报告 / **规则的条件保留率**也要说明分母。例如长度规则删掉前置检查后剩余记录的 10%，不等于删掉全部原始记录的 10%。报告既可列绝对数，也可列相对前一阶段和相对总输入的比例。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P024 / L75 | 如何发现误删 / ## 如何发现误删 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P025 / L77 | 如何发现误删 / 只抽样查看被保留文本，会漏掉最重要的问题：好内容是否被删了。一个基本审计同时取四组样本：随机保留、随机拒绝、接近阈值、稀有来源。抽样 ID、seed 和规则版本应留档，避免每次挑到方便解释的样本。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P026 / L79 | 如何发现误删 / 对于语言判别，短文和混合语言往往更难。把中文代码注释判为“非英文”，只能说明它不符合英文过滤器的目标，不能说明它低质量。因此质量标签必须和数据目标联系起来。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P027 / L81 | 如何发现误删 / 还应检查长度分布的变化。清洗后平均长度增加，可能因为短模板被删除，也可能因为所有简短定义都被误删。一个平均值无法区分这两种情况，应查看分位数和具体样本。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P028 / L83 | 清洗报告怎样读 / ## 清洗报告怎样读 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P029 / L85 | 清洗报告怎样读 / 下面是纯演算示例： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P030 / L87 | 清洗报告怎样读 / &#124; 阶段 &#124; 文档数 &#124; 占原始比例 &#124; &#124;---&#124;---:&#124;---:&#124; &#124; 输入 &#124; 10,000 &#124; 100% &#124; &#124; schema 通过 &#124; 9,900 &#124; 99% &#124; &#124; 正文非空 &#124; 9,600 &#124; 96% &#124;… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P031 / L95 | 清洗报告怎样读 / 我们还不知道这份清洗是否成功。需要继续查看每个来源的保留率、token 数损耗和人工检查结果。若代码来源只保留 20%，总量 87% 的表面稳定可能掩盖严重偏差。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P032 / L97 | 清洗报告怎样读 / 这些数字不来自仓库实测。教学报告必须标明演算与实测的区别，正式结果则链接原始统计文件。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P033 / L99 | 当前代码做到哪一步 / ## 当前代码做到哪一步 | 补写/P1 | T09；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P034 / L101 | 当前代码做到哪一步 / `ttt.data` 对字段类型和空白文本做了有限检查，但没有集成上面的完整质量流水线，也不会自动生成语言或 PII 报告。我们把清洗正文写在这里，是为建立可实现的上游规范，而不是声称命令已经包含这些能力。 | 补写/P1 | T09；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P035 / L103 | 当前代码做到哪一步 / 清洗版本产出的文档应写入新目录。下一章会在这个稳定视图上识别重复；如果每次去重前的规范化都不同，重复率本身就失去了可比性。 | 补写/P1 | T09；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P036 / L105 | 本章最小检查、预期产物与失败诊断 / ## 本章最小检查、预期产物与失败诊断 | 补写/P1 | T09；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P037 / L107 | 本章最小检查、预期产物与失败诊断 / ~~~bash python -m ttt.data \ --input examples/sample.txt \ --output outputs/cleaning-sample \ --sharded --toke… | 改写/P0 | F01、T09；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-12_data_cleaning-P038 / L114 | 本章最小检查、预期产物与失败诊断 / 这条命令只检查“清洗后的文本能否进入 tokenizer”；它不会替代上游的语言、PII、长度或质量报告。成功时应有 &lt;code&gt;outputs/cleaning-sample/manifest.json&lt;/code&gt;；… | 补写/P1 | T09；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P039 / L116 | 本章小结与下一问 / ## 本章小结与下一问 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-12_data_cleaning-P040 / L118 | 本章小结与下一问 / 清洗规则改变的是训练分布，不是一个抽象的“干净”标签。每条规则都要记录版本、阈值、保留/拒绝计数和抽样审计。下一章将在固定清洗版本上计算 exact 与 near duplicate，避免把规范化差异误认为重复差异。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/13_deduplication.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| zh-13_deduplication-P001 / L1 | 7　去重：控制同一段信息出现多少次 / # 7 去重：控制同一段信息出现多少次 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P002 / L3 | 重复实际上是一种隐式采样权重 / ## 重复实际上是一种隐式采样权重 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P003 / L5 | 重复实际上是一种隐式采样权重 / 设语料中一篇文档原本出现一次，另一个模板出现一千次。即使采样器均匀抽 token，模板也会在优化中获得更大的权重。我们不是只浪费了磁盘空间，还改变了目标分布。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P004 / L7 | 重复实际上是一种隐式采样权重 / 重复还会穿过评估边界。训练集有某篇文章，验证集包含它的镜像，模型在验证时可能只是在预测见过的片段。去重研究已经把训练重复与评估重叠作为具体问题加以测量；这里参考[原始去重论文](https://arxiv.org/abs… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P005 / L9 | Exact dedup 从什么内容计算 hash / ## Exact dedup 从什么内容计算 hash | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P006 / L11 | Exact dedup 从什么内容计算 hash / 最简单的方法是对文本编码后计算 SHA256。两个完全相同的字节序列得到相同摘要，于是可以保留一个代表。但“比较什么字节”需要先定义：原始 HTML、抽取文本、NFC 后的文本，还是去掉所有空白的文本？ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P007 / L13 | Exact dedup 从什么内容计算 hash / 不同定义对应不同去重范围。去空白可能合并仅排版不同的散文，却也可能把有意义的代码缩进抹掉。所以可以为去重单独生成 comparison key，并保留原始训练文本。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P008 / L15 | Exact dedup 从什么内容计算 hash / ~~~python import hashlib import unicodedata def exact_key(text): normalized = unicodedata.normalize("NFC", tex… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| zh-13_deduplication-P009 / L25 | Exact dedup 从什么内容计算 hash / 这个教学函数只定义 exact key，不负责来源合并、磁盘索引或大规模并行。正式实现不能假设一个包含所有全文的 Python 字典能放进内存。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P010 / L27 | 为什么还需要近似去重 / ## 为什么还需要近似去重 | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P011 / L29 | 为什么还需要近似去重 / 假设两篇文档只差页眉和日期，全文 hash 会不同。近似方法先把文本变成集合或指纹，再寻找相似候选。 | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P012 / L31 | 为什么还需要近似去重 / 一种直观做法是取连续的词或字符 n-gram，形成 shingle 集合 \(A\) 与 \(B\)，用 Jaccard 衡量重叠： | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P013 / L33 | 为什么还需要近似去重 / $$ J(A,B)=\frac{&#124;A\cap B&#124;}{&#124;A\cup B&#124;}. $$ | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-13_deduplication-P014 / L37 | 为什么还需要近似去重 / 例如 \(A=\{\mathrm{ab},\mathrm{bc},\mathrm{cd}\}\)，\(B=\{\mathrm{ab},\mathrm{bc},\mathrm{ce}\}\)，交集有 2 个、并集有 4 个… | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P015 / L39 | 为什么还需要近似去重 / MinHash 和 LSH 用来缩小候选检索成本；候选命中不是最终真值。参数、seed、shingle 规则和最后确认方式都要保存，阈值也应在有代表性的样本上检查。 | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P016 / L41 | 相似度不具有传递性 / ## 相似度不具有传递性 | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P017 / L43 | 相似度不具有传递性 / 一个容易忽视的例子是：A 与 B 达到阈值，B 与 C 达到阈值，但 A 与 C 未必达到阈值。若用连通分量形成簇，三个文档仍会被分到一起。 | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P018 / L45 | 相似度不具有传递性 / 这不是实现 bug，而是聚类规则的结果。我们需要决定保留哪个代表：最长文本、发布时间最早、元数据最完整，还是来源优先级最高。规则必须稳定，不能随文件遍历顺序改变。 | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P019 / L47 | 相似度不具有传递性 / 保留代表时还应输出重复映射：被删除 ID、保留 ID、关系类型、相似度与理由。否则某篇文档被删后，无法解释它是垃圾内容还是有效副本。 | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P020 / L49 | 全局去重与局部去重的区别 / ## 全局去重与局部去重的区别 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P021 / L51 | 全局去重与局部去重的区别 / 每个来源内部都去重，不代表合并以后没有重复。网页、书籍节选和教程可能跨来源重叠。相反，直接全局去重又可能把一个来源的覆盖量全部归到另一个来源。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P022 / L53 | 全局去重与局部去重的区别 / 所以报告至少分为来源内、来源间、训练与验证之间三类。长文档还可能只共享若干段落，文档级指标不能说明重复子串的强度；是否进一步做段落去重，应作为新的规则和版本。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P023 / L55 | 全局去重与局部去重的区别 / 阈值降低通常会合并更多候选，但“删得更多”不等于“数据更好”。去重后要检查长度、领域和语言分布，以及稀有内容是否被一个模板化簇吞掉。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P024 / L57 | 与切分和当前代码的关系 / ## 与切分和当前代码的关系 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P025 / L59 | 与切分和当前代码的关系 / 推荐先确定重复簇，再按簇切分，避免同簇成员分到训练和验证。对评测基准的污染检测另做一次，因为 benchmark 与训练数据不一定采用同一单位。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P026 / L61 | 与切分和当前代码的关系 / 当前 `ttt/data.py` 和 `ttt/dataset.py` 不包含 MinHash 或语义去重器。数据准备命令会保留输入顺序，任何正式去重结果都应由上游显式产出，并和 token manifest 关联。小型… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P027 / L63 | 与切分和当前代码的关系 / 下一章把这些簇变成评估边界。我们要先定义什么叫“未见”，再用验证 loss 判断模型。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P028 / L65 | 本章最小检查、预期报告与失败诊断 / ## 本章最小检查、预期报告与失败诊断 | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P029 / L67 | 本章最小检查、预期报告与失败诊断 / 正式去重会先生成一个新的 JSONL 和报告，再进入 tokenizer： | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P030 / L69 | 本章最小检查、预期报告与失败诊断 / ~~~bash python -m ttt.data \ --input &lt;deduped-jsonl&gt; \ --output outputs/dedup-v1 \ --sharded --tokenizer byte … | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-13_deduplication-P031 / L76 | 本章最小检查、预期报告与失败诊断 / &lt;code&gt;&lt;deduped-jsonl&gt;&lt;/code&gt; 是需要替换的上游产物，不是仓库中的现成文件。报告至少应给出 &lt;code&gt;kept=890&lt;/code&gt;、&lt;code&gt;exact_removed=80&lt;… | 改写/P0 | F21、T10；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P032 / L78 | 本章小结与下一问 / ## 本章小结与下一问 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-13_deduplication-P033 / L80 | 本章小结与下一问 / Exact hash 只回答字节是否相同，MinHash/LSH 只负责缩小候选，人工或明确的合并规则才决定是否删除。去重簇必须在切分前固定，并与 token manifest 关联。下一章把簇变成 train/vali… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/14_split_and_contamination.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| zh-14_split_and_contamination-P001 / L1 | 8　划分训练集，并定义什么叫“未见数据” / # 8 划分训练集，并定义什么叫“未见数据” | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P002 / L3 | 随机切分为什么不总是足够 / ## 随机切分为什么不总是足够 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P003 / L5 | 随机切分为什么不总是足够 / 若一套语料包含同一教程的十个镜像，把每个文档独立随机分到训练和验证，几乎可以肯定两边都会出现相关文本。文件名不重叠，不等于内容没有重叠。 | 改写/P0 | F20；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P004 / L7 | 随机切分为什么不总是足够 / 先问评估目标。如果我们希望预测同类来源的新文档，可以按文档簇切分；如果希望检验未来新闻，按时间切分更有意义；如果希望检验未见代码项目，则需要项目级边界。没有一种切分能同时回答所有问题。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P005 / L9 | 随机切分为什么不总是足够 / 因此“验证集大小 1%”只是比例，完整协议还包括单位、时间、来源和关联规则。一个比另一个更低的验证 loss，只有在评估分布明确时才有解释。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P006 / L11 | 把相关样本留在一起 / ## 把相关样本留在一起 | 补写/P1 | T11；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P007 / L13 | 把相关样本留在一起 / 用 `split_group` 表示共同归属，例如重复簇 ID、书籍 ID 或代码仓库 ID。稳定 hash 可以让新增文档不改变已有组的切分： | 补写/P1 | T11；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P008 / L15 | 把相关样本留在一起 / ~~~python import hashlib def choose_split(group_id): digest = hashlib.sha256(group_id.encode("utf-8")).digest(… | 补写/P1 | T11；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-14_split_and_contamination-P009 / L28 | 把相关样本留在一起 / 这个示例给每组固定归属，期望比例约为 1%、2%、97%，但不保证 token 比例精确相等。一组超长书籍可能让比例明显偏移。`hash()` 不适合代替这里的 SHA256，因为进程间的默认 hash 行为可能不同。 | 补写/P1 | T11；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P010 / L30 | 把相关样本留在一起 / hash 切分的稳定性依赖 group ID 稳定。若先用文件遍历序号生成 ID，改变并行度就可能改变分组；这时稳定 hash 只是把不稳定输入变成稳定-looking 输出。 | 补写/P1 | T11；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P011 / L32 | 为什么应在 tokenization 和 packing 前切分 / ## 为什么应在 tokenization 和 packing 前切分 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P012 / L34 | 为什么应在 tokenization 和 packing 前切分 / 如果把所有文档拼成 token 流再抽 1% 给验证，相邻窗口可能共享大量文本。长度 1,024 的两个窗口只偏移一个 token，就会有 1,023 token 重叠。这样的验证显然很难代表未见数据。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P013 / L36 | 为什么应在 tokenization 和 packing 前切分 / 更清楚的顺序是：文档归组，按组切分，分别编码，再生成各自 manifest。物理 shard 可以切开文档，但不同集合的输入不能先混合后分片。训练的随机窗口也不能跨越 train 与 validation 的逻辑流。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P014 / L38 | 为什么应在 tokenization 和 packing 前切分 / 当前 loader 可以读取一个 manifest 中的连续 token，但并不会替我们创建训练/验证集合。这个边界必须在准备输入时建立。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P015 / L40 | 污染检测是另一层比较 / ## 污染检测是另一层比较 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P016 / L42 | 污染检测是另一层比较 / 去重寻找训练语料内部的重复，污染检测比较训练输入与评估对象。可以用全文 exact match、长子串、规范化题干或近似匹配产生候选，再结合任务单位审核。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P017 / L44 | 污染检测是另一层比较 / 例如一道数学题改了人名和数字，字节 hash 无法识别；一段常见定义在许多地方出现，也不必然说明完整题目泄漏。我们需要记录匹配方式和匹配范围，而不是把所有候选一律解释成污染。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P018 / L46 | 污染检测是另一层比较 / 评估协议也包含调参接触。反复查看某个集合并选择模型，会让它逐渐承担开发集角色。保留一个冻结测试集，就是为最终结论留出没有反复参与选择的证据。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P019 / L48 | 一个切分报告怎样暴露问题 / ## 一个切分报告怎样暴露问题 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P020 / L50 | 一个切分报告怎样暴露问题 / 想象验证集只占文档的 2%，却占 token 的 18%。这可能是长文档簇集中落入验证，也可能是错误的分组。先查看长度分位数和来源表，再决定是否调整。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P021 / L52 | 一个切分报告怎样暴露问题 / 另一个例子是训练与验证 exact overlap 为零，但同域名、同模板的文档很多。这个报告仍不足以声称“无污染”。应说明已经检查的范围，以及未覆盖的近似或语义关系。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P022 / L54 | 一个切分报告怎样暴露问题 / 正式报告至少列：每组文档数与 token 数、语言和来源分布、组间交集、匹配候选数量、人工审核结论、规则版本与输入摘要。它提供的是有边界的证据，不是对所有未知评测的绝对保证。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P023 / L56 | 走向下一章 / ## 走向下一章 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P024 / L58 | 走向下一章 / 教学小语料主要用于管线检查，不值得据其验证 loss 讨论泛化能力。正式语料应建立独立 manifest 和冻结评估输入，再比较模型。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P025 / L60 | 走向下一章 / 切分固定了模型可以接触哪些文档。下一章讨论同一篇文档如何变成不同长度的整数序列，以及这种变化为什么会改变参数和计算成本。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P026 / L62 | 本章最小检查、预期证据与失败诊断 / ## 本章最小检查、预期证据与失败诊断 | 补写/P1 | T11；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P027 / L64 | 本章最小检查、预期证据与失败诊断 / 切分后的两个 JSONL 必须分别生成 manifest： | 补写/P1 | T11；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P028 / L66 | 本章最小检查、预期证据与失败诊断 / ~~~bash python -m ttt.data \ --input &lt;train-jsonl&gt; \ --output outputs/split-train \ --sharded --tokenizer byte… | 补写/P1 | T11；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-14_split_and_contamination-P029 / L77 | 本章最小检查、预期证据与失败诊断 / 预期产物是两个独立的 &lt;code&gt;manifest.json&lt;/code&gt; 和一份 split/contamination 报告；报告应列出组数、文档数、token 数、来源分布、候选匹配和人工审核结果。若 train … | 补写/P1 | T11；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P030 / L79 | 本章小结与下一问 / ## 本章小结与下一问 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-14_split_and_contamination-P031 / L81 | 本章小结与下一问 / 切分单位必须与评估问题一致：文档簇、项目、时间或任务对象。Exact overlap 为零只排除了最窄的一类重复，不能声称没有近似或语义污染。下一章固定每个 split 的 tokenizer 和词表，进入 token … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/15_tokenizer.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| zh-15_tokenizer-P001 / L1 | 9　Tokenizer：从文本到固定维度的词表 / # 9 Tokenizer：从文本到固定维度的词表 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P002 / L3 | Tokenizer 不只是预处理脚本 / ## Tokenizer 不只是预处理脚本 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P003 / L5 | Tokenizer 不只是预处理脚本 / 模型接收的是整数序列，不是 Unicode 字符。Tokenizer 决定一段文字被切成多少个位置、哪些字符串拥有独立 ID、未知字符如何处理、EOS 如何表示。它因此同时影响上下文容量、embedding 参数、输出头… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P004 / L7 | Tokenizer 不只是预处理脚本 / 例如相同的 4,000 字符文本，如果 tokenizer A 产生 1,000 token、tokenizer B 产生 2,000 token，那么在相同的 sequence length 下，两个模型看到的字符上下… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P005 / L9 | Tokenizer 不只是预处理脚本 / 一段文本在进入模型前至少经过两步：先由 tokenizer 按固定规则切分，再把每个片段映射成词表中的整数 ID。记文本为 $u$，tokenizer 为 $\tau$，则可以写成 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P006 / L11 | Tokenizer 不只是预处理脚本 / $$ \tau(u)=s=(s_1,\ldots,s_L), \qquad s_t\in\{0,\ldots,V-1\}. $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| zh-15_tokenizer-P007 / L16 | Tokenizer 不只是预处理脚本 / 这里的 token 不一定是完整单词，也可能是字节、字符或子词；ID 只是索引，数值大小规律不表示语义相似度。Tokenizer 因此同时定义了模型看到的序列长度、可表示的字符串集合和特殊 token 的边界语义。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P008 / L18 | Tokenizer 和词表是什么关系 / ## Tokenizer 和词表是什么关系 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P009 / L20 | Tokenizer 和词表是什么关系 / **词表**是 tokenizer 产生的一个组成部分。它记录哪些 token 可以使用，以及每个 token 对应哪个整数 ID。例如，词表可能包含： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P010 / L22 | Tokenizer 和词表是什么关系 / ~~~text "low" → 1523 "er" → 4987 "&lt;eos&gt;" → 2 ~~~ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| zh-15_tokenizer-P011 / L28 | Tokenizer 和词表是什么关系 / **Tokenizer** 是更完整的系统，至少包括： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P012 / L30 | Tokenizer 和词表是什么关系 / - normalizer：如何处理 Unicode、大小写和空格； - pre-tokenizer：如何确定初始切分边界； - vocabulary：token 到 ID 的映射； - segmentation rule… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P013 / L37 | Tokenizer 和词表是什么关系 / 因此，词表不是先独立训练好、再交给 tokenizer 使用的对象。训练 BPE 时，词表和 merge rules 通常一起学习；训练 Unigram 时，词表和子词概率一起产生。只有 Byte tokenizer 这… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P014 / L39 | Tokenizer 和词表是什么关系 / 还要区分 tokenizer 训练和语言模型训练： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P015 / L41 | Tokenizer 和词表是什么关系 / ~~~text 文本语料 → tokenizer 训练 → vocabulary + 切分规则 → 固定 tokenizer → token IDs → Transformer 训练 embedding 和模型参数 ~~… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| zh-15_tokenizer-P016 / L50 | Tokenizer 和词表是什么关系 / Transformer 训练的是 embedding、attention、MLP 等神经网络参数；它不会在预训练过程中自动改变词表。更换词表或切分规则，就等于改变了模型输入空间，通常需要重新生成 token 分片并重新初… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P017 / L52 | 词表大小怎样进入参数量 / ## 词表大小怎样进入参数量 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P018 / L54 | 词表大小怎样进入参数量 / 设词表为 \(V\)，隐藏维度为 \(d\)。输入 embedding 是 \(V\times d\)，参数量为 \(Vd\)。如果不共享输出权重，`lm_head` 还需要一个 \(d\times V\) 的矩阵；如果… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P019 / L56 | 词表大小怎样进入参数量 / 这里的 **tied embedding** 指输入和输出共享参数。输入时，token ID 通过 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P020 / L58 | 词表大小怎样进入参数量 / $$ E\in\mathbb{R}^{V\times d} $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| zh-15_tokenizer-P021 / L62 | 词表大小怎样进入参数量 / 查表得到隐藏向量；输出时，隐藏状态 \(h\in\mathbb{R}^{d}\) 直接使用同一个矩阵的转置计算词表 logits： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P022 / L64 | 词表大小怎样进入参数量 / $$ z=hE^{\mathsf{T}}\in\mathbb{R}^{V}. $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| zh-15_tokenizer-P023 / L68 | 词表大小怎样进入参数量 / 如果不共享，则需要独立的输出矩阵 \(W_{\mathrm{out}}\in\mathbb{R}^{d\times V}\)，并计算 \(z=hW_{\mathrm{out}}\)。因此 tied embedding 少… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P024 / L70 | 词表大小怎样进入参数量 / 以 \(V=259\)、\(d=16\) 的 tiny 配置为例，embedding 只有 4,144 个参数；不共享时输出头再增加同样数量。若把词表扩大到 32,000，单个 embedding 就有 512,000 … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P025 / L72 | 词表大小怎样进入参数量 / 这解释了一个常见疑问：选定 tokenizer 后，MLP 后的分类头是否固定？MLP 的输出接口必须是 \(d\)，因为它要回到 residual stream；随后分类头把 \(d\) 映射到 \(V\)。因此最后 … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P026 / L74 | 子词 tokenizer 是怎样训练出来的 / ## 子词 tokenizer 是怎样训练出来的 | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P027 / L76 | 子词 tokenizer 是怎样训练出来的 / Tokenizer 的训练不是训练 Transformer，也不是让模型通过反向传播学习参数。它通常是在一份代表性文本上统计片段，最后得到一套固定的词表、切分规则和 token ID。模型预训练开始以后，Tokenize… | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P028 / L78 | 子词 tokenizer 是怎样训练出来的 / 训练前必须先固定： | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P029 / L80 | 子词 tokenizer 是怎样训练出来的 / - tokenizer 训练语料的 manifest 和版本； - Unicode 规范化、空格处理和预分词规则； - 目标词表大小 \(V\)； - BPE、WordPiece 或 Unigram 等算法； - BOS… | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P030 / L87 | 子词 tokenizer 是怎样训练出来的 / 以 BPE 为例，训练过程可以理解为： | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P031 / L89 | 子词 tokenizer 是怎样训练出来的 / 1. 选择基础符号。它们可以是字符，也可以是 UTF-8 byte； 2. 统计训练语料中相邻符号对的频率； 3. 把最高频的符号对合并成一个新 token； 4. 用新的符号重新统计相邻组合； 5. 重复合并，直到达到… | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P032 / L95 | 子词 tokenizer 是怎样训练出来的 / 例如，某些文本中经常出现相邻的 `l` 和 `o`，算法可能先学习 `lo`，之后再把 `lo` 和 `w` 合并成 `low`。训练结束后，Tokenizer 保存的是词表和合并顺序；编码新文本时，按照这套规则把文本切… | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P033 / L97 | 子词 tokenizer 是怎样训练出来的 / WordPiece 和 BPE 都学习子词，但合并标准不同。WordPiece 通常使用基于似然的训练准则，许多实现编码时采用贪心的最长匹配；BERT 使用的就是这类方法。最长匹配是常见实现行为，不应当被当成 WordP… | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P034 / L99 | 子词 tokenizer 是怎样训练出来的 / Unigram 则先建立较大的候选子词集合，再逐步删除贡献较小的候选。对同一段文本，它可以评估多种切分方式并选择概率较高的方案。SentencePiece 是支持 BPE 和 Unigram 的工具框架，而不是与它们完全… | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P035 / L101 | 子词 tokenizer 是怎样训练出来的 / Tokenizer 训练结束后，至少要保存： | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P036 / L103 | 子词 tokenizer 是怎样训练出来的 / - vocabulary 和 token-to-ID 映射； - BPE merge rules 或 Unigram 概率； - normalizer 和 pre-tokenizer 配置； - special toke… | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P037 / L109 | 子词 tokenizer 是怎样训练出来的 / 评估不能只看词表是否成功生成。还要在独立的代表性语料上测量 tokens/byte、tokens/document、中文和英文的长度差异、代码片段的切分、unknown 或 fallback 比例，以及 EOS 和其他特… | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P038 / L111 | 实际如何训练一个 tokenizer / ## 实际如何训练一个 tokenizer | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P039 / L113 | 实际如何训练一个 tokenizer / 实际训练 tokenizer 时，不需要先选择一个 Transformer。训练对象是 BPE 的合并规则、Unigram 的子词概率或 WordPiece 的词表准则。Transformer 只在 tokenizer … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P040 / L115 | 准备训练语料 / ### 准备训练语料 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P041 / L117 | 准备训练语料 / Tokenizer 训练语料应来自预训练数据的 `train` 部分，并使用与正式数据相同的 Unicode 规范化和文本抽取规则。若原始数据是 JSONL，先抽取 `text` 字段，生成每行一篇文档的纯文本文件；不要… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P042 / L119 | 准备训练语料 / 训练前要写清楚： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P043 / L121 | 准备训练语料 / - 语料 manifest 和版本； - 文档边界以及换行、空格如何保留； - normalizer 和 pre-tokenizer； - BPE、Unigram 或 WordPiece； - 目标词表大小和计数口径； … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P044 / L128 | 用 SentencePiece 训练 BPE 或 Unigram / ### 用 SentencePiece 训练 BPE 或 Unigram | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P045 / L130 | 用 SentencePiece 训练 BPE 或 Unigram / SentencePiece 可以直接从原始文本训练 BPE 或 Unigram。下面的命令是算法示例，需要额外安装 `sentencepiece` CLI；它不在仓库默认的 `tokenizer` extra 中。将 `… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P046 / L132 | 用 SentencePiece 训练 BPE 或 Unigram / ~~~bash spm_train \ --input=data/tokenizer/train.txt \ --model_prefix=outputs/tokenizer-32k \ --model_type=bpe… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| zh-15_tokenizer-P047 / L145 | 用 SentencePiece 训练 BPE 或 Unigram / 命令会生成 tokenizer 模型和词表文件。正式记录还应保存完整命令、SentencePiece 版本、输入 manifest 的 hash 和输出文件摘要。对于多语言语料，`character_coverage`、… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P048 / L147 | 用仓库脚本训练 Byte-level BPE / ### 用仓库脚本训练 Byte-level BPE | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P049 / L149 | 用仓库脚本训练 Byte-level BPE / 仓库现在提供了实际生成 tokenizer 文件的入口：[scripts/train_tokenizer.py](https://github.com/WenbinWang12/Modern-Sequence-Model… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P050 / L151 | 用仓库脚本训练 Byte-level BPE / 先安装 tokenizer 依赖： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P051 / L153 | 用仓库脚本训练 Byte-level BPE / ~~~bash uv sync --extra tokenizer ~~~ | 补写/P1 | F25；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-15_tokenizer-P052 / L157 | 用仓库脚本训练 Byte-level BPE / 然后从 JSONL 的 `text` 字段训练一个 32K Byte-level BPE： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P053 / L159 | 用仓库脚本训练 Byte-level BPE / ~~~bash uv run python scripts/train_tokenizer.py \ --input data/tokenizer/train.jsonl \ --output outputs/token… | 补写/P1 | F25；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-15_tokenizer-P054 / L168 | 用仓库脚本训练 Byte-level BPE / 训练 Unigram 时改用 `--algorithm unigram`；如果不希望从 byte 开始，可以加入 `--no-byte-level`。输出目录包含 `tokenizer.json`、special-tok… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P055 / L170 | 用仓库脚本训练 Byte-level BPE / 脚本的主要代码路径是：读取 JSONL 并生成临时纯文本、构造 `models.BPE` 或 `models.Unigram`、创建对应 trainer、调用 `tokenizer.train(...)`，最后用 `Pr… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P056 / L172 | 直接使用 Hugging Face Tokenizers API / ### 直接使用 Hugging Face Tokenizers API | 改写/P0 | F22；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P057 / L174 | 直接使用 Hugging Face Tokenizers API / 也可以用 `tokenizers` 库在 Python 中显式定义 normalizer、pre-tokenizer 和 trainer： | 改写/P0 | F22；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P058 / L176 | 直接使用 Hugging Face Tokenizers API / ~~~python from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers tokenizer = Tokenize… | 改写/P0 | F22；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-15_tokenizer-P059 / L191 | 直接使用 Hugging Face Tokenizers API / 这段代码训练的是 Byte-level BPE。若更换到 Unigram，需要更换 model 和 trainer，同时重新记录 tokenizer 的配置；不能只修改文件名或 `vocab_size`。 | 改写/P0 | F22；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P060 / L193 | 训练完成后怎样接入模型 / ### 训练完成后怎样接入模型 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P061 / L195 | 训练完成后怎样接入模型 / 训练完成后先做独立检查： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P062 / L197 | 训练完成后怎样接入模型 / ~~~text tokenizer files → encode/decode round trip → token ID 范围检查 → special token 检查 → tokens/byte 和长度分布 → 生成… | 改写/P0 | F23；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-15_tokenizer-P063 / L208 | 训练完成后怎样接入模型 / 当前仓库的 `ttt.data` 负责加载训练好的 tokenizer 并生成 token 分片；subword tokenizer 的训练入口是 `scripts/train_tokenizer.py`。训练完成后，可… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P064 / L210 | 这些步骤在仓库中的代码位置 / ### 这些步骤在仓库中的代码位置 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P065 / L212 | 这些步骤在仓库中的代码位置 / - [`scripts/train_tokenizer.py`](https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch/blob/main… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P066 / L219 | 这些步骤在仓库中的代码位置 / 本节中的 SentencePiece 命令和 `tokenizers` Python 示例解释算法；仓库实际的生成入口是 [`scripts/train_tokenizer.py`](https://github.com… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P067 / L221 | Byte tokenizer 是什么 / ## Byte tokenizer 是什么 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P068 / L223 | Byte tokenizer 是什么 / 当前仓库的 `ByteTokenizer` 把 UTF-8 字节值 0–255 映射到 ID 3–258，并保留 0、1、2 给 pad、BOS、EOS。因此： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P069 / L225 | Byte tokenizer 是什么 / ~~~text byte 0 → token 3 byte 255 → token 258 BOS → 1 EOS → 2 PAD → 0 vocab_size = 259 ~~~ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| zh-15_tokenizer-P070 / L234 | Byte tokenizer 是什么 / 它的优点是任何 UTF-8 文本都能编码，不需要先下载词表，也没有未知字符。缺点是一个汉字通常占三个 UTF-8 字节，一个常见英文词也可能拆成多个位置，序列效率较低。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P071 / L236 | Byte tokenizer 是什么 / 这个 tokenizer 适合验证接口和恢复逻辑，不适合作为正式中英混合基模的最终选择。正式 tokenizer 要在代表性语料上测量 tokens/byte、长度分布和特殊 token 使用，不能只比较词表大小。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P072 / L238 | 特殊 token 如何进入训练协议 / ## 特殊 token 如何进入训练协议 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P073 / L240 | 特殊 token 如何进入训练协议 / 特殊 token 不是普通词。若 tokenizer 训练或编码时自动在开头加入 BOS，数据脚本又手动加入一次，模型会看到重复标记；若 EOS 在文档间丢失，连续 packing 会让边界变得不可见。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P074 / L242 | 特殊 token 如何进入训练协议 / 因此，特殊 token 的字符串、ID、添加位置和是否参与 loss 都要写入 tokenizer 配置和数据 manifest。BOS、EOS、PAD、UNK 以及对话模板中的控制 token，都必须由 tokeniz… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P075 / L244 | 分词效率要按来源统计 / ## 分词效率要按来源统计 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P076 / L246 | 分词效率要按来源统计 / 设原始文本的 UTF-8 字节数为 \(B\)，token 数为 \(T\)，可用 \(T/B\) 衡量编码密度。它不是质量分数，但能说明 sequence length 的实际覆盖。 | 修订/P2 | T21；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P077 / L248 | 分词效率要按来源统计 / 至少统计： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P078 / L250 | 分词效率要按来源统计 / &#124; 统计 &#124; 为什么需要 &#124; &#124;---&#124;---&#124; &#124; tokens/byte &#124; 估计上下文和数据成本 &#124; &#124; tokens/document &#124; 查看长短文分布 &#124; &#124; fallback/unknown 比例 &#124; 发现… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P079 / L259 | 分词效率要按来源统计 / 一个 tokenizer 可能让英文更紧凑，却让中文或代码显著变长。总体均值不能替代分组统计。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P080 / L261 | 当前代码和替换边界 / ## 当前代码和替换边界 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P081 / L263 | 当前代码和替换边界 / 当前仓库提供两个加载路径：`ttt.data` 中的 `ByteTokenizer`，以及通过 tokenizer extra 调用 Hugging Face 的 `HuggingFaceTokenizer`。后者使用 … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P082 / L265 | 当前代码和替换边界 / 数据准备入口是： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P083 / L267 | 当前代码和替换边界 / ~~~bash python -m ttt.data --input corpus.jsonl --output data/pretrain --sharded --tokenizer byte python -m tt… | 改写/P0 | F24；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-15_tokenizer-P084 / L272 | 当前代码和替换边界 / 三条实际路径的边界如下： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P085 / L274 | 当前代码和替换边界 / &#124; 路径 &#124; BOS &#124; EOS &#124; loss mask &#124; &#124;---&#124;---&#124;---&#124;---&#124; &#124; `prepare_data` 普通 NPY &#124; `encode` 默认加入 &#124; 每篇文档追加 &#124; 没有 mask 文件… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P086 / L280 | 当前代码和替换边界 / 当前仓库的 `scripts/train_tokenizer.py` 已提供 BPE 和 Unigram 训练入口，暂不提供 WordPiece 训练入口。本章中的 SentencePiece 命令和 `tokenize… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P087 / L282 | 当前代码和替换边界 / 当前 manifest 会记录 tokenizer 名称、词表大小、输入文件摘要和分片摘要，但不会自动记录特殊 token ID、normalizer、远端 revision 或 tokenizer 文件哈希。这些字段属… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P088 / L284 | 当前代码和替换边界 / 训练脚本的 `tokenizer_metadata.json` 同时记录请求词表大小和实际词表大小。接入模型时必须使用实际值，并让它与模型配置中的 `vocab_size`、数据 manifest 中的 `vocab_s… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P089 / L286 | 当前代码和替换边界 / 更换 tokenizer 后必须同步配置中的 vocab_size 与特殊 ID，并重新生成所有 token 分片。旧分片中的整数仍然合法，不代表它们在新词表中具有相同含义。checkpoint 也不能只改一个 JSON… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P090 / L288 | 当前代码和替换边界 / 下一章从 token 流进入固定窗口，处理 label shift、EOS 边界和 loss mask。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P091 / L290 | 本章最小检查、预期证据与失败诊断 / ## 本章最小检查、预期证据与失败诊断 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P092 / L292 | 本章最小检查、预期证据与失败诊断 / 先用仓库内样例确认当前 byte tokenizer 的词表和分片行为： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P093 / L294 | 本章最小检查、预期证据与失败诊断 / ~~~bash python -m ttt.data \ --input examples/sample.txt \ --output outputs/tokenizer-sample \ --sharded --tok… | 改写/P0 | F01；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-15_tokenizer-P094 / L301 | 本章最小检查、预期证据与失败诊断 / 预期 manifest 中的 &lt;code&gt;vocab_size&lt;/code&gt; 为 259，普通文本的 EOS 被追加，分片路径中的 BOS 监督 mask 按当前实现处理。若词表、特殊 ID 或 mask 与文中不一致，… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P095 / L303 | 本章小结与下一问 / ## 本章小结与下一问 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-15_tokenizer-P096 / L305 | 本章小结与下一问 / Tokenizer 同时决定整数序列、上下文覆盖、embedding/输出头参数量和特殊 token 语义。BPE/Unigram 训练脚本是生成新 tokenizer 的入口，&lt;code&gt;ttt.data&lt;/code&gt;… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/16_packing_and_masks.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| zh-16_packing_and_masks-P001 / L1 | 10　Packing、标签与 Mask：一个 batch 是怎样构造出来的 / # 10 Packing、标签与 Mask：一个 batch 是怎样构造出来的 | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P002 / L3 | 模型真正训练的不是文档 / ## 模型真正训练的不是文档 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P003 / L5 | 模型真正训练的不是文档 / 训练循环看到的是代码字段 `input_ids`、`labels` 和 `loss_mask`，不是“这是一篇文章”。一个简单的 next-token 样本可以写成： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P004 / L7 | 模型真正训练的不是文档 / 先用四个可读的符号代替 token：原序列 `[A, B, C, D]` 的输入是 `[A, B, C]`，标签是向左错开一位的 `[B, C, D]`。模型在看到 `A` 后预测 `B`，看到 `A,B` 后预测 `C… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P005 / L9 | 模型真正训练的不是文档 / \[ \begin{aligned} \text{tokens}&=(x_1,x_2,x_3,x_4),\\ X&=(x_1,x_2,x_3),\\ Y&=(x_2,x_3,x_4),\\ M&=(1,1,1). \en… | 修订/P2 | T21；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-16_packing_and_masks-P006 / L18 | 模型真正训练的不是文档 / 输入位置 \(i\) 的预测目标是原序列的 \(i+1\)。如果把 input 和 label 取成相同序列，模型就不再执行标准的 next-token 任务；如果 shift 多一位，模型会学习错位目标，却可能仍然产生… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P007 / L20 | 模型真正训练的不是文档 / 在当前实现中，`TokenDataset.rows(starts, length)` 返回长度为 `length` 的 token 窗口，并返回去掉第一列后的 mask。`BatchStream.next_batch_w… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P008 / L22 | 模型真正训练的不是文档 / 如果 `[A,B,C,D]` 中的 `D` 是 padding，第三个预测位置的 mask 应为 0；如果 `B` 来自 prompt，prompt 的 mask 也可以为 0，但它仍然可以作为上下文被模型看到。mask… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P009 / L24 | 独立文档与连续流 / ## 独立文档与连续流 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P010 / L26 | 独立文档与连续流 / 有两种常见 packing 语义。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P011 / L28 | 独立文档与连续流 / **独立文档 packing** 在每篇文档后添加 EOS，并保证注意力或 loss 不跨越边界。它容易解释：文档结束后，模型不会把下一篇文档当成上一篇的自然延续。 | 改写/P0 | F26；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P012 / L30 | 独立文档与连续流 / **连续流 packing** 将多个文档连接成一个 token 流，训练效率高，但必须定义 EOS 和边界。若只加入 EOS 而仍允许 attention 跨越它，模型可以学习跨文档的统计关系；这可能是有意的，也可能泄… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P013 / L32 | 独立文档与连续流 / 当前仓库的 `prepare_data` 和 `prepare_sharded_data` 会把每个输入文档编码后追加 EOS，再把 token 依次写入分片。`TokenDataset.rows` 按全局 token … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P014 / L34 | Loss mask 与 padding / ## Loss mask 与 padding | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P015 / L36 | Loss mask 与 padding / mask 的含义是“这个 label 是否参与 loss”，不是“这个 token 是否存在”。prompt-response 监督可以让 prompt 进入上下文但 mask 为 0；padding 也可以 mask … | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P016 / L38 | Loss mask 与 padding / `token_cross_entropy` 会将 mask 为 0 的 target 替换成安全 ID，然后把对应 loss 置零。替换是为了避免无效 ID 触发索引错误，不是让模型学习这个安全 ID。平均 loss 的… | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P017 / L40 | Loss mask 与 padding / 跨设备或跨样本聚合时，也要用有效 token 数作权重。直接平均每个样本的 loss，会让一条有效 token 很少的样本和一条有效 token 很多的样本拥有相同权重。 | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P018 / L42 | 一个可以手算的例子 / ## 一个可以手算的例子 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P019 / L44 | 一个可以手算的例子 / 设 token 序列为 \(z=(10,11,12,2,20)\)，其中 2 是 EOS。长度为 4 的训练窗口是 \(X=(10,11,12,2)\)、\(Y=(11,12,2,20)\)。 | 修订/P2 | T21；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P020 / L46 | 一个可以手算的例子 / 如果我们把 EOS 后的 20 视为下一篇文档的第一个 token，最后一项是否训练取决于 packing 约定。若希望文档之间不互相监督，mask 可以是 \(M=(1,1,1,0)\)；若把整个 token 流当作连… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P021 / L48 | 一个可以手算的例子 / 这就是为什么“加了 EOS”还不足以说明边界语义。必须同时写出 input、label、mask 和 attention mask。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P022 / L50 | 当前训练路径 / ## 当前训练路径 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P023 / L52 | 当前训练路径 / `ttt/inner_loop.sequence_loss` 接收 `tokens` 和可选的 `loss_mask`，将序列拆成 ttt_chunk_size 的块；尾部 padding 的位置 mask 为 0。`t… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P024 / L54 | 当前训练路径 / 这段代码解决了 token-level mask 和 TTT chunk 的关系，但没有提供按文档边界重置 KV cache 的通用接口。若要加入独立文档 packing，需要在数据和模型状态之间新增契约，而不是只修改 … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P025 / L56 | 当前训练路径 / 下一章将 token 分片、mask 分片和 manifest 放进一个可恢复的 Dataset；这样我们可以检查文件内容，而不只是检查 batch shape。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P026 / L58 | 本章最小检查、预期证据与失败诊断 / ## 本章最小检查、预期证据与失败诊断 | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P027 / L60 | 本章最小检查、预期证据与失败诊断 / ~~~bash python -m ttt.data \ --input examples/sample.txt \ --output outputs/packing-sample \ --sharded --token… | 改写/P0 | F01、T13；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-16_packing_and_masks-P028 / L67 | 本章最小检查、预期证据与失败诊断 / 随后用一个固定窗口检查 &lt;code&gt;input_ids&lt;/code&gt;、&lt;code&gt;labels&lt;/code&gt; 和 &lt;code&gt;loss_mask&lt;/code&gt; 的长度是否分别是 \(L\)、\(L\)、\(L\)，并手算… | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P029 / L69 | 本章小结与下一问 / ## 本章小结与下一问 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-16_packing_and_masks-P030 / L71 | 本章小结与下一问 / Packing 定义窗口，label shift 定义预测目标，loss mask 定义哪些目标有监督；EOS 只表达边界，不自动决定边界之后的监督。下一章把这些数组写入带 hash 和 offset 的 Dataset… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/17_dataset_and_manifest.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| zh-17_dataset_and_manifest-P001 / L1 | 11　Dataset、分片与 Manifest：把数据变成可恢复输入 / # 11 Dataset、分片与 Manifest：把数据变成可恢复输入 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P002 / L3 | 为什么一个目录不够 / ## 为什么一个目录不够 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P003 / L5 | 为什么一个目录不够 / 如果训练脚本只是读取目录中的所有 `.npy`，它依赖了文件系统的排序、隐藏文件和人工约定。换一台机器、增加一个分片或清理临时文件，都可能改变 token 流。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P004 / L7 | 为什么一个目录不够 / Manifest 把这些隐含条件写成数据版本。当前 loader 期待的核心字段是： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P005 / L9 | 为什么一个目录不够 / ~~~json { "format_version": 1, "vocab_size": 259, "shards": [ { "tokens": "v-00000.npy", "mask": "v-00000.mask… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| zh-17_dataset_and_manifest-P006 / L25 | 为什么一个目录不够 / `tokens` 和 `mask` 是相对于 manifest 所在目录的文件名，loader 还要求分片不能逃出该目录。这个限制看似保守，却避免 manifest 通过绝对路径偷偷读取另一台机器的文件。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P007 / L27 | TokenDataset 做了哪些安全检查 / ## TokenDataset 做了哪些安全检查 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P008 / L29 | TokenDataset 做了哪些安全检查 / `TokenDataset.__init__` 会检查 manifest 版本和 vocab_size；逐个验证 token 与 mask 文件的 SHA256；使用 memory mapping 读取一维非空整数数组；… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P009 / L31 | TokenDataset 做了哪些安全检查 / 这些检查有不同作用。Hash 证明内容没有被替换，范围检查证明输入不会让 embedding 越界，shape 检查证明 token 与 mask 能配对。一个通过 hash 的文件仍可能是错误词表下生成的合法整数，因此… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P010 / L33 | TokenDataset 做了哪些安全检查 / 当前 `TokenDataset` 还会计算分片端点，并让任意全局 start 经过 `searchsorted` 找到物理 shard。窗口跨 shard 时，代码按各分片的局部 offset 取值。这是让大数据不必完… | 改写/P0 | F27；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P011 / L35 | 一个跨分片窗口的手算 / ## 一个跨分片窗口的手算 | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P012 / L37 | 一个跨分片窗口的手算 / 把两个分片写成最小例子： | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P013 / L39 | 一个跨分片窗口的手算 / &#124; 物理 shard &#124; token 内容 &#124; count &#124; 全局结束 offset &#124; &#124;---&#124;---&#124;---:&#124;---:&#124; &#124; &lt;code&gt;v-00000.npy&lt;/code&gt; &#124; &lt;code&gt;[10, 11, … | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P014 / L44 | 一个跨分片窗口的手算 / 这里的 offset 是代码数组索引的边界，采用半开区间：第一个分片覆盖全局位置 [0,4)，第二个覆盖 [4,9)。如果请求 &lt;code&gt;start=2, length=5&lt;/code&gt;，窗口要读取全局位置 2、3、4… | 改写/P0 | F27、T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P015 / L46 | 一个跨分片窗口的手算 / ~~~text global positions: 2, 3 &#124; 4, 5, 6 physical shard: v-0 &#124; v-1 local offsets: 2, 3 &#124; 0, 1, 2 tokens: 12,13… | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-17_dataset_and_manifest-P016 / L53 | 一个跨分片窗口的手算 / 如果模型窗口长度是 5，训练代码还需要额外读取一个 token 来构造 next-token target；因此可用窗口长度和底层读取长度不是同一个数字。分片边界不会自动插入 EOS，也不会重置 attention；它只… | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P017 / L55 | Manifest 身份与文件身份 / ## Manifest 身份与文件身份 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P018 / L57 | Manifest 身份与文件身份 / manifest 有自己的 SHA256，分片也有各自 SHA256。两层都需要保留： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P019 / L59 | Manifest 身份与文件身份 / - 分片 hash 用来发现单个文件损坏或替换； - manifest hash 用来发现分片顺序、计数或配置被改； - 输入原始文件 hash 用来追溯 token 是从什么内容生成的。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P020 / L63 | Manifest 身份与文件身份 / 如果只保存分片 hash，却把 manifest 重新排序，单个文件都没有变化，训练序列却会改变。反过来，manifest 一样但文件内容被替换，分片 hash 会阻止读取。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P021 / L65 | Manifest 身份与文件身份 / 当前 prepare_sharded_data 给每次生成的数据分配 UUID 版本，并拒绝覆盖已有 manifest。这能保护旧数据，但也说明“重复运行得到字节完全相同的目录”并非当前目标。若需要内容寻址，应把版本从随… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P022 / L67 | Dataset 与随机采样器 / ## Dataset 与随机采样器 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P023 / L69 | Dataset 与随机采样器 / Dataset 描述“有哪些 token”，采样器描述“下一步取哪里”。`BatchStream` 使用 NumPy Generator 随机取 start；state_dict 保存 generator state、数… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P024 / L71 | Dataset 与随机采样器 / 恢复时必须同时验证这些字段。仅使用同一个 seed 重新创建 generator，不会跳到中断时的内部状态；仅保存 generator state 而不验证数据摘要，又可能在新数据上继续执行。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P025 / L73 | Dataset 与随机采样器 / `sample_rows(step, indices)` 使用 `SeedSequence` 从 seed、step 和 index 派生独立窗口，适合固定探针和评估样本。它与 `next_batch` 的随机流不是同一… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P026 / L75 | 大数据的实际约束 / ## 大数据的实际约束 | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P027 / L77 | 大数据的实际约束 / memory mapping 减少了 Python 堆内存，但不会消除存储带宽和随机访问成本。窗口完全随机时，操作系统缓存命中率可能低；分片太小会增加文件管理和 hash 开销，太大又会让单个文件难以迁移。 | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P028 / L79 | 大数据的实际约束 / 工程上通常会把逻辑 shard、物理 shard 和训练 batch 分开设计。逻辑上可以按文档簇组织，物理上按顺序写大文件，训练时再按 worker/rank 切分。当前仓库是教学实现，先保证校验与恢复，再谈异步预取、… | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P029 / L81 | 大数据的实际约束 / 下一章讨论多个来源如何混合。只有知道 sampler 读取的是哪个来源，我们才能控制训练配方。 | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P030 / L83 | 本章最小检查、预期产物与失败诊断 / ## 本章最小检查、预期产物与失败诊断 | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P031 / L85 | 本章最小检查、预期产物与失败诊断 / 先生成一个小分片，再只读检查 manifest： | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P032 / L87 | 本章最小检查、预期产物与失败诊断 / ~~~bash python -m ttt.data \ --input examples/sample.txt \ --output outputs/dataset-sample \ --sharded --token… | 改写/P0 | F01、T14；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-17_dataset_and_manifest-P033 / L95 | 本章最小检查、预期产物与失败诊断 / 预期产物是 manifest、token/mask shard 和每个 shard 的 digest。若两个 shard 的累计 offset 不连续，或窗口跨 shard 后 token 顺序改变，不能继续训练；先检查… | 改写/P0 | F27、T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P034 / L97 | 本章小结与下一问 / ## 本章小结与下一问 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-17_dataset_and_manifest-P035 / L99 | 本章小结与下一问 / Dataset 是逻辑 token 流和采样协议，shard 只是存储片段，manifest 是两者之间可校验的契约。恢复不仅要有 seed，还要有数据 digest、采样位置和窗口协议。下一章讨论这些窗口来自哪些来源，… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/18_data_mixture.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| zh-18_data_mixture-P001 / L1 | 12　数据混合：训练配方不是文件大小 / # 12 数据混合：训练配方不是文件大小 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-18_data_mixture-P002 / L3 | 为什么直接拼文件会产生偏差 / ## 为什么直接拼文件会产生偏差 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-18_data_mixture-P003 / L5 | 为什么直接拼文件会产生偏差 / 假设网页有 900 GB，代码有 50 GB，书籍有 50 GB。把三者拼成一个流，模型大约 90% 的 token 来自网页。这个比例可能是合理的，也可能完全不符合研究目标，但它不是“自然发生的客观比例”，而是文件容量… | 改写/P0 | F28；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-18_data_mixture-P004 / L7 | 为什么直接拼文件会产生偏差 / 另一个极端是每个来源按文档均匀采样。短代码片段和长书籍章节会获得不同的 token 权重。我们要先选择目标分布，再选择一个 sampler 实现它。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-18_data_mixture-P005 / L9 | 从 token 预算写配方 / ## 从 token 预算写配方 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-18_data_mixture-P006 / L11 | 从 token 预算写配方 / 一个配方示例： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-18_data_mixture-P007 / L13 | 从 token 预算写配方 / ~~~json { "sources": [ {"name": "web", "weight": 0.70}, {"name": "code", "weight": 0.20}, {"name": "books", "w… | 改写/P0 | F29；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-18_data_mixture-P008 / L27 | 从 token 预算写配方 / `weight` 是目标抽样概率，不保证最终严格命中。来源可能提前耗尽，清洗后可用量也会改变。若使用温度采样，可按原始权重的幂重新归一化；温度小于 1 会提高小来源的相对权重，但具体公式和耗尽策略必须写入实现。 | 改写/P0 | F29；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-18_data_mixture-P009 / L29 | 从 token 预算写配方 / 数据混合应至少有两张表：配方表描述计划，消费表描述实际。消费表按有效 token 统计，而不是只按 batch 次数，因为 mask、padding 和丢弃样本会改变有效量。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-18_data_mixture-P010 / L31 | 一个温度采样的算例 / ## 一个温度采样的算例 | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-18_data_mixture-P011 / L33 | 一个温度采样的算例 / 设三个来源的原始权重为 \([0.8, 0.15, 0.05]\)，温度 \(\tau=0.5\)，用 \(w_i^{1/\tau}\) 作为新权重，得到 \([0.64, 0.0225, 0.0025]\)，归一化后小… | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-18_data_mixture-P012 / L35 | 一个温度采样的算例 / 若目标是提升小来源，采用 \(w_i^\tau\) 且 \(\tau&lt;1\) 的约定，示意值为 \([0.894,0.387,0.224]\)，归一化后分别约为 59.4%、25.7%、14.9%。这些数只用于说明公式，… | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-18_data_mixture-P013 / L37 | 确定性要落到 rank 和恢复 / ## 确定性要落到 rank 和恢复 | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-18_data_mixture-P014 / L39 | 确定性要落到 rank 和恢复 / 单机 sampler 保存 seed 和 RNG state 已经不够分布式使用。多个 rank 需要决定：先全局抽来源再按 rank 分发，还是每个 rank 独立抽样；两者会产生不同重复率和通信量。 | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-18_data_mixture-P015 / L41 | 确定性要落到 rank 和恢复 / 更稳妥的设计是从全局 sample index 派生来源和窗口，再让 rank 取自己的索引。这样恢复时只要恢复全局 step，就能重建各 rank 的数据顺序。当前仓库的 `BatchStream` 是单一 token… | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-18_data_mixture-P016 / L43 | 如何验证混合比例 / ## 如何验证混合比例 | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-18_data_mixture-P017 / L45 | 如何验证混合比例 / 训练前，固定 seed 生成一段不更新模型的 sample plan，统计每个来源的目标与实际 token。训练中累计记录来源 token、mask 有效 token和丢弃 token。恢复后，把恢复前后的 sample… | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-18_data_mixture-P018 / L47 | 如何验证混合比例 / 还要看短期窗口的波动。总体 70/20/10 不代表每个 1,000 token 的小批量都接近这个比例；小来源在短窗口里可能有较大方差。报告可同时给累计比例和滑动窗口区间。 | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-18_data_mixture-P019 / L49 | 如何验证混合比例 / 下一章进行训练前验收，把 tokenizer、manifest、mask、混合配方和模型配置放到同一个检查点上。 | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-18_data_mixture-P020 / L51 | 本章最小检查、预期证据与失败诊断 / ## 本章最小检查、预期证据与失败诊断 | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-18_data_mixture-P021 / L53 | 本章最小检查、预期证据与失败诊断 / 当前仓库没有来源级混合 sampler，因此先把配方保存为 JSON，再检查它是否能被实验读取： | 改写/P0 | F31、T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-18_data_mixture-P022 / L55 | 本章最小检查、预期证据与失败诊断 / ~~~bash python -c "import json; p='outputs/recipe.json'; print(json.dumps(json.load(open(p, encoding='utf-8'))… | 改写/P0 | F31、T15；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-18_data_mixture-P023 / L59 | 本章最小检查、预期证据与失败诊断 / 成功输出只能证明 recipe 文件可解析。真正的混合实验还必须记录每个来源的目标 token、实际 token、有效 mask token、sample index 和恢复前后是否重复；如果只有权重而没有消费日志，不能… | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-18_data_mixture-P024 / L61 | 本章小结与下一问 / ## 本章小结与下一问 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-18_data_mixture-P025 / L63 | 本章小结与下一问 / 混合配方是显式概率分布和 token 预算，不是文件大小的副产品。温度公式必须写明约定，rank 和恢复协议必须从全局 sample index 定义。下一章把 tokenizer、manifest、mask 和 rec… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/19_data_preflight.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| zh-19_data_preflight-P001 / L1 | 13　训练前验收：在提交长作业之前发现错误 / # 13 训练前验收：在提交长作业之前发现错误 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P002 / L3 | Preflight 为什么值得单独成章 / ## Preflight 为什么值得单独成章 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P003 / L5 | Preflight 为什么值得单独成章 / 一个 4 步 tiny 训练通常能在几分钟内发现错误；一个 3 天作业可能在下载、编译和几千步之后才失败。Preflight 不训练模型，它把“输入能否组成一个自洽实验”提前变成明确的 pass、warning 或 fa… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P004 / L7 | Preflight 为什么值得单独成章 / 它不是只检查文件存在。它需要把数据、tokenizer、模型配置、采样和输出目录放到同一个关系图里。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P005 / L9 | 八项验收 / ## 八项验收 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P006 / L11 | 八项验收 / **词表对齐。** manifest 的 vocab_size、tokenizer 的 vocab_size、model config 的 vocab_size 必须相同。特殊 token ID 必须位于词表范围内，且数… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P007 / L13 | 八项验收 / **文件完整。** 每个 token 和 mask shard 存在，SHA256 一致，count 与数组长度一致，数组是一维整数，mask shape 相同。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P008 / L15 | 八项验收 / **样本可训练。** sequence length 至少为 1，窗口能取得 length+1 个 token，有效 mask 不全为零，token 不越界。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P009 / L17 | 八项验收 / **边界清楚。** 文档切分、EOS、packing 和是否跨文档监督已经写入版本记录。若当前实现允许跨 EOS 取窗口，报告不能声称它隔离了文档上下文。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P010 / L19 | 八项验收 / **数据分布。** train/validation 的文档簇不重叠；来源、语言、长度和有效 token 统计可以对账；污染报告没有未解释的 blocked 项。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P011 / L21 | 八项验收 / **采样确定。** seed、混合配方、shuffle、rank 分工、起始位置和 resume 语义有记录。当前仓库单流随机 sampler 的限制也要显示出来。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P012 / L23 | 八项验收 / **模型接口。** sequence length 不超过 max_position_embeddings；ttt_chunk_size、sliding_window 和 batch mask 满足配置约束；fast M… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P013 / L25 | 八项验收 / **输出安全。** 输出目录为空，或明确指定了 resume/initialize-from；checkpoint 的数据指纹与当前输入一致。 | 改写/P0 | F32；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P014 / L27 | 把验收写成报告 / ## 把验收写成报告 | 补写/P1 | T16；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P015 / L29 | 把验收写成报告 / 推荐每一项写成结构化记录： | 补写/P1 | T16；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P016 / L31 | 把验收写成报告 / ~~~json { "name": "vocab_alignment", "status": "pass", "inputs": { "manifest": "&lt;sha256&gt;", "config": "&lt;sha256&gt;… | 补写/P1 | T16；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-19_data_preflight-P017 / L47 | 把验收写成报告 / `warning` 应该包含人工决定和影响范围；`fail` 应阻止提交长作业。报告本身保存到 run 目录，并被 experiment.json 引用。 | 补写/P1 | T16；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P018 / L49 | 把验收写成报告 / 当前仓库还没有独立的 `ttt.preflight` 命令。读者可以先使用数据 loader 的校验和 tiny 训练完成最小验收；后续应把上面的检查抽成无 GPU 的命令，方便调度器在申请 GPU 之前运行。 | 补写/P1 | T16；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P019 / L51 | 一个真实的验收报告 / ## 一个真实的验收报告 | 改写/P0 | F30、T16；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P020 / L53 | 一个真实的验收报告 / 下面是一个 tiny 运行的简化报告。文件和窗口检查通过，跨 EOS 的连续流被标成 warning，而词表不一致直接 fail： | 改写/P0 | F30、T16；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P021 / L55 | 一个真实的验收报告 / ~~~json { "checks": [ {"name": "file_integrity", "status": "pass", "message": "2 shards and masks match their … | 改写/P0 | F30、T16；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-19_data_preflight-P022 / L66 | 一个真实的验收报告 / 这个结果不能靠“先跑起来再说”处理。应先确认模型配置是否应当使用 byte tokenizer 的 259 个位置；如果确实要换成 32K tokenizer，就必须重新生成 token shard、更新 manifes… | 改写/P0 | F30、T16；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P023 / L68 | 一个真实的验收报告 / ~~~bash python -c "from ttt.dataset import TokenDataset; TokenDataset('runs/example_shards/manifest.json', 259… | 改写/P0 | F30、F31、T16；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-19_data_preflight-P024 / L72 | 一个真实的验收报告 / 这条命令只能证明 loader 的文件和范围检查通过，不能替代上面未实现的 split、污染和分布检查。 | 改写/P0 | F30、T16；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P025 / L74 | 从本章进入模型 / ## 从本章进入模型 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P026 / L76 | 从本章进入模型 / 验收通过只说明输入自洽，不说明模型已经学会语言。下一章定义 dense baseline：固定这些输入，测量 loss、梯度、恢复和计算成本，随后才有资格讨论 TTT。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P027 / L78 | 本章最小检查、预期报告与失败诊断 / ## 本章最小检查、预期报告与失败诊断 | 补写/P1 | T16；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P028 / L80 | 本章最小检查、预期报告与失败诊断 / 当前没有独立的 &lt;code&gt;ttt.preflight&lt;/code&gt; 命令，先强制 loader 读取 manifest： | 补写/P1 | T16；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P029 / L82 | 本章最小检查、预期报告与失败诊断 / ~~~bash python -c "from ttt.dataset import TokenDataset; TokenDataset('outputs/dataset-sample/manifest.json', … | 补写/P1 | T16；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-19_data_preflight-P030 / L86 | 本章最小检查、预期报告与失败诊断 / 预期输出是 &lt;code&gt;manifest and shards: pass&lt;/code&gt;；这只证明 loader 的范围和 hash 检查通过。若得到 vocabulary mismatch，先比较 &lt;code&gt;voca… | 补写/P1 | T16；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P031 / L88 | 本章小结与下一问 / ## 本章小结与下一问 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-19_data_preflight-P032 / L90 | 本章小结与下一问 / Preflight 的输出应明确区分 pass、warning 和 fail。通过只说明输入对象相互一致，不说明模型质量、泛化或架构比较公平。下一章在同一数据身份上建立 dense baseline。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/20_baseline_model.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| zh-20_baseline_model-P001 / L1 | 15　Dense Causal LM 基线：先证明训练系统能学习 / # 15 Dense Causal LM 基线：先证明训练系统能学习 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P002 / L3 | 基线不是最小的玩具 / ## 基线不是最小的玩具 | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P003 / L5 | 基线不是最小的玩具 / 我们要比较 TTT、Loop 和 MoE，必须先有一个不会把数据、目标或恢复问题隐藏起来的 dense baseline。它不追求结构新颖，而要把输入 token 变成下一个 token 的 logits，并在固定数据上… | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P004 / L7 | 基线不是最小的玩具 / 基线至少固定：tokenizer 和特殊 token、训练/验证 manifest、sequence length、有效 token 预算、初始化、优化器、学习率计划、随机种子、精度、硬件和 checkpoint 格式。… | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P005 / L9 | 从 logits 到 loss / ## 从 logits 到 loss | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P006 / L11 | 从 logits 到 loss / `logits` 先不要理解成概率，它只是模型给每个候选 token 的分数。例如词表只有 `cat`、`sat`、`dog`，某个位置的分数是 $(2,1,0)$，softmax 才会把它们变成和为 1 的概率： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P007 / L13 | 从 logits 到 loss / $$ p_v=\frac{\exp z_v}{\sum_u\exp z_u}. $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| zh-20_baseline_model-P008 / L17 | 从 logits 到 loss / 如果正确答案是 `sat`，这个分数对应的概率约为 $0.245$，负对数似然约为 $1.407$。模型把正确答案的概率提高到 $0.8$ 时，损失降为 $-\log(0.8)\approx0.223$。因此交叉熵不是一… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P009 / L19 | 从 logits 到 loss / 模型产生 \(z_{t,v}\)，其中 \(t\) 是位置、\(v\) 是词表 ID。交叉熵为： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P010 / L21 | 从 logits 到 loss / $$ \ell_t=-\log\frac{\exp z_{t,y_t}} {\sum_{v=1}^{V}\exp z_{t,v}}. $$ | 改写/P0 | F33；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-20_baseline_model-P011 / L26 | 从 logits 到 loss / 直接对大词表计算完整 softmax 可能占用显存，因此仓库的 `loss.py` 提供按 token 和 vocab block 处理的路径。数学目标不因分块而改变；实现只是把求和拆成可执行的块。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P012 / L28 | 从 logits 到 loss / 有效平均 loss 为： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P013 / L30 | 从 logits 到 loss / $$ \mathcal L= \frac{\sum_t m_t\ell_t}{\max(\sum_t m_t,1)}, $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| zh-20_baseline_model-P014 / L35 | 从 logits 到 loss / 其中 \(m_t\) 是 mask。最后的 max 防止全零 mask 时除零，但全零样本本身应该被 preflight 拒绝。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P015 / L37 | 从 logits 到 loss / 对 logits 的梯度有一个很有用的形式： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P016 / L39 | 从 logits 到 loss / $$ \frac{\partial\ell_t}{\partial z_{t,v}} =p_{t,v}-\mathbf{1}[v=y_t]. $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| zh-20_baseline_model-P017 / L44 | 从 logits 到 loss / 正确 token 的梯度是 $p_{t,y_t}-1$，会推动它的分数上升；其他 token 的梯度是它们当前的概率，会推动它们的分数下降。反向传播只是把这个局部误差信号沿模型的计算图继续传回 embedding、att… | 改写/P0 | F36；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P018 / L46 | 从 logits 到 loss / 这个结果可以直接从损失展开得到： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P019 / L48 | 从 logits 到 loss / $$ \ell_t=-z_{t,y_t}+\log\left(\sum_u\exp z_{t,u}\right), \qquad \frac{\partial\ell_t}{\partial z_{t,v}} =-\ma… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| zh-20_baseline_model-P020 / L55 | 从 logits 到 loss / 第二项正是 softmax 概率，所以才得到上面的 $p_{t,v}-\mathbf{1}[v=y_t]$。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P021 / L57 | 一次最小的梯度更新 / ## 一次最小的梯度更新 | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P022 / L59 | 一次最小的梯度更新 / 先只在 logits 空间做一个手算，隔离“梯度方向”这个概念。仍使用分数 \(\mathbf z=(2,1,0)\)，正确 token 是第二个位置 &lt;code&gt;sat&lt;/code&gt;，所以 \(\mathbf p\ap… | 改写/P0 | T17、F36；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P023 / L64 | 一次最小的梯度更新 / $$ \mathbf g\approx(0.665,-0.755,0.090). $$ | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-20_baseline_model-P024 / L68 | 一次最小的梯度更新 / 如果暂时把 logits 当成可以直接更新的变量，并取学习率 \(\eta=0.1\)，一次普通 SGD 是 | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P025 / L71 | 一次最小的梯度更新 / $$ \mathbf z'=\mathbf z-\eta\mathbf g \approx(1.9335,1.0755,-0.009). $$ | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| zh-20_baseline_model-P026 / L76 | 一次最小的梯度更新 / 更新后正确 token 的分数从 1 增加到约 1.076，softmax 概率从约 0.245 上升到约 0.271。真实训练不会直接把 logits 当参数；它会把 \(\mathbf g\) 沿计算图传回 \(W\… | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P027 / L82 | 基线配置到底关闭了什么 / ## 基线配置到底关闭了什么 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P028 / L84 | 基线配置到底关闭了什么 / ~~~text num_experts = 0 use_fast_mlp = false inner.enabled = false loop_mode = none ~~~ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| zh-20_baseline_model-P029 / L91 | 基线配置到底关闭了什么 / 这些开关表达“没有额外动态状态、没有路由、没有深度循环”。注意力仍然可以是 sliding 或 full；如果要研究注意力类型，需把它作为独立变量，不能把所有默认项混成一个 baseline 名称。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P030 / L93 | 基线配置到底关闭了什么 / 仓库的 configs/tiny.json 是训练配置包装，model 子对象最终由 TTTE2EConfig 解析。configs/125m 目录下的配置是配方和消融候选，125M 是档位名，不代表启用 MLP′、专家… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P031 / L95 | 训练一步如何发生 / ## 训练一步如何发生 | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P032 / L97 | 训练一步如何发生 / `ttt/training.py` 先根据配置创建 AdamW 和学习率 schedule。每步： | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P033 / L99 | 训练一步如何发生 / 1. BatchStream 取窗口与 mask； 2. `sequence_loss` 对每个样本计算 forward、内循环和有效 loss； 3. 外循环对静态参数求梯度； 4. 梯度裁剪并由 Optax 更新参数… | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P034 / L106 | 训练一步如何发生 / 当 `inner.enabled=false` 时，`sequence_loss` 仍沿用同一条路径，只是不更新 fast weights。这个共享路径很重要：C/F 对照改变的是更新行为，而不是另一套模型实现。 | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P035 / L108 | 用 loss 判断“可学习” / ## 用 loss 判断“可学习” | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P036 / L110 | 用 loss 判断“可学习” / 教学运行只需要观察方向和数值健康： | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P037 / L112 | 用 loss 判断“可学习” / - loss 是有限值； - 多步运行没有立即变成 NaN； - 梯度范数不是持续为零或无界增长； - 重复恢复后，下一步与未中断运行接近； - held-out loss 的计算使用独立 manifest。 | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P038 / L118 | 用 loss 判断“可学习” / “loss 下降”本身不是能力证明。随机数据也可能在有限步内被记忆，极小样本更容易过拟合。tiny baseline 的任务是验证接口和优化，不是证明架构可扩展。 | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P039 / L120 | 用 loss 判断“可学习” / 可以记录训练 loss 与 held-out loss 的差值。如果训练下降而 held-out 上升，先检查数据切分、重复和采样，再讨论正则化。数据量太小时，曲线只能作为排错信号。 | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P040 / L122 | 参数、激活和实际成本 / ## 参数、激活和实际成本 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P041 / L124 | 参数、激活和实际成本 / 参数量可由模型树叶元素数量统计，但不等于每步激活内存。TTT 还会保存或重算快权重、梯度和 chunk 状态；MoE 的总参数可能很大，但每个 token 激活的专家较少。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P042 / L126 | 参数、激活和实际成本 / `ttt/profile.py` 当前提供参数量、fast MLP 参数量、prefill/decode 时间、状态字节、设备峰值内存和 XLA cost analysis。cost analysis 是估计，不是硬件实… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P043 / L128 | 基线通过后的比较纪律 / ## 基线通过后的比较纪律 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P044 / L130 | 基线通过后的比较纪律 / 对于任一新模型，实验表至少包含： | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P045 / L132 | 基线通过后的比较纪律 / &#124; 维度 &#124; 必须固定或报告 &#124; &#124;---&#124;---&#124; &#124; 数据 &#124; manifest、有效 token、来源与污染状态 &#124; &#124; 模型 &#124; 总参数、激活参数、层结构、词表 &#124; &#124; 算法 &#124; optimizer、学习率、内循… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P046 / L140 | 基线通过后的比较纪律 / 如果 TTT 只在更长训练或更多 token 下胜出，应报告预算差异；如果 Loop 每步更慢但质量更好，应同时报告质量-计算曲线。这样 baseline 才是研究参照，而不是一个被快速越过的占位页。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P047 / L142 | 基线通过后的比较纪律 / 下一部分进入模型内部：先逐层追踪一个 token，再解释 TTT 的快权重究竟在何时改变。 | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P048 / L144 | 本章最小运行、预期证据与失败诊断 / ## 本章最小运行、预期证据与失败诊断 | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P049 / L146 | 本章最小运行、预期证据与失败诊断 / 先用样例 token 和 tiny 配置启动一个短运行： | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P050 / L148 | 本章最小运行、预期证据与失败诊断 / ~~~bash bash scripts/prepare_sample.sh outputs/sample-tokens.npy JAX_PLATFORMS=cpu bash scripts/run_train.sh \… | 改写/P0 | F05、T17；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| zh-20_baseline_model-P051 / L156 | 本章最小运行、预期证据与失败诊断 / 成功目录至少应包含 &lt;code&gt;experiment.json&lt;/code&gt;、&lt;code&gt;environment.json&lt;/code&gt;、&lt;code&gt;metrics.jsonl&lt;/code&gt;、&lt;code&gt;status.j… | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P052 / L158 | 本章小结与下一问 / ## 本章小结与下一问 | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| zh-20_baseline_model-P053 / L160 | 本章小结与下一问 / Dense baseline 固定数据身份、预算、配置和恢复协议，用 logits、mask loss、梯度、checkpoint 和 held-out loss 证明训练系统至少是可解释的。它不会证明 TTT、Loop… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/en/notation.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| en-notation-P001 / L1 | Notation and conventions / # Notation and conventions | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-notation-P002 / L3 | Overview / ## Overview | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-notation-P003 / L5 | Overview / This book discusses data, tensors, probability, optimization, and runtime state in one chain. If a symbol chan… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-notation-P004 / L7 | Overview / After this chapter, you should be able to turn text into token IDs, construct \(X\), \(Y\), and \(M\), and exp… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-notation-P005 / L9 | A running example: “the cat sits” / ## A running example: “the cat sits” | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-notation-P006 / L11 | A running example: “the cat sits” / For hand calculation, suppose a tokenizer maps a short sentence and an end marker to the following **teaching … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-notation-P007 / L13 | A running example: “the cat sits” / ~~~text text token: the cat sits &lt;EOS&gt; token ID: 17 42 31 2 math position: 1 2 3 4 code index: 0 1 2 3 ~~~ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| en-notation-P008 / L20 | A running example: “the cat sits” / Real tokenizers may assign different IDs. An ID is only an integer coordinate in the vocabulary. The model see… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-notation-P009 / L22 | A running example: “the cat sits” / $$ \text{mathematical position }t \quad\longleftrightarrow\quad \text{code index }t-1. $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| en-notation-P010 / L28 | A running example: “the cat sits” / Thus &lt;code&gt;tokens[2]&lt;/code&gt; is mathematical position \(t=3\), not position 2. One-based sums are convenient in… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-notation-P011 / L30 | From a token stream to \(X,Y,M\) / ## From a token stream to \(X,Y,M\) | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-notation-P012 / L32 | From a token stream to \(X,Y,M\) / A causal language model predicts the next token from the context available so far. For the four-token stream, … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-notation-P013 / L34 | From a token stream to \(X,Y,M\) / $$ X=(x_1,x_2,x_3)=(17,42,31), \qquad Y=(y_1,y_2,y_3)=(42,31,2). $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| en-notation-P014 / L40 | From a token stream to \(X,Y,M\) / Position 1 predicts “cat” after “the”; position 2 predicts “sits” after “the cat”; position 3 predicts EOS aft… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-notation-P015 / L42 | From a token stream to \(X,Y,M\) / ~~~python input_ids = [17, 42, 31] labels = [42, 31, 2] loss_mask = [1, 1, 1] ~~~ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| en-notation-P016 / L48 | From a token stream to \(X,Y,M\) / The code field &lt;code&gt;input_ids&lt;/code&gt; is mathematical \(X\), &lt;code&gt;labels&lt;/code&gt; is \(Y\), and &lt;code&gt;loss_mask… | 改写/P0 | F02；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-notation-P017 / L50 | Indices, shapes, and batches / ## Indices, shapes, and batches | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-notation-P018 / L52 | Indices, shapes, and batches / &#124; Object &#124; Symbol &#124; Convention &#124; &#124;---&#124;---&#124;---&#124; &#124; batch sample &#124; \(b\) &#124; \(b\in\{1,\ldots,B\}\) &#124; &#124; token posit… | 修订/P2 | T21；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-notation-P019 / L64 | Indices, shapes, and batches / A batch of inputs is | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-notation-P020 / L66 | Indices, shapes, and batches / $$ X\in\{0,\ldots,V-1\}^{B\times L}. $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| en-notation-P021 / L70 | Indices, shapes, and batches / For \(B=1,L=3\), &lt;code&gt;input_ids&lt;/code&gt; has shape &lt;code&gt;[1, 3]&lt;/code&gt;. Looking up each ID produces hidden vect… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-notation-P022 / L72 | Embeddings, hidden states, logits, and probabilities / ## Embeddings, hidden states, logits, and probabilities | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-notation-P023 / L74 | Embeddings, hidden states, logits, and probabilities / These terms name different objects: | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-notation-P024 / L76 | Embeddings, hidden states, logits, and probabilities / 1. A **token ID** is an integer coordinate, such as 17. 2. An **embedding** is the \(d\)-dimensional vector lo… | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-notation-P025 / L82 | Embeddings, hidden states, logits, and probabilities / Let the embedding matrix be \(E\in\mathbb{R}^{V\times d}\). Token 17 selects \(E_{17,:}\). If a model produces… | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-notation-P026 / L84 | Embeddings, hidden states, logits, and probabilities / $$ z=hW_{\mathrm{out}}\in\mathbb{R}^{V}. $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-notation-P027 / L88 | Embeddings, hidden states, logits, and probabilities / The component \(z_v\) is a score, not a probability. Softmax gives: | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-notation-P028 / L90 | Embeddings, hidden states, logits, and probabilities / $$ p_\theta(v\mid x_{1:t}) = \frac{\exp z_v}{\sum_{u=0}^{V-1}\exp z_u}. $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-notation-P029 / L96 | Embeddings, hidden states, logits, and probabilities / The denominator sums every candidate, so the probabilities sum to one. Vocabulary size \(V\) therefore fixes b… | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-notation-P030 / L98 | Loss, gradient, and two clocks / ## Loss, gradient, and two clocks | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-notation-P031 / L100 | Loss, gradient, and two clocks / For target \(y_t\), the per-token negative log likelihood is: | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-notation-P032 / L102 | Loss, gradient, and two clocks / $$ \ell_t^{\mathrm{tok}}(\theta) = -\log p_\theta(y_t\mid x_{1:t}). $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-notation-P033 / L108 | Loss, gradient, and two clocks / With a mask \(m_t\in\{0,1\}\), the window loss is: | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-notation-P034 / L110 | Loss, gradient, and two clocks / $$ \mathcal{L} = \frac{\sum_{t=1}^{L}m_t\ell_t^{\mathrm{tok}}} {\max(\sum_{t=1}^{L}m_t,1)}. $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-notation-P035 / L117 | Loss, gradient, and two clocks / Why does the gradient become \(p-\text{one-hot}\)? For logits \(z\) and correct class \(y\): | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-notation-P036 / L119 | Loss, gradient, and two clocks / $$ \ell = -\log\frac{\exp z_y}{\sum_v\exp z_v} = -z_y+\log\sum_v\exp z_v. $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-notation-P037 / L127 | Loss, gradient, and two clocks / Differentiating component \(j\), the first term contributes \(-\mathbf{1}[j=y]\), and the log-sum-exp term con… | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-notation-P038 / L129 | Loss, gradient, and two clocks / $$ \frac{\partial\ell}{\partial z_j} = p_j-\mathbf{1}[j=y]. $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-notation-P039 / L135 | Loss, gradient, and two clocks / The correct class receives a negative gradient, so gradient descent raises its score; incorrect classes receiv… | 改写/P0 | T01、F36；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-notation-P040 / L137 | Loss, gradient, and two clocks / Static parameters are updated across training steps: | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-notation-P041 / L139 | Loss, gradient, and two clocks / $$ \theta_{k+1} = \operatorname{Opt}_{\mathrm{outer}} \left(\theta_k,\nabla_\theta\mathcal{L}_{\mathrm{outer}}… | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-notation-P042 / L146 | Loss, gradient, and two clocks / TTT fast weights can change within a sequence or chunk: | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-notation-P043 / L148 | Loss, gradient, and two clocks / $$ \phi_{t+1} = \phi_t-\eta_{\mathrm{inner}} \nabla_\phi\mathcal{L}_{\mathrm{inner}}. $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-notation-P044 / L155 | Loss, gradient, and two clocks / The indices are different clocks: \(k\) describes long-lived training state, while \(t\) describes short-lived… | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-notation-P045 / L157 | Datasets, distributions, and state owners / ## Datasets, distributions, and state owners | 修订/P2 | T21；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-notation-P046 / L159 | Datasets, distributions, and state owners / A dataset \(\mathcal{D}\) is a checkable object with files, manifests, and hashes. A distribution \(D\) is wha… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-notation-P047 / L161 | Datasets, distributions, and state owners / State also needs an owner and lifetime: | 修订/P2 | T21；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-notation-P048 / L163 | Datasets, distributions, and state owners / &#124; State &#124; Symbol &#124; Owner &#124; Lifetime &#124; &#124;---&#124;---&#124;---&#124;---&#124; &#124; static parameters &#124; \(\theta\) &#124; model/checkpoint &#124; … | 修订/P2 | T21；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-notation-P049 / L171 | Datasets, distributions, and state owners / “Model state” is too broad to audit. Loading parameters without optimizer and sampler state may succeed as a f… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-notation-P050 / L173 | Code checkpoint and evidence / ## Code checkpoint and evidence | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-notation-P051 / L175 | Code checkpoint and evidence / Notation itself does not require a GPU. On the remote CPU machine, generate a tiny token file and shard: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-notation-P052 / L177 | Code checkpoint and evidence / ~~~bash bash scripts/prepare_sample.sh outputs/sample-tokens.npy python3 -m ttt.data --input examples/sample.t… | 改写/P0 | F01；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-notation-P053 / L182 | Code checkpoint and evidence / Success should produce a token file, shard files, and &lt;code&gt;manifest.json&lt;/code&gt;. Inspect &lt;code&gt;vocab_size&lt;/co… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-notation-P054 / L184 | Summary and next question / ## Summary and next question | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-notation-P055 / L186 | Summary and next question / This chapter fixed positions, shapes, \(X/Y/M\), embeddings, hidden states, logits, probabilities, loss, gradi… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-notation-P056 / L188 | Writing rules / ## Writing rules | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-notation-P057 / L190 | Writing rules / Scalars use italic letters, vectors bold lowercase, matrices bold uppercase, and sets or losses calligraphic l… | 改写/P0 | F04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |

### docs/book/en/00_why_pretrain.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| en-00_why_pretrain-P001 / L1 | Prologue: Why train a base model from scratch / # Prologue: Why train a base model from scratch | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P002 / L3 | Overview / ## Overview | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P003 / L5 | Overview / If the goal is to ship an application, downloading a checkpoint is usually the shortest route. This book takes… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P004 / L7 | Overview / The central question is: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P005 / L9 | Overview / &gt; Under controlled data, representation, training budget, and system conditions, how do sequence models store,… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P006 / L11 | Overview / “From scratch” is a way to make the answer auditable. It does not claim that a tiny model is useful. It gives … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P007 / L13 | A concrete scene: predicting “sat” / ## A concrete scene: predicting “sat” | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P008 / L15 | A concrete scene: predicting “sat” / When a person sees “the cat”, “sat” is a plausible continuation. A model does not receive the words directly. … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P009 / L17 | A concrete scene: predicting “sat” / ~~~text text token: the cat sat &lt;EOS&gt; token ID: 17 42 31 2 ~~~ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| en-00_why_pretrain-P010 / L22 | A concrete scene: predicting “sat” / At the first position, the model receives 17 and assigns a score to every vocabulary ID. It does not return on… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P011 / L24 | Prerequisites and outcomes / ## Prerequisites and outcomes | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P012 / L26 | Prerequisites and outcomes / You need only basic arrays, functions, and command-line usage. Linear algebra, probability, and gradients are … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P013 / L28 | Prerequisites and outcomes / 1. distinguish token IDs, embeddings, hidden states, logits, and probabilities; 2. construct \(X\), \(Y\), and… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P014 / L34 | What a base model learns / ## What a base model learns | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P015 / L36 | What a base model learns / For tokenizer outputs \(x_1,\ldots,x_L\), an autoregressive model factorizes the sequence probability: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P016 / L38 | What a base model learns / $$ p_\theta(x_{1:L}) = \prod_{t=1}^{L}p_\theta(x_t\mid x_{&lt;t}). $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| en-00_why_pretrain-P017 / L44 | What a base model learns / Here \(x_{&lt;t}\) is the context before position \(t\), and \(\theta\) denotes model parameters. Training minimi… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P018 / L46 | What a base model learns / $$ \mathcal{L}(\theta) = -\frac{1}{L} \sum_{t=1}^{L} \log p_\theta(x_t\mid x_{&lt;t}). $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| en-00_why_pretrain-P019 / L54 | What a base model learns / The task is local, but useful prediction may require syntax, code dependencies, discourse, or facts from far e… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P020 / L56 | What a base model learns / It does not guarantee reasoning, factuality, or instruction following. Duplicated training data can lower loss… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P021 / L58 | From logits to probability, loss, and gradient / ## From logits to probability, loss, and gradient | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P022 / L60 | From logits to probability, loss, and gradient / Suppose the context is “the cat” and the correct next token is “sat”. For three candidates, let the logits be | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P023 / L62 | From logits to probability, loss, and gradient / $$ z=(2,1,0). $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-00_why_pretrain-P024 / L66 | From logits to probability, loss, and gradient / Softmax is: | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P025 / L68 | From logits to probability, loss, and gradient / $$ p_j=\frac{\exp z_j}{\sum_u\exp z_u}. $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-00_why_pretrain-P026 / L72 | From logits to probability, loss, and gradient / The probabilities are approximately \((0.665,0.245,0.090)\). If the correct token is the second candidate, its… | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P027 / L74 | From logits to probability, loss, and gradient / $$ \ell=-\log(0.245)\approx1.407. $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-00_why_pretrain-P028 / L78 | From logits to probability, loss, and gradient / If the correct probability rises to \(0.8\), the contribution becomes \(-\log(0.8)\approx0.223\). Thus “loss w… | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P029 / L80 | From logits to probability, loss, and gradient / Expanding the loss: | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P030 / L82 | From logits to probability, loss, and gradient / $$ \ell = -\log\frac{\exp z_y}{\sum_v\exp z_v} = -z_y+\log\sum_v\exp z_v. $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-00_why_pretrain-P031 / L90 | From logits to probability, loss, and gradient / Differentiating component \(j\): | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P032 / L92 | From logits to probability, loss, and gradient / $$ \frac{\partial\ell}{\partial z_j} = p_j-\mathbf{1}[j=y]. $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-00_why_pretrain-P033 / L98 | From logits to probability, loss, and gradient / The correct class has a negative gradient, so gradient descent raises its score. Incorrect classes have positi… | 改写/P0 | T01、F36；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P034 / L100 | From logits to probability, loss, and gradient / With padding or prompt masks, only \(m_t=1\) positions contribute: | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P035 / L102 | From logits to probability, loss, and gradient / $$ \mathcal{L} = \frac{\sum_{t=1}^{L} m_t[-\log p_\theta(y_t\mid x_{\le t})]} {\max(\sum_{t=1}^{L}m_t,1)}. $$ | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-00_why_pretrain-P036 / L110 | From logits to probability, loss, and gradient / A wrong input-label shift trains a different task. Counting padding trains the model to predict padding. Those… | 补写/P1 | T01；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P037 / L112 | Why train from scratch / ## Why train from scratch | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P038 / L114 | Why train from scratch / Existing checkpoints are often the right choice for applications. From-scratch training is valuable when we ne… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P039 / L116 | Why train from scratch / The cost is real: licensing, cleaning, storage, stability, GPU scheduling, communication, and recovery become … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P040 / L118 | The six coupled parts / ## The six coupled parts | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P041 / L120 | The six coupled parts / &#124; Part &#124; Question &#124; Typical failure &#124; &#124;---&#124;---&#124;---&#124; &#124; Data &#124; What distribution did the model actually see? &#124; n… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P042 / L129 | The six coupled parts / Changing vocabulary size \(V\) changes the embedding and output head. Changing sequence length \(L\) changes a… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P043 / L131 | Budget: parameters, tokens, throughput, and wall time / ## Budget: parameters, tokens, throughput, and wall time | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P044 / L133 | Budget: parameters, tokens, throughput, and wall time / Let \(N_{\mathrm{param}}\) be trainable parameters, \(N_{\mathrm{tok}}\) consumed tokens, and \(r_{\mathrm{tok… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P045 / L135 | Budget: parameters, tokens, throughput, and wall time / $$ F_{\mathrm{rough}} \approx 6N_{\mathrm{param}}N_{\mathrm{tok}}. $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| en-00_why_pretrain-P046 / L141 | Budget: parameters, tokens, throughput, and wall time / This does not fully cover long-context attention, special kernels, communication, or TTT inner updates. It can… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P047 / L143 | Budget: parameters, tokens, throughput, and wall time / &#124; Quantity &#124; Example &#124; Meaning &#124; &#124;---&#124;---:&#124;---&#124; &#124; training tokens &#124; \(10^9\) &#124; a budget, not file bytes &#124; &#124; st… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P048 / L150 | Budget: parameters, tokens, throughput, and wall time / Model and data budgets must be reported together. With \(d=768\), increasing vocabulary from 32K to 128K incre… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P049 / L152 | First runnable checkpoint / ## First runnable checkpoint | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P050 / L154 | First runnable checkpoint / On the designated remote CPU machine, create the tiny data artifacts: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P051 / L156 | First runnable checkpoint / ~~~bash bash scripts/prepare_sample.sh outputs/sample-tokens.npy python3 -m ttt.data \ --input examples/sample… | 改写/P0 | F01；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-00_why_pretrain-P052 / L166 | First runnable checkpoint / You should see &lt;code&gt;outputs/sample-tokens.npy&lt;/code&gt;, shard files, and &lt;code&gt;manifest.json&lt;/code&gt;. Check &lt;cod… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P053 / L168 | First runnable checkpoint / The book uses three completion levels: **runnable** means the entry point and dependencies work; **learnable**… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P054 / L170 | Mainline and Frontier Lab / ## Mainline and Frontier Lab | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P055 / L172 | Mainline and Frontier Lab / The mainline follows dependencies: notation, experiment contract, data, dense baseline, state and compute, abl… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P056 / L174 | Mainline and Frontier Lab / A frontier page may propose a hypothesis, but it cannot reuse a paper's numbers as this project's result. It e… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P057 / L176 | Summary and next question / ## Summary and next question | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-00_why_pretrain-P058 / L178 | Summary and next question / This chapter established why we train, what the model predicts, how softmax/NLL/gradients connect, and how bud… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/en/01_experiment_contract.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| en-01_experiment_contract-P001 / L1 | 1. The Experiment Contract: Making a Run Interpretable / # 1. The Experiment Contract: Making a Run Interpretable | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P002 / L3 | Overview: turn a comparison into a question / ## Overview: turn a comparison into a question | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P003 / L5 | Overview: turn a comparison into a question / Suppose we train the same tiny model twice: once with the TTT inner loop disabled and once with it enabled. A … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P004 / L7 | Overview: turn a comparison into a question / After this chapter, you should be able to fill in the research variable, fixed variables, budget, and evidence… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P005 / L9 | Running example: tiny-contract-001 / ## Running example: tiny-contract-001 | 改写/P0 | F05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P006 / L11 | Running example: tiny-contract-001 / We use &lt;code&gt;tiny-contract-001&lt;/code&gt; throughout. It reads &lt;code&gt;configs/tiny.json&lt;/code&gt; and byte-token shard… | 改写/P0 | F05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P007 / L13 | Running example: tiny-contract-001 / &#124; Field &#124; Control &#124; Treatment &#124; &#124;---&#124;---&#124;---&#124; &#124; Run identity &#124; &lt;code&gt;tiny-contract-001-C&lt;/code&gt; &#124; &lt;code&gt;tiny-c… | 改写/P0 | F06、F35、F05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P008 / L21 | Running example: tiny-contract-001 / Write this table before looking at a result. It prevents the comparison rule from changing after the numbers a… | 改写/P0 | F05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P009 / L23 | Decide what the experiment is meant to prove / ## Decide what the experiment is meant to prove | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P010 / L25 | Decide what the experiment is meant to prove / This book does not treat “the code ran” as a research conclusion. The question is whether, under the same data… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P011 / L27 | Decide what the experiment is meant to prove / Before starting a run, write one claim in a fixed form: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P012 / L29 | Decide what the experiment is meant to prove / &gt; Holding ______ fixed, change only ______, and measure whether ______ changes. | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P013 / L31 | Decide what the experiment is meant to prove / For example: hold the tokenizer, effective training tokens, sequence length, optimizer, and hardware fixed; en… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P014 / L33 | The identity of a run / ## The identity of a run | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P015 / L35 | The identity of a run / A run is more than a command. Treat it as an object with an identity: | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P016 / L37 | The identity of a run / $$ R=(S,C,I_{\mathrm{data}},E,K,Q_{\mathrm{budget}},O), $$ | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-01_experiment_contract-P017 / L41 | The identity of a run / where: | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P018 / L43 | The identity of a run / - \(S\): source commit; - \(C\): parsed model and training configuration; - \(I_{\mathrm{data}}\): data, token… | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P019 / L51 | The identity of a run / Reproduction does not always mean bitwise identical floating-point values. It means that the objects defining … | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P020 / L53 | Variables that must be fixed or reported / ## Variables that must be fixed or reported | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P021 / L55 | Variables that must be fixed or reported / &#124; Category &#124; What to record &#124; &#124;---&#124;---&#124; &#124; Data &#124; sources, cleaning version, deduplication, split manifests, ef… | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P022 / L65 | Variables that must be fixed or reported / A comparison may change one of these categories, but the changed item must be named as the research variable. … | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P023 / L67 | Variables that must be fixed or reported / A non-attributable example is a control with a byte tokenizer and sequence length 16, compared with a treatmen… | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P024 / L69 | Three levels of evidence / ## Three levels of evidence | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P025 / L71 | Three levels of evidence / Report every run at three levels. | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P026 / L73 | Three levels of evidence / **Runnable** means that dependencies, devices, configuration, and data entry points are valid and the program … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P027 / L75 | Three levels of evidence / **Learnable** means that loss and gradients are finite, training state changes, checkpoints can be loaded, and… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P028 / L77 | Three levels of evidence / **Interpretable** means that the control and treatment differ only in the declared variable, evaluation uses a… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P029 / L79 | Three levels of evidence / A four-step tiny run can establish that the interface and optimization path are connected. It does not establi… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P030 / L81 | The run directory must explain itself / ## The run directory must explain itself | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P031 / L83 | The run directory must explain itself / A formal run should preserve at least: | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P032 / L85 | The run directory must explain itself / ~~~text experiment.json # inputs, parsed configuration, and budget source.json # commit and worktree state env… | 改写/P0 | F06、T03；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-01_experiment_contract-P033 / L96 | The run directory must explain itself / JSON, JSONL, and checkpoints are the factual source. The Markdown report is the reading interface; every numbe… | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P034 / L98 | The run directory must explain itself / For &lt;code&gt;tiny-contract-001&lt;/code&gt;, the reader should be able to trace each claim to one artifact: | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P035 / L100 | The run directory must explain itself / &#124; Claim &#124; Primary artifact &#124; Passing condition &#124; &#124;---&#124;---&#124;---&#124; &#124; The intended configuration ran &#124; &lt;code&gt;experi… | 改写/P0 | F06、T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P036 / L108 | Resume is different from initialize-from / ## Resume is different from initialize-from | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P037 / L110 | Resume is different from initialize-from / `resume` continues the same optimization process. It must restore model parameters, optimizer state, global st… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P038 / L112 | Resume is different from initialize-from / `initialize-from` uses old parameters to start a new experiment. It receives a new run identity and may create… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P039 / L114 | Resume is different from initialize-from / Loading parameters alone is not the same as resuming training. | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P040 / L116 | Failures are evidence / ## Failures are evidence | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P041 / L118 | Failures are evidence / Failures may occur during installation, device discovery, XLA compilation, data loading, gradient updates, che… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P042 / L120 | Failures are evidence / The purpose of a failure record is to reduce the unknowns in the next run, not to make the project look uninte… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P043 / L122 | Bound every conclusion / ## Bound every conclusion | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P044 / L124 | Bound every conclusion / Every report should answer: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P045 / L126 | Bound every conclusion / 1. What actually changed? 2. Which measurements changed? 3. Can those changes be attributed to the declared va… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P046 / L132 | Bound every conclusion / Use the following form when needed: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P047 / L134 | Bound every conclusion / &gt; In the tiny setting, holding \(A\), \(B\), and \(C\) fixed, changing \(D\) produced \(E\). This supports \(F… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P048 / L136 | Bound every conclusion / The environment chapter answers where a run executes; the data and representation chapters answer what the mod… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P049 / L138 | Summary / ## Summary | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-01_experiment_contract-P050 / L140 | Summary / - A run identity contains source, resolved configuration, data fingerprints, environment, random and optimizer… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/en/01_environment.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| en-01_environment-P001 / L1 | 2. Environment: the first trusted training run / # 2. Environment: the first trusted training run | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-01_environment-P002 / L3 | Overview: prove that the machine can run / ## Overview: prove that the machine can run | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-01_environment-P003 / L5 | Overview: prove that the machine can run / This chapter does not try to train a large model in one command. We first make a smaller claim for &lt;code&gt;tiny-… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_environment-P004 / L7 | Overview: prove that the machine can run / After this chapter, you should be able to read the JAX backend and device list, inspect the length and vocabul… | 改写/P0 | F06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_environment-P005 / L9 | Prerequisites and running input / ## Prerequisites and running input | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-01_environment-P006 / L11 | Prerequisites and running input / You only need a terminal, basic shell commands, and the idea that JSON files in an output directory are machin… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_environment-P007 / L13 | Why the environment belongs to the experiment / ## Why the environment belongs to the experiment | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-01_environment-P008 / L15 | Why the environment belongs to the experiment / The first failure in a from-scratch project is often not a wrong equation. It is a Python version, a missing C… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_environment-P009 / L17 | Why the environment belongs to the experiment / Think of the environment as the laboratory bench. The model is the instrument, the dataset is the material, an… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_environment-P010 / L19 | The three layers of a machine contract / ## The three layers of a machine contract | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-01_environment-P011 / L21 | The three layers of a machine contract / We record three layers because each one answers a different debugging question: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_environment-P012 / L23 | The three layers of a machine contract / &#124; Layer &#124; Beginner question &#124; Typical evidence &#124; &#124;---&#124;---&#124;---&#124; &#124; System &#124; Can this machine see the requested d… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_environment-P013 / L29 | The three layers of a machine contract / The layers are related but cannot replace one another. A correct Python installation does not guarantee enough… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_environment-P014 / L31 | What the repository provides / ## What the repository provides | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-01_environment-P015 / L33 | What the repository provides / The shell scripts are thin entry points; Python owns the experiment semantics so the same run can be submitted… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_environment-P016 / L35 | What the repository provides / - `scripts/bootstrap_linux.sh` installs the pinned Python and dependencies. Setting `TTT_INSTALL_CUDA=1` selec… | 改写/P0 | F06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_environment-P017 / L42 | What the repository provides / The sample file is a wiring test, not a language benchmark. It can reveal an out-of-range token, an impossible… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_environment-P018 / L44 | What the repository provides / For the tiny configuration, the first sample-data check should answer three questions: are there at least &lt;cod… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_environment-P019 / L46 | A minimal first run / ## A minimal first run | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_environment-P020 / L48 | A minimal first run / Run the following on the target Linux machine or inside a GPU allocation. For a published result, replace `mai… | 补写/P1 | T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_environment-P021 / L50 | A minimal first run / ~~~bash git clone https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch.git cd Modern-Sequence-M… | 改写/P0 | F07、T03；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-01_environment-P022 / L66 | A minimal first run / On success, the environment JSON should contain Python and JAX versions, the backend, visible device count, an… | 改写/P0 | F07、T03、F06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_environment-P023 / L68 | A minimal first run / On a CPU-only validation node, use `TTT_INSTALL_CUDA=0` and omit `--require-gpu`. CPU runs validate data paths… | 改写/P0 | F07、T03；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_environment-P024 / L70 | What happens during the first step / ## What happens during the first step | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-01_environment-P025 / L72 | What happens during the first step / “The command started” does not mean that steady-state training has begun. A first step usually contains four s… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_environment-P026 / L74 | What happens during the first step / 1. parse the JSON and fill default values; 2. initialize parameters and derive random keys; 3. trace the JAX f… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_environment-P027 / L79 | What happens during the first step / If the first step takes 40 seconds and later steps take 2 seconds, averaging those five steps would report a m… | 改写/P0 | F06、F35；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_environment-P028 / L81 | The expected evidence directory / ## The expected evidence directory | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-01_environment-P029 / L83 | The expected evidence directory / A successful tiny run should leave evidence similar to: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_environment-P030 / L85 | The expected evidence directory / ~~~text experiment.json # expanded configuration, commit, data identity, seed environment.json # Python, JAX b… | 改写/P0 | F06；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-01_environment-P031 / L94 | The expected evidence directory / The exact checkpoint filename can change, but the meaning of the records must not. Completion means that the l… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_environment-P032 / L96 | Reading a failure without guessing / ## Reading a failure without guessing | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-01_environment-P033 / L98 | Reading a failure without guessing / If `check_environment.py` cannot find a GPU, inspect the backend and device list before changing the model. If… | 改写/P0 | F06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_environment-P034 / L100 | Reading a failure without guessing / If a resumed run diverges, compare the data fingerprint, sampler position, optimizer state, and global step be… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_environment-P035 / L102 | Reading a failure without guessing / The next chapter turns this directory into a formal experiment identity: the environment answers **where** the… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_environment-P036 / L104 | Minimal checkpoint and expected evidence / ## Minimal checkpoint and expected evidence | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-01_environment-P037 / L106 | Minimal checkpoint and expected evidence / On a node with the dependencies installed, the shortest CPU path is: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_environment-P038 / L108 | Minimal checkpoint and expected evidence / ~~~bash bash scripts/prepare_sample.sh outputs/sample-tokens.npy JAX_PLATFORMS=cpu bash scripts/run_train.sh \… | 改写/P0 | F05；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-01_environment-P039 / L116 | Minimal checkpoint and expected evidence / On a GPU node, use &lt;code&gt;bash scripts/run_gpu_train.sh&lt;/code&gt; after the device check. A passing directory cont… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-01_environment-P040 / L118 | Minimal checkpoint and expected evidence / If the command fails, inspect the sample-token path, the parsed &lt;code&gt;configs/tiny.json&lt;/code&gt;, and the backen… | 改写/P0 | F06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-01_environment-P041 / L120 | Summary / ## Summary | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-01_environment-P042 / L122 | Summary / - System, Python, and experiment layers need separate records. - Sample tokens first validate vocabulary range… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/en/02_reproducible_runs.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| en-02_reproducible_runs-P001 / L1 | 3. Repositories and reproducible runs / # 3. Repositories and reproducible runs | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P002 / L3 | Overview: interrupt a tiny run and prove that it continues / ## Overview: interrupt a tiny run and prove that it continues | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P003 / L5 | Overview: interrupt a tiny run and prove that it continues / “The checkpoint opens” is not a recovery experiment. We need to show that an uninterrupted run and an interrup… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P004 / L7 | Overview: interrupt a tiny run and prove that it continues / After this chapter, you should be able to distinguish &lt;code&gt;--resume&lt;/code&gt; from &lt;code&gt;--initialize-from&lt;/code… | 改写/P0 | F06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P005 / L9 | Prerequisites and comparison table / ## Prerequisites and comparison table | 补写/P1 | T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P006 / L11 | Prerequisites and comparison table / Suppose a tiny run saves at step 2. One path continues to step 4; the other resumes from step 2 and also conti… | 补写/P1 | T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P007 / L13 | Prerequisites and comparison table / &#124; Check &#124; Uninterrupted &#124; Resumed &#124; Passing condition &#124; &#124;---&#124;---:&#124;---:&#124;---&#124; &#124; Starting point &#124; step 0 &#124; step 2… | 改写/P0 | F35、F09、T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P008 / L21 | Prerequisites and comparison table / Recovery is therefore a state comparison, not a file-existence check. | 补写/P1 | T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P009 / L23 | A run is an object / ## A run is an object | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P010 / L25 | A run is an object / A command proves that a program started. It does not explain why the result looks the way it does. A reproduci… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P011 / L27 | A run is an object / We write that identity as | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P012 / L29 | A run is an object / $$ R=(S,C,I_{\mathrm{data}},E,K,Q_{\mathrm{budget}},O), $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| en-02_reproducible_runs-P013 / L33 | A run is an object / where $S$ is the source commit, $C$ the resolved configuration, $I_{\mathrm{data}}$ the fingerprints of data, … | 改写/P0 | F06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P014 / L35 | A run is an object / ~~~text experiment id ├── source: git commit ├── config: model / optimizer / data ├── input: manifest and SHA2… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| en-02_reproducible_runs-P015 / L45 | A run is an object / Without one of these fields, reproduction becomes an approximation. | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P016 / L47 | Resume semantics / ## Resume semantics | 补写/P1 | T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P017 / L49 | Resume semantics / resume restores parameters, optimizer state, consumed-data position, and random state. It is for continuing an… | 补写/P1 | T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P018 / L51 | Resume semantics / initialize-from loads parameters but creates a new optimizer and experiment identity. It is useful when initia… | 改写/P0 | F08、T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P019 / L53 | A small failure case / ## A small failure case | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P020 / L55 | A small failure case / Suppose a checkpoint was written at global step 100. Its parameters have seen 100 batches, and AdamW has accum… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P021 / L57 | A small failure case / 1. the optimizer moments are zero, so the update direction changes abruptly; 2. the schedule may restart at st… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P022 / L62 | A small failure case / “The checkpoint opens” is therefore not a recovery test. Use the same short run in two paths: one uninterrupte… | 改写/P0 | F35；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P023 / L64 | A small failure case / The opposite operation, initializing TTT from a dense baseline, creates a new identity: | 改写/P0 | F08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P024 / L66 | A small failure case / ~~~text dense baseline: source=S1, data=D1, step=1000, optimizer=O_dense TTT initialize: source=S2, data=D1, s… | 改写/P0 | F08；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-02_reproducible_runs-P025 / L71 | A small failure case / The two runs share an initial parameter file but are not one continuous curve. A report must not append the se… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P026 / L73 | Code and artifacts / ## Code and artifacts | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P027 / L75 | Code and artifacts / Even a failure during environment setup should leave a structured directory: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P028 / L77 | Code and artifacts / &#124; File or field &#124; Purpose &#124; &#124;---&#124;---&#124; &#124; status.json &#124; lifecycle such as running, failed, completed, or resumab… | 改写/P0 | F06、F35；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P029 / L86 | Code and artifacts / This lets a reader distinguish an unavailable device, an invalid configuration, and a run that diverged after … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P030 / L88 | Code and artifacts / - ttt/provenance.py records Python, JAX, device, driver, PBS, and source information. - scripts/record_environ… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P031 / L93 | Code and artifacts / Machine-readable JSONL and status files are the source of truth. Markdown reports should reference them instea… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P032 / L95 | Code and artifacts / ~~~bash git rev-parse HEAD bash scripts/record_environment.sh --output outputs/run/environment.json sha256sum … | 改写/P0 | F31；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-02_reproducible_runs-P033 / L107 | Code and artifacts / Failures should leave their output directory and error status. A failed compile or incompatible checkpoint is … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P034 / L109 | Transition / ## Transition | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P035 / L111 | Transition / Two runs are comparable only when tokens, tokenizer, sequence length, effective batch, optimizer, seed, archit… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P036 / L113 | Minimal recovery checkpoint / ## Minimal recovery checkpoint | 补写/P1 | F10、T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P037 / L115 | Minimal recovery checkpoint / Generate sample data and start a tiny run that writes intermediate state: | 补写/P1 | F10、T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P038 / L117 | Minimal recovery checkpoint / ~~~bash bash scripts/prepare_sample.sh outputs/sample-tokens.npy bash scripts/run_train.sh \ --config configs/… | 改写/P0 | F05、F10、T04；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-02_reproducible_runs-P039 / L126 | Minimal recovery checkpoint / On success, inspect &lt;code&gt;outputs/tiny-contract-001-C/status.json&lt;/code&gt;, &lt;code&gt;metrics.jsonl&lt;/code&gt;, &lt;code&gt;la… | 补写/P1 | F10、T04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P040 / L128 | Summary / ## Summary | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-02_reproducible_runs-P041 / L130 | Summary / - Resume continues one optimization process; initialize-from creates a new experiment identity. - The random s… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/en/03_configuration.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| en-03_configuration-P001 / L1 | 14. Configuration: one contract for every tensor shape / # 14. Configuration: one contract for every tensor shape | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-03_configuration-P002 / L3 | Overview: give every number a source / ## Overview: give every number a source | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-03_configuration-P003 / L5 | Overview: give every number a source / In &lt;code&gt;tiny-contract-001&lt;/code&gt;, the vocabulary has 259 entries, the residual width is 16, and attention has… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-03_configuration-P004 / L7 | Overview: give every number a source / After this chapter, you should be able to load a configuration, list its resolved defaults, hand-calculate a b… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-03_configuration-P005 / L9 | Prerequisites and tiny target / ## Prerequisites and tiny target | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-03_configuration-P006 / L11 | Prerequisites and tiny target / You only need the idea that a matrix maps one collection of numbers to another. The running input is &lt;code&gt;inp… | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-03_configuration-P007 / L13 | Why configuration must be explicit / ## Why configuration must be explicit | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-03_configuration-P008 / L15 | Why configuration must be explicit / Suppose a model silently chooses the number of heads from a tensor shape inside `forward`. The JSON may say `h… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-03_configuration-P009 / L17 | Vocabulary fixes the output space / ## Vocabulary fixes the output space | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-03_configuration-P010 / L19 | Vocabulary fixes the output space / Let \(V\) be the tokenizer vocabulary size and \(d\) the hidden width. The input embedding is | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-03_configuration-P011 / L21 | Vocabulary fixes the output space / $$ E\in\mathbb{R}^{V\times d}. $$ | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-03_configuration-P012 / L25 | Vocabulary fixes the output space / At each position the residual stream is a vector in \(\mathbb{R}^{d}\). The MLP can temporarily use an interme… | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-03_configuration-P013 / L27 | Vocabulary fixes the output space / $$ \mathbf z_{b,t}=\mathbf h_{b,t}W_{\mathrm{out}},\qquad W_{\mathrm{out}}\in\mathbb{R}^{d\times V}. $$ | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-03_configuration-P014 / L32 | Vocabulary fixes the output space / The current implementation uses `bias=False`. With tied embeddings, \(W_{\mathrm{out}}=E^{\mathsf T}\); withou… | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-03_configuration-P015 / L34 | Vocabulary fixes the output space / For example, if \(V=259\) and \(d=16\), the tied embedding/head has \(259\times16=4{,}144\) scalar entries. A … | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-03_configuration-P016 / L36 | Four groups of fields / ## Four groups of fields | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-03_configuration-P017 / L38 | Four groups of fields / &#124; Group &#124; Examples &#124; What changes &#124; &#124;---&#124;---&#124;---&#124; &#124; Model size &#124; `hidden_size`, `num_layers`, `num_attention_h… | 改写/P0 | F04；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-03_configuration-P018 / L45 | Four groups of fields / The control \(C_k\) and adaptive variant \(F_k\) must use the same static architecture. \(C_k\) disables test-… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-03_configuration-P019 / L47 | Four groups of fields / Sequence fields such as `max_position_embeddings`, `sliding_window`, and `ttt_chunk_size` are also architectur… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-03_configuration-P020 / L49 | How the code consumes the contract / ## How the code consumes the contract | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-03_configuration-P021 / L51 | How the code consumes the contract / `ttt/models/ttt_e2e/configuration_ttt_e2e.py` defines model, MLP-prime, MoE, loop, and dtype fields. `ttt/mode… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-03_configuration-P022 / L53 | How the code consumes the contract / ~~~python from ttt.models import AutoConfig, AutoModelForCausalLM config = AutoConfig.from_pretrained("configs… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| en-03_configuration-P023 / L60 | How the code consumes the contract / Loading should validate positive dimensions, head divisibility, vocabulary ranges, layer positions, special-to… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-03_configuration-P024 / L62 | Defaults and early errors / ## Defaults and early errors | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-03_configuration-P025 / L64 | Defaults and early errors / Some fields are derived. `num_key_value_heads=None` defaults to the query-head count, `head_dim=None` is deriv… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-03_configuration-P026 / L66 | Defaults and early errors / Examples of errors that should happen during Python initialization: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-03_configuration-P027 / L68 | Defaults and early errors / - `hidden_size` is not divisible by `num_attention_heads`, so a head width cannot be defined; - query heads ar… | 改写/P0 | F11；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-03_configuration-P028 / L74 | Defaults and early errors / An early `ValueError` is cheaper and easier to diagnose than a compiler failure after several minutes. | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-03_configuration-P029 / L76 | A tiny shape audit / ## A tiny shape audit | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-03_configuration-P030 / L78 | A tiny shape audit / Before launching training, write down the resolved values and trace one token: | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-03_configuration-P031 / L80 | A tiny shape audit / ~~~text vocab_size = 259 hidden_size = 16 query_heads = 2, kv_heads = 1 layers = 2 ttt_suffix_length = 1 fast_… | 改写/P0 | F04、T05；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-03_configuration-P032 / L91 | A tiny shape audit / Then draw `embedding -&gt; attention -&gt; MLP/MLP-prime -&gt; residual -&gt; lm_head` and annotate each shape. If one num… | 补写/P1 | T05；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-03_configuration-P033 / L92 | Minimal configuration audit / ## Minimal configuration audit | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-03_configuration-P034 / L94 | Minimal configuration audit / On the remote node, parse the resolved configuration before launching training: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-03_configuration-P035 / L96 | Minimal configuration audit / ~~~bash python3 -c "from ttt.config import load_experiment; import json; print(json.dumps(load_experiment('con… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| en-03_configuration-P036 / L100 | Minimal configuration audit / The output should contain &lt;code&gt;model&lt;/code&gt;, &lt;code&gt;inner&lt;/code&gt;, &lt;code&gt;training&lt;/code&gt;, and &lt;code&gt;tokenizer&lt;/… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-03_configuration-P037 / L102 | Summary / ## Summary | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-03_configuration-P038 / L104 | Summary / - The configuration contract fixes parameter shapes, state boundaries, cost, and comparison fairness. - \(V\) … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/en/04_forward_and_ttt.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| en-04_forward_and_ttt-P001 / L1 | 16. Forward order: how one token travels through TTT-E2E / # 16. Forward order: how one token travels through TTT-E2E | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P002 / L3 | Do not start with a kernel / ## Do not start with a kernel | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P003 / L5 | Do not start with a kernel / TTT-E2E contains several new terms: prefix, suffix, fast weights, inner loss, MLP-prime, and KV cache. The saf… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P004 / L7 | Do not start with a kernel / The model first looks up a token ID in an embedding table. Each block then sends the hidden state through atte… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P005 / L9 | Do not start with a kernel / ~~~text token ids ↓ embedding ↓ attention / gated delta + residual ↓ fast MLP-prime + residual (selected later… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| en-04_forward_and_ttt-P006 / L23 | Do not start with a kernel / Every arrow is an interface for a shape, a mask, a cache, and a gradient. Moving MLP-prime before or after the… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P007 / L25 | Do not start with a kernel / &lt;figure class="book-figure"&gt; &lt;img src="../../../assets/figures/ttt-timeline.svg" alt="The timing of token chun… | 补写/P1 | T20；具体修复、证据及双语问题见上表 | SVG本体与网页未看 |
| en-04_forward_and_ttt-P008 / L30 | The order implemented in this repository / ## The order implemented in this repository | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P009 / L32 | The order implemented in this repository / The forward path in &lt;code&gt;TTTE2EForCausalLM._block&lt;/code&gt; first executes the attention branch. A layer whose t… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P010 / L34 | The order implemented in this repository / If the configuration contains &lt;code&gt;fast_mlp&lt;/code&gt;, the block then reads the current fast-weight version of M… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P011 / L36 | The order implemented in this repository / The conceptual order is therefore: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P012 / L38 | The order implemented in this repository / ~~~text attention / gated delta → fast MLP-prime → static MLP or MoE ~~~ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| en-04_forward_and_ttt-P013 / L44 | The order implemented in this repository / This is an additional adaptation path, not a replacement of the original feed-forward network. If fast weights… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P014 / L46 | Prefix and suffix / ## Prefix and suffix | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P015 / L48 | Prefix and suffix / The configuration divides the block stack into a prefix and a suffix. The prefix runs the static part once and… | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P016 / L50 | Prefix and suffix / This division avoids repeating all static computation inside the inner loop, but it creates an alignment requi… | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P017 / L52 | Prefix and suffix / In the current training path, &lt;code&gt;sequence_loss&lt;/code&gt; pads a token sequence to a multiple of &lt;code&gt;ttt_chun… | 改写/P0 | F14、T06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P018 / L54 | Why fast weights are not ordinary parameters / ## Why fast weights are not ordinary parameters | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P019 / L56 | Why fast weights are not ordinary parameters / Ordinary parameters are shared by many examples and updated by the outer optimizer. They survive from one trai… | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P020 / L58 | Why fast weights are not ordinary parameters / Use two time indices to keep the clocks separate: | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P021 / L60 | Why fast weights are not ordinary parameters / $$ \theta_{k+1} = \operatorname{AdamW}\!\left( \theta_k,\, \nabla_\theta \mathcal{L}_{\mathrm{outer}} \right),… | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-04_forward_and_ttt-P022 / L69 | Why fast weights are not ordinary parameters / $$ \phi_{t+1} = \phi_t -\eta_{\mathrm{inner}} \nabla_\phi \mathcal{L}_{\mathrm{inner}} \left(x_t;\theta,\phi_t… | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-04_forward_and_ttt-P023 / L79 | Why fast weights are not ordinary parameters / Here \( \theta \) denotes static model parameters and \( \phi \) denotes fast weights. The outer loop learns h… | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P024 / L81 | Prefix prediction, label use, and causality / ## Prefix prediction, label use, and causality | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P025 / L83 | Prefix prediction, label use, and causality / A safe causal update has this order: | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P026 / L85 | Prefix prediction, label use, and causality / ~~~text use fast_t to predict y_t → compute inner loss from the observed label → update to fast_{t+1} → let la… | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-04_forward_and_ttt-P027 / L92 | Prefix prediction, label use, and causality / The label used for an inner update must be available at the point where the update happens. If a suffix label … | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P028 / L94 | Prefix prediction, label use, and causality / The test is simple to state. Change a later suffix token while keeping the prefix fixed, then compare the logi… | 补写/P1 | T06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P029 / L96 | Prefix prediction, label use, and causality / The current inner step computes a value and gradient with respect to fast weights, then applies &lt;code&gt;sgd_upda… | 改写/P0 | F15、T06；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P030 / L98 | A small chunk calculation / ## A small chunk calculation | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P031 / L100 | A small chunk calculation / Let the sequence length be \(8\) and the chunk size be \(4\). The numbers \(0\) through \(7\) below are code a… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P032 / L102 | A small chunk calculation / ~~~text chunk 0: positions 0, 1, 2, 3 chunk 1: positions 4, 5, 6, 7 ~~~ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| en-04_forward_and_ttt-P033 / L107 | A small chunk calculation / Chunk 0 predicts with \( \phi_0 \), obtains \( \mathcal{L}_0 \), and updates to \( \phi_1 \). Chunk 1 predicts… | 改写/P0 | F14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P034 / L109 | A small chunk calculation / This is why chunk size is an algorithmic variable, not merely a batching detail. A smaller chunk lets the fast… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P035 / L111 | How to validate that the forward pass is semantically correct / ## How to validate that the forward pass is semantically correct | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P036 / L113 | How to validate that the forward pass is semantically correct / A useful validation set contains at least four checks: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P037 / L115 | How to validate that the forward pass is semantically correct / 1. Compare the dense path with fast MLP disabled, including shapes and numerical values. 2. Change suffix toke… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P038 / L120 | How to validate that the forward pass is semantically correct / The GPU validation script writes architecture and runtime reports. Every report must be tied to the configurat… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P039 / L122 | How to validate that the forward pass is semantically correct / The next chapter turns this forward order into an explicit inner loop and outer loop, and follows the loss thr… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P040 / L124 | Shape walk for one token / ## Shape walk for one token | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P041 / L126 | Shape walk for one token / Use the tiny configuration with \(d=16\), \(V=259\), batch \(B=1\), and \(L=4\) positions: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P042 / L128 | Shape walk for one token / &#124; Stage &#124; Code object &#124; Shape &#124; Check &#124; &#124;---&#124;---&#124;---&#124;---&#124; &#124; Input &#124; &lt;code&gt;input_ids&lt;/code&gt; &#124; \([1,4]\) &#124; every… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P043 / L136 | Shape walk for one token / The last column matters as much as the shape: it explains why each interface has that shape. | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P044 / L138 | A numerical causality assertion / ## A numerical causality assertion | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P045 / L140 | A numerical causality assertion / Let code indices 0 and 1 be the prefix and 2 and 3 be the suffix. Run twice with the same initial parameters a… | 改写/P0 | F12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P046 / L142 | A numerical causality assertion / ~~~text max_abs_diff(logits_a[:, 0:2, :], logits_b[:, 0:2, :]) &lt;= 1e-6 max_abs_diff(logits_a[:, 2:, :], logits… | 改写/P0 | F12；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-04_forward_and_ttt-P047 / L147 | A numerical causality assertion / The first line failing indicates future information reached the prefix. The second line may change because the… | 改写/P0 | F12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P048 / L149 | Minimal validation / ## Minimal validation | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P049 / L151 | Minimal validation / On a remote GPU node, run &lt;code&gt;bash scripts/validate_gpu.sh outputs/gpu-validation.json&lt;/code&gt;. The report sh… | 改写/P0 | F13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P050 / L153 | Summary / ## Summary | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-04_forward_and_ttt-P051 / L155 | Summary / - Forward order determines what each residual path and state can observe. - Fast weights are sequence or sessi… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/en/10_data_contract.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| en-10_data_contract-P001 / L1 | 4. Pretraining Data: Distribution, Quality, and Reproducibility / # 4. Pretraining Data: Distribution, Quality, and Reproducibility | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P002 / L3 | Why data is the first design decision / ## Why data is the first design decision | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-10_data_contract-P003 / L5 | Why data is the first design decision / Pretraining is not a matter of putting a pile of text on a GPU and waiting for the model to absorb “the intern… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P004 / L7 | Why data is the first design decision / Data changes the result in at least three ways: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P005 / L9 | Why data is the first design decision / - **Capability coverage.** Web pages, code, books, papers, and conversations contain different vocabularies, s… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P006 / L13 | Why data is the first design decision / This small example shows why file counts do not describe the training distribution: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P007 / L15 | Why data is the first design decision / &#124; Source &#124; Documents &#124; Mean tokens &#124; Share after direct token concatenation &#124; Possible bias &#124; &#124;---&#124;---:&#124;---:&#124;-… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P008 / L21 | Overview: trace one short text first / ## Overview: trace one short text first | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-10_data_contract-P009 / L23 | Overview: trace one short text first / This chapter starts with a short running example rather than a large download. We follow “The cat sits.” from … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P010 / L25 | Overview: trace one short text first / Suppose the raw collection has two records: record \(r_1\) contains “The cat sits.” and record \(r_2\) is a we… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P011 / L27 | Overview: trace one short text first / In the byte-tokenizer teaching setup, characters become integer IDs. For illustration, suppose the stream is \… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P012 / L29 | Overview: trace one short text first / The smallest checkpoint is the command below. It proves that the input object can be read by the loader; it do… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P013 / L31 | Overview: trace one short text first / ~~~bash .venv/bin/python -m ttt.data \ --input examples/sample.txt \ --output outputs/book-data-contract \ --s… | 改写/P0 | F01；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-10_data_contract-P014 / L38 | Overview: trace one short text first / On success, &lt;code&gt;outputs/book-data-contract/manifest.json&lt;/code&gt; and token/mask shards should exist. On failu… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P015 / L40 | Overview: trace one short text first / Direct concatenation makes web tokens dominate; source-uniform sampling gives a different distribution. Both a… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P016 / L42 | Overview: trace one short text first / &lt;figure class="book-figure"&gt; &lt;img src="../../../assets/figures/data-funnel.svg" alt="Data pipeline from raw re… | 补写/P1 | T20；具体修复、证据及双语问题见上表 | SVG本体与网页未看 |
| en-10_data_contract-P017 / L47 | What is a tokenizer? Turning text into model-readable tokens / ## What is a tokenizer? Turning text into model-readable tokens | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-10_data_contract-P018 / L49 | What is a tokenizer? Turning text into model-readable tokens / A model cannot consume Unicode text directly. A tokenizer is a rule or learned model that encodes text as an i… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P019 / L51 | What is a tokenizer? Turning text into model-readable tokens / The conversion can be written as | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P020 / L53 | What is a tokenizer? Turning text into model-readable tokens / $$ u \xrightarrow{\ \tau\ } s=(s_1,\ldots,s_L), \qquad s_t\in\{0,\ldots,V-1\}, $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| en-10_data_contract-P021 / L58 | What is a tokenizer? Turning text into model-readable tokens / where $\tau$ is the tokenizer, $V$ is the vocabulary size, and $L$ is the number of tokens produced for this t… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P022 / L60 | What is a tokenizer? Turning text into model-readable tokens / A tokenizer also defines special-token semantics, such as EOS for a document boundary, BOS for a sequence star… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P023 / L62 | Turn the motivation into a measurable distribution / ## Turn the motivation into a measurable distribution | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P024 / L64 | Turn the motivation into a measurable distribution / We now define two objects that are often left implicit: the **training data distribution** and the **predictio… | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P025 / L66 | Turn the motivation into a measurable distribution / Let one raw record be | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P026 / L68 | Turn the motivation into a measurable distribution / $$ r_i=(u_i,m_i), $$ | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-10_data_contract-P027 / L72 | Turn the motivation into a measurable distribution / where $u_i$ is text and $m_i$ contains metadata such as source, license, language, time, and document group. L… | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P028 / L74 | Turn the motivation into a measurable distribution / If we sample uniformly from \(N\) raw records, then | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P029 / L76 | Turn the motivation into a measurable distribution / $$ P_{\mathrm{raw}}(r_i)=\frac{1}{N}. $$ | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-10_data_contract-P030 / L80 | Turn the motivation into a measurable distribution / For two records \(r_1\) and \(r_2\), each has probability \(1/2\). A collection of raw files is not yet the fi… | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P031 / L82 | Turn the motivation into a measurable distribution / $$ C_{\mathrm{train}}=\{r_1,\ldots,r_n\}. $$ | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-10_data_contract-P032 / L86 | Turn the motivation into a measurable distribution / If cleaning, splitting, and source mixing form a transformation $C$, the document distribution can be written … | 改写/P0 | F17、T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P033 / L88 | Turn the motivation into a measurable distribution / $$ P_{\mathrm{doc}}=C_{\#}P_{\mathrm{raw}}, $$ | 改写/P0 | F17、T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-10_data_contract-P034 / L92 | Turn the motivation into a measurable distribution / where $\#$ denotes pushing a distribution through a transformation. For a fixed document collection, if the sa… | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P035 / L94 | Turn the motivation into a measurable distribution / $$ P_{\mathrm{doc}}(r_i)= \frac{w_i}{\sum_{j=1}^{n}w_j}. $$ | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-10_data_contract-P036 / L99 | Turn the motivation into a measurable distribution / Uniform document sampling, length-weighted sampling, and source-level sampling therefore define different dist… | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P037 / L101 | Turn the motivation into a measurable distribution / Document distribution and training-position distribution are also different objects. Uniformly selecting docum… | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P038 / L103 | Turn the motivation into a measurable distribution / A tokenizer $\tau$ maps document text to a token sequence and pushes the document distribution into a token-se… | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P039 / L105 | Turn the motivation into a measurable distribution / $$ \tau(u_i)=s^{(i)}=(s^{(i)}_1,\ldots,s^{(i)}_{L_i}), \qquad s^{(i)}_t\in\{0,\ldots,V-1\}, $$ | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-10_data_contract-P040 / L110 | Turn the motivation into a measurable distribution / where $V$ is the vocabulary size. A window sampler then applies sequence length, packing, EOS, and document-bo… | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P041 / L112 | Turn the motivation into a measurable distribution / $$ P_{\mathrm{tok}}=\tau_{\#}P_{\mathrm{doc}}, \qquad P_{\mathrm{win}}=\Pi_{\#}P_{\mathrm{tok}}. $$ | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-10_data_contract-P042 / L118 | Turn the motivation into a measurable distribution / For document-local sampling, this can be written explicitly as: | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P043 / L120 | Turn the motivation into a measurable distribution / $$ P_{\mathrm{win}}(X,Y,M) = \sum_i P_{\mathrm{doc}}(r_i) \,P_{\Pi}(X,Y,M\mid \tau(u_i)). $$ | 改写/P0 | F18、T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-10_data_contract-P044 / L127 | Turn the motivation into a measurable distribution / The complete chain can be summarized as: | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P045 / L129 | Turn the motivation into a measurable distribution / $$ P_{\mathrm{raw}} \xrightarrow{\,C\,}P_{\mathrm{doc}} \xrightarrow{\,\tau\,}P_{\mathrm{tok}} \xrightarrow{\,… | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-10_data_contract-P046 / L137 | Turn the motivation into a measurable distribution / For a concatenated logical stream, $\Pi$ acts on the packed stream rather than one document, but the same idea… | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P047 / L139 | What exactly is prediction loss? / ## What exactly is prediction loss? | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P048 / L141 | What exactly is prediction loss? / Let a training window contain the consecutive token stream | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P049 / L143 | What exactly is prediction loss? / $$ s=(s_1,s_2,\ldots,s_{L+1}). $$ | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-10_data_contract-P050 / L147 | What exactly is prediction loss? / A causal language model uses the token at each position as input and the following token as target: | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P051 / L149 | What exactly is prediction loss? / $$ x_t=s_t,\qquad y_t=s_{t+1}, \qquad X=(x_1,\ldots,x_L),\quad Y=(y_1,\ldots,y_L). $$ | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-10_data_contract-P052 / L155 | What exactly is prediction loss? / For model parameters $\theta$, the loss at position $t$ is the negative log probability assigned to the observ… | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P053 / L157 | What exactly is prediction loss? / $$ \ell_t(\theta)=-\log p_\theta(y_t\mid x_{1:t}). $$ | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-10_data_contract-P054 / L161 | What exactly is prediction loss? / If the model emits logits $\mathbf{z}_\theta(x_{1:t})$ at position $t$, then | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P055 / L163 | What exactly is prediction loss? / $$ p_\theta(v\mid x_{1:t}) = \operatorname{softmax}(\mathbf{z}_\theta(x_{1:t}))_v. $$ | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-10_data_contract-P056 / L169 | What exactly is prediction loss? / If $M=(m_1,\ldots,m_L)$ is a loss mask, with $m_t=1$ for supervised positions and $m_t=0$ for ignored position… | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P057 / L171 | What exactly is prediction loss? / $$ \ell(\theta;X,Y,M) = -\frac{\sum_{t=1}^{L}m_t\log p_\theta(y_t\mid x_{1:t})} {\sum_{t=1}^{L}m_t}. $$ | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-10_data_contract-P058 / L178 | What exactly is prediction loss? / The population objective induced by the pipeline is | 改写/P0 | F16、T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P059 / L180 | What exactly is prediction loss? / $$ \mathcal{L}(\theta;P_{\mathrm{win}}) = \mathbb{E}_{(X,Y,M)\sim P_{\mathrm{win}}} \left[\ell(\theta;X,Y,M)\r… | 改写/P0 | F16、T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-10_data_contract-P060 / L187 | What exactly is prediction loss? / A finite run sees $N$ sampled windows. If the loss is averaged over all unmasked tokens, the empirical objecti… | 改写/P0 | F16、T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P061 / L189 | What exactly is prediction loss? / $$ \widehat{\mathcal{L}}_N(\theta) = \frac{\sum_{k=1}^{N}\sum_{t=1}^{L_k} m^{(k)}_t\ell^{(k)}_t(\theta)} {\sum… | 改写/P0 | F16、T07；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-10_data_contract-P062 / L197 | What exactly is prediction loss? / Minibatch optimization approximates gradients of this empirical objective. “The model learns from the data” th… | 改写/P0 | F16、T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P063 / L199 | What exactly is prediction loss? / Validation loss uses the same definition with an independent $P_{\mathrm{val}}$ in place of $P_{\mathrm{win}}$… | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P064 / L201 | What exactly is prediction loss? / This is why the chapter treats data as part of the objective. Changing $P_{\mathrm{doc}}$, $\tau$, $\Pi$, or $… | 补写/P1 | T07；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P065 / L203 | How a raw document becomes a training signal / ## How a raw document becomes a training signal | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-10_data_contract-P066 / L205 | How a raw document becomes a training signal / The data pipeline is not just a file transfer to the GPU. It is a sequence of transformations that changes the… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P067 / L207 | How a raw document becomes a training signal / ~~~mermaid flowchart LR A[Raw file] --&gt; B[Document record and metadata] B --&gt; C[Clean, deduplicate, split] C -… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| en-10_data_contract-P068 / L218 | How a raw document becomes a training signal / For an illustrative token sequence: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P069 / L220 | How a raw document becomes a training signal / &#124; Object &#124; Value &#124; &#124;---&#124;---&#124; &#124; Token stream \(s\) &#124; \((10,11,12,2)\) &#124; &#124; Code field `input_ids` (mathematical … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P070 / L227 | How a raw document becomes a training signal / The window loss is | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P071 / L229 | How a raw document becomes a training signal / $$ -\frac{1}{3} \left[ \log p_\theta(11\mid 10) +\log p_\theta(12\mid 10,11) +\log p_\theta(2\mid 10,11,12) \r… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| en-10_data_contract-P072 / L238 | How a raw document becomes a training signal / Masking the final EOS changes both the numerator and denominator. Crossing a document boundary changes the ava… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P073 / L240 | What each transformation changes / ## What each transformation changes | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-10_data_contract-P074 / L242 | What each transformation changes / **Source selection changes support.** Web text, code, books, and papers expose different vocabularies, lengths… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P075 / L244 | What each transformation changes / **Cleaning changes the sample set.** Removing navigation pages, corrupted text, and parser failures changes wh… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P076 / L246 | What each transformation changes / **Deduplication changes weights.** Repeating one document one hundred times increases its probability under $P… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P077 / L248 | What each transformation changes / **Splitting changes the evaluation object.** Document-group splits and contamination checks determine whether … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P078 / L250 | What each transformation changes / **Tokenization changes the sample space and cost.** The same characters can produce different token sequences.… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P079 / L252 | What each transformation changes / **Packing and masks change conditional probabilities.** Cross-document packing, EOS retention, and supervised … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P080 / L254 | A concrete mixture example / ## A concrete mixture example | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-10_data_contract-P081 / L256 | A concrete mixture example / &#124; Source &#124; Documents &#124; Average length &#124; Total tokens &#124; &#124;---&#124;---:&#124;---:&#124;---:&#124; &#124; Web &#124; 900,000 &#124; 1,000 &#124; 900M &#124; &#124;… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P082 / L262 | A concrete mixture example / Concatenation makes web text about 82% of the token stream. Uniform document sampling increases the probabilit… | 改写/P0 | F19；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-10_data_contract-P083 / L264 | A concrete mixture example / If 30% of web tokens are boilerplate duplicates, cleaning may reduce effective web tokens to 630M. If the mixt… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P084 / L266 | A concrete mixture example / The current `BatchStream` samples windows from one logical token stream. It does not implement source-level mi… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P085 / L268 | The pretraining data contract / ## The pretraining data contract | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-10_data_contract-P086 / L270 | The pretraining data contract / Before a long run, the pipeline should answer: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P087 / L272 | The pretraining data contract / 1. **Is the source identifiable?** Can we locate the release, retrieval time, and license? 2. **Is quality exp… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P088 / L278 | The pretraining data contract / Readable data is not necessarily acceptable data. Each requirement needs a saved evidence object and a later p… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P089 / L280 | Data objects and version boundaries / ## Data objects and version boundaries | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-10_data_contract-P090 / L282 | Data objects and version boundaries / **Raw files** are immutable downloads or snapshots such as archives, HTML, JSONL, or repository states. | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P091 / L284 | Data objects and version boundaries / **Document records** contain extracted text and metadata for cleaning, deduplication, splitting, and boundary … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P092 / L286 | Data objects and version boundaries / **Token shards** contain integer sequences and supervision masks plus vocabulary, tokenizer version, dtype, sh… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P093 / L288 | Data objects and version boundaries / **Training windows** are fixed-length sequences sampled from the token stream and converted into `input_ids`, … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P094 / L290 | Data objects and version boundaries / These objects must not share one vague “data version”. A fixed raw file can produce a new document version aft… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P095 / L292 | What the repository currently implements / ## What the repository currently implements | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-10_data_contract-P096 / L294 | What the repository currently implements / `ttt/data.py` provides a byte tokenizer and JSONL-to-token conversion. It can write one-dimensional NPY data o… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P097 / L296 | What the repository currently implements / `ttt/dataset.py` validates manifest version, `vocab_size`, paths, SHA256, token ranges, mask shapes, and mask … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P098 / L298 | What the repository currently implements / ~~~bash .venv/bin/python -m ttt.data \ --input examples/sample.txt \ --output outputs/book-data \ --sharded \ … | 改写/P0 | F01；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-10_data_contract-P099 / L307 | What the repository currently implements / `shard-tokens` is a storage-shard size, not the model sequence length. The current window can cross EOS and do… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P100 / L309 | What the repository currently implements / The manifest also lacks a complete document-offset sidecar, so arbitrary token-to-document tracing remains a r… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P101 / L311 | The route through the remaining data chapters / ## The route through the remaining data chapters | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-10_data_contract-P102 / L313 | The route through the remaining data chapters / - `11_data_acquisition.md`: source versions, licenses, downloads, and provenance; - `12_data_cleaning.md`: nor… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-10_data_contract-P103 / L324 | The route through the remaining data chapters / From the next chapter onward, “data” is no longer a vague input name. We trace each object, how it changes the… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/en/11_data_acquisition.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| en-11_data_acquisition-P001 / L1 | 5. Acquiring data: provenance, licensing, and reproducibility / # 5. Acquiring data: provenance, licensing, and reproducibility | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P002 / L3 | Why data acquisition is part of model design / ## Why data acquisition is part of model design | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P003 / L5 | Why data acquisition is part of model design / When people say that a model was trained on `dataset-a`, they often mean only that a download command mentione… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P004 / L7 | Why data acquisition is part of model design / Data acquisition therefore answers a concrete question: **which bytes were allowed to enter the training pipel… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P005 / L9 | Why data acquisition is part of model design / The book uses a small, original corpus for its first commands. That choice keeps the experiment runnable while… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P006 / L11 | A small example: why a file name is not enough / ## A small example: why a file name is not enough | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P007 / L13 | A small example: why a file name is not enough / Suppose a source page advertises `part-000.jsonl.gz`. On Monday it contains 10,000 documents. On Friday the pu… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P008 / L15 | A small example: why a file name is not enough / If the first run saves only the file name, we cannot tell whether a later loss change came from the architectu… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P009 / L17 | Source records and file records / ## Source records and file records | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P010 / L19 | Source records and file records / Keep a source-level record separate from the file-level record. One publisher release can contain many files, … | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P011 / L21 | Source records and file records / ~~~json { "source_id": "source-a", "publisher": "&lt;publisher&gt;", "source_url": "&lt;canonical-page&gt;", "release": "&lt;… | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-11_data_acquisition-P012 / L39 | Source records and file records / Here `source_id` is a stable label chosen by the project, `release` identifies the upstream version, and `retr… | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P013 / L41 | Source records and file records / This JSON is a provenance record. It is **not** the input schema consumed by `TokenDataset`. The training load… | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P014 / L43 | Source records and file records / For moving resources, prefer a publisher commit, revision, or snapshot over `latest`. An HTTP ETag may help a … | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P015 / L45 | Integrity is different from successful extraction / ## Integrity is different from successful extraction | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P016 / L47 | Integrity is different from successful extraction / An archive can have the expected hash and still produce the wrong training text. For example, an HTML extracto… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P017 / L49 | Integrity is different from successful extraction / Keep raw objects immutable. Write extracted and cleaned text to a new versioned directory, and record the extr… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P018 / L51 | Integrity is different from successful extraction / Also keep failure categories separate: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P019 / L53 | Integrity is different from successful extraction / &#124; event &#124; what it means &#124; how to report it &#124; &#124;---&#124;---&#124;---&#124; &#124; download failure &#124; bytes were not obtained &#124; retr… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P020 / L59 | Integrity is different from successful extraction / Putting all three into a single `removed` counter hides where the distribution changed. A timeout is not evide… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P021 / L61 | Licensing and release boundaries / ## Licensing and release boundaries | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P022 / L63 | Licensing and release boundaries / Review four questions independently: what terms were declared, whether personal information needs handling, wh… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P023 / L65 | Licensing and release boundaries / The raw archive, extracted text, cleaned JSONL, token shards, preprocessing code, and model weights are differ… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P024 / L67 | Licensing and release boundaries / The repository does not perform web crawling, legal review, general HTML extraction, or an automatic license d… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P025 / L69 | Count the acquisition pipeline / ## Count the acquisition pipeline | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P026 / L71 | Count the acquisition pipeline / Consider a purely numerical example. A source list claims 1,000 files. We download 990, parse 980, and obtain … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P027 / L73 | Count the acquisition pipeline / $$ 1000\ \text{listed files} \rightarrow 990\ \text{downloaded} \rightarrow 980\ \text{parsed} \rightarrow 100… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| en-11_data_acquisition-P028 / L81 | Count the acquisition pipeline / Reporting only “99,500 documents” hides ten missing files and ten parsing failures. If the missing files are a… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P029 / L83 | Mapping to this repository / ## Mapping to this repository | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P030 / L85 | Mapping to this repository / For a small text file, the repository command is: | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P031 / L87 | Mapping to this repository / ~~~bash python -m ttt.data \ --input examples/sample.txt \ --output outputs/sample-tokens.npy \ --tokenizer by… | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-11_data_acquisition-P032 / L94 | Mapping to this repository / For JSONL records with a `text` field and a versioned shard manifest: | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P033 / L96 | Mapping to this repository / ~~~bash python -m ttt.data \ --input corpus.jsonl \ --output data/pretrain-v1 \ --sharded \ --tokenizer byte ~… | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-11_data_acquisition-P034 / L104 | Mapping to this repository / These commands encode already selected text; they do not create the source record above. Before running them o… | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P035 / L106 | Minimal run, evidence, and failure diagnosis / ## Minimal run, evidence, and failure diagnosis | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P036 / L108 | Minimal run, evidence, and failure diagnosis / Start with the teaching text that actually exists in the repository: | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P037 / L110 | Minimal run, evidence, and failure diagnosis / ~~~bash python -m ttt.data \ --input examples/sample.txt \ --output outputs/acquisition-sample.npy \ --tokeniz… | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-11_data_acquisition-P038 / L117 | Minimal run, evidence, and failure diagnosis / The expected artifact is a one-dimensional token file. It proves only that the tokenizer can read the input; i… | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P039 / L119 | Minimal run, evidence, and failure diagnosis / ~~~bash python -m ttt.data \ --input examples/sample.txt \ --output outputs/acquisition-shards \ --sharded --t… | 改写/P0 | F01、T08；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-11_data_acquisition-P040 / L126 | Minimal run, evidence, and failure diagnosis / On success, &lt;code&gt;outputs/acquisition-shards/manifest.json&lt;/code&gt; should contain &lt;code&gt;vocab_size&lt;/code&gt;, an i… | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P041 / L128 | Minimal run, evidence, and failure diagnosis / For a real JSONL corpus, use the placeholder &lt;code&gt;&lt;prepared-jsonl&gt;&lt;/code&gt; in the documentation and repl… | 补写/P1 | T08；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P042 / L130 | Summary and next question / ## Summary and next question | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P043 / L132 | Summary and next question / This chapter separates a URL (location), a release (version), and a digest (byte identity). It separates downl… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-11_data_acquisition-P044 / L134 | Summary and next question / The next chapter defines conservative cleaning rules. We will keep the original text and make each rejection r… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/en/12_data_cleaning.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| en-12_data_cleaning-P001 / L1 | 6. Cleaning text: every rule changes the distribution / # 6. Cleaning text: every rule changes the distribution | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P002 / L3 | Why “clean” is not a natural label / ## Why “clean” is not a natural label | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P003 / L5 | Why “clean” is not a natural label / Consider these three records: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P004 / L7 | Why “clean” is not a natural label / ~~~text A. Home &#124; Login &#124; Related links &#124; Next page B. def normalize(text): return text.strip() C. The proof h… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| en-12_data_cleaning-P005 / L14 | Why “clean” is not a natural label / A looks like navigation boilerplate. B is short and contains punctuation, but it is useful code. C contains sy… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P006 / L16 | Why “clean” is not a natural label / Cleaning is therefore a distribution decision. The goal is to reduce obvious noise while measuring false delet… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P007 / L18 | Three kinds of processing / ## Three kinds of processing | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P008 / L20 | Three kinds of processing / Keep these operations separate because they answer different questions: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P009 / L22 | Three kinds of processing / 1. **Format repair** changes representation without intending to change meaning, such as converting CRLF to LF… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P010 / L26 | Three kinds of processing / If all three write `bad_text`, a sudden retention drop cannot be diagnosed. A stable rejection taxonomy might … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P011 / L28 | A conservative normalization function / ## A conservative normalization function | 补写/P1 | T09；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P012 / L30 | A conservative normalization function / Unicode normalization illustrates why the training view and the comparison view should be distinguished. NFC c… | 补写/P1 | T09；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P013 / L32 | A conservative normalization function / The following teaching function returns both a decision and a reason: | 补写/P1 | T09；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P014 / L34 | A conservative normalization function / ~~~python import unicodedata def inspect_document(record, *, min_chars=1, max_chars=100_000): text = record.ge… | 补写/P1 | T09；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-12_data_cleaning-P015 / L57 | A conservative normalization function / The order is part of the contract. Type errors are not empty strings; normalization happens before length meas… | 补写/P1 | T09；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P016 / L59 | A conservation equation for the report / ## A conservation equation for the report | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P017 / L61 | A conservation equation for the report / Let $n_{\mathrm{input}}$ be the number of records entering one cleaning stage, $n_{\mathrm{kept}}$ the records… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P018 / L63 | A conservation equation for the report / $$ n_{\mathrm{input}} = n_{\mathrm{kept}} + \sum_r n_{\mathrm{rejected},r}. $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| en-12_data_cleaning-P019 / L67 | A conservation equation for the report / The equation is only meaningful when all terms use the same unit and stage. If one HTML page is split into thr… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P020 / L69 | A conservation equation for the report / Also state the denominator of every percentage. “The length rule removed 10%” could mean 10% of records after … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P021 / L71 | A numerical example and a failure mode / ## A numerical example and a failure mode | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P022 / L73 | A numerical example and a failure mode / Imagine 10,000 input documents. Schema validation keeps 9,900; non-empty checks keep 9,600; length rules keep … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P023 / L75 | A numerical example and a failure mode / &#124; stage &#124; documents &#124; of raw input &#124; &#124;---&#124;---:&#124;---:&#124; &#124; input &#124; 10,000 &#124; 100% &#124; &#124; schema valid &#124; 9,900 &#124; 99% &#124; … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P024 / L83 | A numerical example and a failure mode / The total retention rate does not tell us whether cleaning succeeded. If a code source retains only 20% while … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P025 / L85 | A numerical example and a failure mode / Short and mixed-language records are especially difficult for language detection. A Chinese code comment being… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P026 / L87 | Mapping to the repository boundary / ## Mapping to the repository boundary | 补写/P1 | T09；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P027 / L89 | Mapping to the repository boundary / The current `ttt.data` path performs limited schema and whitespace checks while converting text into tokens. I… | 补写/P1 | T09；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P028 / L91 | Mapping to the repository boundary / For the repository's educational input, the interface is: | 补写/P1 | T09；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P029 / L93 | Mapping to the repository boundary / ~~~bash python -m ttt.data \ --input examples/sample.txt \ --output data/clean-corpus-v1 \ --sharded \ --token… | 改写/P0 | F01、T09；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-12_data_cleaning-P030 / L101 | Mapping to the repository boundary / The command assumes the JSONL already contains a `text` field and records an input digest in `manifest.json`. … | 补写/P1 | T09；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P031 / L103 | Minimal checkpoint, evidence, and failure diagnosis / ## Minimal checkpoint, evidence, and failure diagnosis | 补写/P1 | T09；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P032 / L105 | Minimal checkpoint, evidence, and failure diagnosis / ~~~bash python -m ttt.data \ --input examples/sample.txt \ --output outputs/cleaning-sample \ --sharded --toke… | 改写/P0 | F01、T09；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-12_data_cleaning-P033 / L112 | Minimal checkpoint, evidence, and failure diagnosis / This command checks only that the cleaned teaching text can reach the tokenizer. It does not replace language,… | 补写/P1 | T09；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P034 / L114 | Summary and next question / ## Summary and next question | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-12_data_cleaning-P035 / L116 | Summary and next question / Cleaning rules change the training distribution; “clean” is not a hidden binary label. Save the rule version, … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/en/13_deduplication.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| en-13_deduplication-P001 / L1 | 7. Deduplication: control implicit sampling weights / # 7. Deduplication: control implicit sampling weights | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-13_deduplication-P002 / L3 | Why repeated text is a modeling problem / ## Why repeated text is a modeling problem | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-13_deduplication-P003 / L5 | Why repeated text is a modeling problem / Suppose one useful article appears once while a navigation template appears 1,000 times. A sampler that treats… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-13_deduplication-P004 / L7 | Why repeated text is a modeling problem / Duplicates can also cross an evaluation boundary. If a paragraph occurs in training and its mirror occurs in v… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-13_deduplication-P005 / L9 | Exact deduplication: define the bytes first / ## Exact deduplication: define the bytes first | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-13_deduplication-P006 / L11 | Exact deduplication: define the bytes first / Exact deduplication computes a digest for a comparison view and keeps one representative. The comparison view … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-13_deduplication-P007 / L13 | Exact deduplication: define the bytes first / For example, two records that differ only in line endings should usually share a key, while two code snippets … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-13_deduplication-P008 / L15 | Exact deduplication: define the bytes first / ~~~python import hashlib import unicodedata def exact_key(text): normalized = unicodedata.normalize("NFC", tex… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| en-13_deduplication-P009 / L26 | Exact deduplication: define the bytes first / The function maps a string to a 256-bit digest. It does not decide which source wins, build a disk index, or s… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-13_deduplication-P010 / L28 | Near duplicates: a worked similarity example / ## Near duplicates: a worked similarity example | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-13_deduplication-P011 / L30 | Near duplicates: a worked similarity example / Full-text hashes miss documents that differ by a date, a page header, or a tracking URL. Near deduplication fi… | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-13_deduplication-P012 / L32 | Near duplicates: a worked similarity example / $$ J(A,B)=\frac{&#124;A\cap B&#124;}{&#124;A\cup B&#124;}. $$ | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-13_deduplication-P013 / L36 | Near duplicates: a worked similarity example / If $A=\{\mathrm{ab},\mathrm{bc},\mathrm{cd}\}$ and $B=\{\mathrm{ab},\mathrm{bc},\mathrm{ce}\}$, the intersecti… | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-13_deduplication-P014 / L38 | Near duplicates: a worked similarity example / MinHash and locality-sensitive hashing (LSH) reduce the number of candidate pairs that need full comparison. T… | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-13_deduplication-P015 / L40 | Similarity is not transitive / ## Similarity is not transitive | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-13_deduplication-P016 / L42 | Similarity is not transitive / Let A and B pass a similarity threshold, and B and C pass it, while A and C do not. If we form a connected com… | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-13_deduplication-P017 / L44 | Similarity is not transitive / The project must choose a stable representative rule: longest text, earliest publication, most complete metada… | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-13_deduplication-P018 / L46 | Local, global, and evaluation deduplication / ## Local, global, and evaluation deduplication | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-13_deduplication-P019 / L48 | Local, global, and evaluation deduplication / Deduplicating inside each source does not remove overlap between sources. A tutorial, a book excerpt, and a we… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-13_deduplication-P020 / L50 | Local, global, and evaluation deduplication / At minimum, report three layers separately: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-13_deduplication-P021 / L52 | Local, global, and evaluation deduplication / &#124; comparison &#124; question &#124; &#124;---&#124;---&#124; &#124; within source &#124; does one publisher repeat its own records? &#124; &#124; across so… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-13_deduplication-P022 / L58 | Local, global, and evaluation deduplication / Long documents add another choice: document-level deduplication can miss a copied paragraph. Paragraph or subs… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-13_deduplication-P023 / L60 | A count and quality report / ## A count and quality report | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-13_deduplication-P024 / L62 | A count and quality report / Suppose 1,000 records enter exact deduplication, 80 share keys with an earlier record, and 50 near-duplicate c… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-13_deduplication-P025 / L64 | A count and quality report / “More removal” is not the same as “better data.” A low threshold can merge genuinely different versions, and a… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-13_deduplication-P026 / L66 | Mapping to this repository / ## Mapping to this repository | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-13_deduplication-P027 / L68 | Mapping to this repository / The current `ttt/data.py` and `ttt/dataset.py` implement tokenization, manifests, SHA256 checks for token arti… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-13_deduplication-P028 / L70 | Mapping to this repository / ~~~bash python -m ttt.data \ --input data/dedup-v1.jsonl \ --output data/tokens-dedup-v1 \ --sharded \ --token… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| en-13_deduplication-P029 / L78 | Mapping to this repository / Run deduplication before assigning related documents to train and validation. Contamination checks against pub… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-13_deduplication-P030 / L80 | Minimal checkpoint, expected report, and failure diagnosis / ## Minimal checkpoint, expected report, and failure diagnosis | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-13_deduplication-P031 / L82 | Minimal checkpoint, expected report, and failure diagnosis / The production sequence creates a new JSONL and report before tokenization: | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-13_deduplication-P032 / L84 | Minimal checkpoint, expected report, and failure diagnosis / ~~~bash python -m ttt.data \ --input &lt;deduped-jsonl&gt; \ --output outputs/dedup-v1 \ --sharded --tokenizer byte … | 补写/P1 | T10；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-13_deduplication-P033 / L91 | Minimal checkpoint, expected report, and failure diagnosis / &lt;code&gt;&lt;deduped-jsonl&gt;&lt;/code&gt; is an upstream artifact that the reader must replace; it is not a repositor… | 改写/P0 | F21、T10；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-13_deduplication-P034 / L93 | Summary and next question / ## Summary and next question | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-13_deduplication-P035 / L95 | Summary and next question / Exact hashing answers whether bytes match; MinHash/LSH narrows candidates; a review or explicit merge rule dec… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/en/14_split_and_contamination.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| en-14_split_and_contamination-P001 / L1 | 8. Splits and contamination: define what “unseen” means / # 8. Splits and contamination: define what “unseen” means | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P002 / L3 | Why random records can produce an easy validation set / ## Why random records can produce an easy validation set | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P003 / L5 | Why random records can produce an easy validation set / Imagine ten mirrors of one tutorial. If each record is assigned independently, some copies almost certainly en… | 改写/P0 | F20；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P004 / L7 | Why random records can produce an easy validation set / The split unit must follow the evaluation question. Use a document cluster to test new documents of known sour… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P005 / L9 | Keep related records together / ## Keep related records together | 补写/P1 | T11；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P006 / L11 | Keep related records together / Let `split_group` be a stable identifier for a duplicate cluster, book, repository, or other unit that should … | 补写/P1 | T11；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P007 / L13 | Keep related records together / ~~~python import hashlib def choose_split(group_id): digest = hashlib.sha256(group_id.encode("utf-8")).digest(… | 补写/P1 | T11；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-14_split_and_contamination-P008 / L27 | Keep related records together / The expected group proportions are about 1%, 2%, and 97%, but token proportions are not guaranteed. One long b… | 补写/P1 | T11；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P009 / L29 | Split before tokenization and packing / ## Split before tokenization and packing | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P010 / L31 | Split before tokenization and packing / If we concatenate all documents, tokenize the stream, and then sample validation windows, adjacent windows can… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P011 / L33 | Split before tokenization and packing / The safer order is: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P012 / L35 | Split before tokenization and packing / 1. assign documents or groups to train, validation, and test; 2. encode each set separately; 3. pack each set … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P013 / L40 | Split before tokenization and packing / Physical shard files may split a long token stream, but a training sampler must not create a logical window th… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P014 / L42 | Contamination is a separate comparison / ## Contamination is a separate comparison | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P015 / L44 | Contamination is a separate comparison / Internal deduplication asks whether training records repeat one another. Contamination analysis compares train… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P016 / L46 | Contamination is a separate comparison / For example, changing names and numbers in a math problem can defeat a byte hash, while a common definition ca… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P017 / L48 | Contamination is a separate comparison / Evaluation access also matters. Repeatedly inspecting one “test” set while selecting models turns it into a de… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P018 / L50 | Read the split report with numbers / ## Read the split report with numbers | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P019 / L52 | Read the split report with numbers / Suppose validation contains 2% of documents but 18% of tokens. This may be a legitimate result of long books, … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P020 / L54 | Read the split report with numbers / Now suppose exact train/validation overlap is zero, but many records share a domain and page template. The rep… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P021 / L56 | Read the split report with numbers / A split report should include document and token counts per group and split, language and source distributions… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P022 / L58 | Mapping to this repository / ## Mapping to this repository | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P023 / L60 | Mapping to this repository / The repository's sample data is for pipeline checks, not for a claim about generalization. After producing ind… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P024 / L62 | Mapping to this repository / ~~~bash python -m ttt.data \ --input data/train-v1.jsonl \ --output data/tokens-train-v1 \ --sharded \ --token… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| en-14_split_and_contamination-P025 / L76 | Mapping to this repository / Then pass the training manifest to the training job and use the validation manifest only for evaluation. Keep … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P026 / L78 | Minimal checkpoint, evidence, and failure diagnosis / ## Minimal checkpoint, evidence, and failure diagnosis | 补写/P1 | T11；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P027 / L80 | Minimal checkpoint, evidence, and failure diagnosis / The two split JSONL files must produce separate manifests: | 补写/P1 | T11；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P028 / L82 | Minimal checkpoint, evidence, and failure diagnosis / ~~~bash python -m ttt.data \ --input &lt;train-jsonl&gt; \ --output outputs/split-train \ --sharded --tokenizer byte… | 补写/P1 | T11；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-14_split_and_contamination-P029 / L93 | Minimal checkpoint, evidence, and failure diagnosis / The expected artifacts are two independent &lt;code&gt;manifest.json&lt;/code&gt; files and one split/contamination report… | 补写/P1 | T11；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P030 / L95 | Summary and next question / ## Summary and next question | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-14_split_and_contamination-P031 / L97 | Summary and next question / The split unit must match the evaluation question: document cluster, project, time period, or task object. Zer… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/en/15_tokenizer.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| en-15_tokenizer-P001 / L1 | 9. Tokenizer and vocabulary / # 9. Tokenizer and vocabulary | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P002 / L3 | A tokenizer is more than preprocessing / ## A tokenizer is more than preprocessing | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P003 / L5 | A tokenizer is more than preprocessing / A tokenizer determines sequence length, character coverage, special-token semantics, embedding size, output-he… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P004 / L7 | A tokenizer is more than preprocessing / Before a model sees text, the tokenizer performs two linked operations: it segments the text according to fixe… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P005 / L9 | A tokenizer is more than preprocessing / $$ \tau(u)=s=(s_1,\ldots,s_L), \qquad s_t\in\{0,\ldots,V-1\}. $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| en-15_tokenizer-P006 / L14 | A tokenizer is more than preprocessing / A token need not be a complete word; it may be a byte, character, or subword. The ID is an index, so its numer… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P007 / L16 | How the tokenizer relates to the vocabulary / ## How the tokenizer relates to the vocabulary | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P008 / L18 | How the tokenizer relates to the vocabulary / The **vocabulary** is one component produced by a tokenizer. It records which tokens are available and which i… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P009 / L20 | How the tokenizer relates to the vocabulary / ~~~text "low" -&gt; 1523 "er" -&gt; 4987 "&lt;eos&gt;" -&gt; 2 ~~~ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| en-15_tokenizer-P010 / L26 | How the tokenizer relates to the vocabulary / The **tokenizer** is the larger system. It includes: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P011 / L28 | How the tokenizer relates to the vocabulary / - a normalizer for Unicode, case, and whitespace; - a pre-tokenizer for initial boundaries; - the vocabulary a… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P012 / L35 | How the tokenizer relates to the vocabulary / The vocabulary is therefore not trained separately and then handed to the tokenizer. BPE training usually lear… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P013 / L37 | How the tokenizer relates to the vocabulary / Keep tokenizer training separate from language-model training: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P014 / L39 | How the tokenizer relates to the vocabulary / ~~~text text corpus → tokenizer training → vocabulary + segmentation rules → freeze tokenizer → token IDs → Tr… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| en-15_tokenizer-P015 / L48 | How the tokenizer relates to the vocabulary / Transformer pretraining learns embeddings, attention, MLP, and other neural parameters. It does not automatica… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P016 / L50 | Vocabulary size and parameter count / ## Vocabulary size and parameter count | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P017 / L52 | Vocabulary size and parameter count / With vocabulary \(V\) and hidden size \(d\): | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P018 / L54 | Vocabulary size and parameter count / $$ E\in\mathbb R^{V\times d},\qquad \mathrm{logits}\in\mathbb R^{\ldots\times V}. $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| en-15_tokenizer-P019 / L59 | Vocabulary size and parameter count / Tied embeddings use the transpose of \(E\) for `lm_head`; untied embeddings add another \(d\times V\) matrix. … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P020 / L61 | Vocabulary size and parameter count / **Tied embeddings** means that the input embedding and the output vocabulary projection share parameters. Inpu… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P021 / L63 | Vocabulary size and parameter count / $$ E\in\mathbb{R}^{V\times d} $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| en-15_tokenizer-P022 / L67 | Vocabulary size and parameter count / to look up hidden vectors. At the output, a hidden state \(h\in\mathbb{R}^{d}\) uses the transpose of the same… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P023 / L69 | Vocabulary size and parameter count / $$ z=hE^{\mathsf{T}}\in\mathbb{R}^{V}. $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| en-15_tokenizer-P024 / L73 | Vocabulary size and parameter count / With untied embeddings, the model stores a separate \(W_{\mathrm{out}}\in\mathbb{R}^{d\times V}\) and computes… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P025 / L75 | Vocabulary size and parameter count / For the tiny configuration with $V=259$ and $d=16$, the embedding has 4,144 parameters; an untied output head … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P026 / L77 | Vocabulary size and parameter count / The MLP output interface remains $d$, because it returns to the residual stream. The vocabulary projection the… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P027 / L79 | How a subword tokenizer is trained / ## How a subword tokenizer is trained | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P028 / L81 | How a subword tokenizer is trained / Training a tokenizer is not training a Transformer. It is usually a corpus-statistics procedure that produces … | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P029 / L83 | How a subword tokenizer is trained / Before training, fix: | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P030 / L85 | How a subword tokenizer is trained / - the tokenizer-training manifest and version; - Unicode normalization, whitespace handling, and pre-tokenizat… | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P031 / L92 | How a subword tokenizer is trained / For BPE, the process is: | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P032 / L94 | How a subword tokenizer is trained / 1. Choose base symbols, such as characters or UTF-8 bytes; 2. count adjacent symbol pairs in the training corp… | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P033 / L100 | How a subword tokenizer is trained / For example, if `l` and `o` often occur together, the procedure may learn `lo`, and later merge `lo` with `w` … | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P034 / L102 | How a subword tokenizer is trained / WordPiece also learns subwords, but uses a likelihood-based criterion and commonly applies greedy longest-matc… | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P035 / L104 | How a subword tokenizer is trained / Unigram starts with a larger candidate set and progressively removes candidates with low contribution. It can … | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P036 / L106 | How a subword tokenizer is trained / After training, save at least: | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P037 / L108 | How a subword tokenizer is trained / - the vocabulary and token-to-ID mapping; - BPE merge rules or Unigram probabilities; - normalizer and pre-tok… | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P038 / L114 | How a subword tokenizer is trained / Evaluation should measure more than whether a vocabulary file was produced. On independent representative data… | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P039 / L116 | How to train a tokenizer in practice / ## How to train a tokenizer in practice | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P040 / L118 | How to train a tokenizer in practice / Tokenizer training does not require choosing a Transformer first. The training object is a set of BPE merge ru… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P041 / L120 | Prepare the corpus / ### Prepare the corpus | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P042 / L122 | Prepare the corpus / Train on the `train` portion of the pretraining data, using the same Unicode normalization and text-extraction… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P043 / L124 | Prepare the corpus / Record the corpus manifest and version, document and whitespace boundaries, normalizer, pre-tokenizer, algorit… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P044 / L126 | Train BPE or Unigram with SentencePiece / ### Train BPE or Unigram with SentencePiece | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P045 / L128 | Train BPE or Unigram with SentencePiece / SentencePiece can train BPE or Unigram directly from raw text. The command below is an algorithm example and r… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P046 / L130 | Train BPE or Unigram with SentencePiece / ~~~bash spm_train \ --input=data/tokenizer/train.txt \ --model_prefix=outputs/tokenizer-32k \ --model_type=bpe… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| en-15_tokenizer-P047 / L143 | Train BPE or Unigram with SentencePiece / Save the complete command, SentencePiece version, input-manifest hash, and output digest. For multilingual dat… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P048 / L145 | Train byte-level BPE with the repository script / ### Train byte-level BPE with the repository script | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P049 / L147 | Train byte-level BPE with the repository script / The repository now provides a real tokenizer-generation entry point: [`scripts/train_tokenizer.py`](https://gi… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P050 / L149 | Train byte-level BPE with the repository script / Install the tokenizer extra: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P051 / L151 | Train byte-level BPE with the repository script / ~~~bash uv sync --extra tokenizer ~~~ | 补写/P1 | F25；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-15_tokenizer-P052 / L155 | Train byte-level BPE with the repository script / Train a 32K byte-level BPE model from the `text` field of a JSONL corpus: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P053 / L157 | Train byte-level BPE with the repository script / ~~~bash uv run python scripts/train_tokenizer.py \ --input data/tokenizer/train.jsonl \ --output outputs/token… | 补写/P1 | F25；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-15_tokenizer-P054 / L166 | Train byte-level BPE with the repository script / Use `--algorithm unigram` for Unigram. Add `--no-byte-level` when the experiment should use whitespace pre-tok… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P055 / L168 | Train byte-level BPE with the repository script / The script reads JSONL into temporary plain-text training files, constructs `models.BPE` or `models.Unigram`, … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P056 / L170 | Use the Hugging Face Tokenizers API directly / ### Use the Hugging Face Tokenizers API directly | 改写/P0 | F22；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P057 / L172 | Use the Hugging Face Tokenizers API directly / The `tokenizers` library lets the experiment define the normalizer, pre-tokenizer, and trainer explicitly: | 改写/P0 | F22；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P058 / L174 | Use the Hugging Face Tokenizers API directly / ~~~python from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers tokenizer = Tokenize… | 改写/P0 | F22；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-15_tokenizer-P059 / L189 | Use the Hugging Face Tokenizers API directly / This trains byte-level BPE. Switching to Unigram requires a different model and trainer and a new recorded con… | 改写/P0 | F22；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P060 / L191 | Connect the trained tokenizer to the model / ### Connect the trained tokenizer to the model | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P061 / L193 | Connect the trained tokenizer to the model / Before training the language model, run an independent check: | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P062 / L195 | Connect the trained tokenizer to the model / ~~~text tokenizer files → encode/decode round trip → token ID range check → special-token check → tokens/byte … | 改写/P0 | F23、T12；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-15_tokenizer-P063 / L206 | Connect the trained tokenizer to the model / The current repository uses `ttt.data` to load a trained tokenizer and generate token shards; the training ent… | 补写/P1 | T12；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P064 / L208 | Where these steps live in the repository / ### Where these steps live in the repository | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P065 / L210 | Where these steps live in the repository / - [`scripts/train_tokenizer.py`](https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch/blob/main… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P066 / L217 | Where these steps live in the repository / The SentencePiece command and `tokenizers` Python example explain the algorithms; the repository's actual gene… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P067 / L219 | Byte tokenizer in this repository / ## Byte tokenizer in this repository | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P068 / L221 | Byte tokenizer in this repository / The repository `ByteTokenizer` maps UTF-8 byte values 0–255 to IDs 3–258 and reserves 0, 1, and 2 for PAD, BOS… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P069 / L223 | Byte tokenizer in this repository / ~~~text byte 0 -&gt; token 3 byte 255 -&gt; token 258 BOS -&gt; 1 EOS -&gt; 2 PAD -&gt; 0 vocab_size = 259 ~~~ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| en-15_tokenizer-P070 / L232 | Byte tokenizer in this repository / It can encode any UTF-8 text without downloading a vocabulary and has no unknown-character case. Its cost is s… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P071 / L234 | Special tokens and data boundaries / ## Special tokens and data boundaries | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P072 / L236 | Special tokens and data boundaries / Special tokens are not ordinary words. If the tokenizer inserts BOS during encoding and the data script insert… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P073 / L238 | Special tokens and data boundaries / The special-token strings, IDs, insertion positions, and loss behavior must therefore be recorded in the token… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P074 / L240 | Measure efficiency by source / ## Measure efficiency by source | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P075 / L242 | Measure efficiency by source / Let $B$ be the UTF-8 byte count of a text and $T$ its token count. The ratio $T/B$ measures encoding density. … | 修订/P2 | T21；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P076 / L244 | Measure efficiency by source / At minimum, report: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P077 / L246 | Measure efficiency by source / &#124; Metric &#124; Why it matters &#124; &#124;---&#124;---&#124; &#124; tokens/byte &#124; estimates context coverage and data cost &#124; &#124; tokens/docu… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P078 / L255 | Measure efficiency by source / A tokenizer can make English more compact while expanding Chinese or code. Overall averages cannot replace gro… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P079 / L257 | Current code and replacement boundary / ## Current code and replacement boundary | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P080 / L259 | Current code and replacement boundary / The repository exposes `ByteTokenizer` and a Hugging Face path through the tokenizer extra. `HuggingFaceTokeni… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P081 / L261 | Current code and replacement boundary / Prepare data with: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P082 / L263 | Current code and replacement boundary / ~~~bash python -m ttt.data --input corpus.jsonl --output data/pretrain --sharded --tokenizer byte python -m tt… | 改写/P0 | F24；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-15_tokenizer-P083 / L268 | Current code and replacement boundary / The three implemented paths have these boundaries: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P084 / L270 | Current code and replacement boundary / &#124; Path &#124; BOS &#124; EOS &#124; loss mask &#124; &#124;---&#124;---&#124;---&#124;---&#124; &#124; `prepare_data` ordinary NPY &#124; added by default in `encode… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P085 / L276 | Current code and replacement boundary / The repository now provides BPE and Unigram training through `scripts/train_tokenizer.py`; it does not yet pro… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P086 / L278 | Current code and replacement boundary / The current manifest records the tokenizer name, vocabulary size, input digest, and shard digests. It does not… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P087 / L280 | Current code and replacement boundary / The training script's `tokenizer_metadata.json` records both the requested and actual vocabulary sizes. Use th… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P088 / L282 | Current code and replacement boundary / Changing the tokenizer requires new token shards and a compatible model configuration; changing only vocab_siz… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P089 / L284 | Minimal checkpoint, evidence, and failure diagnosis / ## Minimal checkpoint, evidence, and failure diagnosis | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P090 / L286 | Minimal checkpoint, evidence, and failure diagnosis / Use the teaching input to inspect the current byte-tokenizer and sharding behavior: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P091 / L288 | Minimal checkpoint, evidence, and failure diagnosis / ~~~bash python -m ttt.data \ --input examples/sample.txt \ --output outputs/tokenizer-sample \ --sharded --tok… | 改写/P0 | F01；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-15_tokenizer-P092 / L295 | Minimal checkpoint, evidence, and failure diagnosis / The manifest should report &lt;code&gt;vocab_size=259&lt;/code&gt;; ordinary text receives EOS, and the sharded path appli… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P093 / L297 | Summary and next question / ## Summary and next question | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P094 / L299 | Summary and next question / A tokenizer fixes integer IDs, context coverage, embedding/output-head cost, and special-token semantics. The … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-15_tokenizer-P095 / L301 | Summary and next question / The next chapter turns the token stream into fixed windows and handles label shifting, EOS boundaries, and los… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/en/16_packing_and_masks.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| en-16_packing_and_masks-P001 / L1 | 10. Packing, labels, and masks / # 10. Packing, labels, and masks | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P002 / L3 | The model does not receive a document / ## The model does not receive a document | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P003 / L5 | The model does not receive a document / People say that a language model “reads a document”, but the training loop receives arrays. The important cont… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P004 / L7 | A four-token example / ### A four-token example | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P005 / L9 | A four-token example / Use letters before using large integer IDs. Suppose a tokenizer produced &lt;code&gt;[A, B, C, D]&lt;/code&gt;. A next-tok… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P006 / L11 | A four-token example / $$ X=(A,B,C), \qquad Y=(B,C,D). $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| en-16_packing_and_masks-P007 / L15 | A four-token example / The first row asks “after &lt;code&gt;A&lt;/code&gt;, what comes next?”; the second asks “after &lt;code&gt;A,B&lt;/code&gt;, what com… | 修订/P2 | T21；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P008 / L17 | A four-token example / $$ X=(x_1,x_2,x_3),\qquad Y=(x_2,x_3,x_4). $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| en-16_packing_and_masks-P009 / L21 | A four-token example / The target at position $i$ is therefore $y_i=x_{i+1}$. If we accidentally set $Y=X$, we train copying rather t… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P010 / L23 | A four-token example / The current loader requests a window of &lt;code&gt;length + 1&lt;/code&gt; tokens. &lt;code&gt;TokenDataset.rows(starts, length… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P011 / L25 | What a mask means / ## What a mask means | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P012 / L27 | What a mask means / A mask answers one narrow question: should this target contribute to the loss? It does not automatically answe… | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P013 / L29 | What a mask means / For example, a prompt can be visible to the model while its targets are masked out. In supervised training, th… | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P014 / L31 | What a mask means / Consider the integer stream | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P015 / L33 | What a mask means / $$ z=(10,11,12,2,20), $$ | 补写/P1 | T21、T13；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-16_packing_and_masks-P016 / L37 | What a mask means / where &lt;code&gt;2&lt;/code&gt; is EOS. A length-four window has $X=(10,11,12,2)$ and $Y=(11,12,2,20)$. If &lt;code&gt;20&lt;/code… | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P017 / L39 | What a mask means / $$ M=(1,1,1,0), $$ | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-16_packing_and_masks-P018 / L43 | What a mask means / so that the final cross-document target is ignored. A continuous-stream policy could use $M=(1,1,1,1)$. The tw… | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P019 / L45 | What a mask means / The masked token loss is commonly written as | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P020 / L47 | What a mask means / $$ \mathcal{L} = \frac{\sum_{i=1}^{L}m_i\,\ell_i}{\max(1,\sum_{i=1}^{L}m_i)}, $$ | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-16_packing_and_masks-P021 / L51 | What a mask means / where $m_i\in[0,1]$ and $\ell_i$ is the cross-entropy for target $y_i$. The &lt;code&gt;max&lt;/code&gt; avoids division b… | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P022 / L53 | What a mask means / The implementation replaces masked target IDs with a safe in-range ID before indexing logits, then multiplies … | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P023 / L55 | Independent documents and continuous streams / ## Independent documents and continuous streams | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P024 / L57 | Independent documents and continuous streams / There are two legitimate packing contracts: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P025 / L59 | Independent documents and continuous streams / - **Independent-document packing** appends EOS and prevents supervision or attention from crossing the boundar… | 改写/P0 | F26；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P026 / L62 | Independent documents and continuous streams / The current &lt;code&gt;prepare_data&lt;/code&gt; and &lt;code&gt;prepare_sharded_data&lt;/code&gt; functions encode each document, ap… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P027 / L64 | Code path and a remote smoke command / ## Code path and a remote smoke command | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P028 / L66 | Code path and a remote smoke command / The relevant path is: | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P029 / L68 | Code path and a remote smoke command / 1. &lt;code&gt;ttt.dataset.prepare_sharded_data&lt;/code&gt; writes token and mask shards plus &lt;code&gt;manifest.json&lt;/code&gt;.… | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P030 / L74 | Code path and a remote smoke command / On a machine with the repository and Python environment already prepared, a small data conversion is: | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P031 / L76 | Code path and a remote smoke command / ~~~ python -m ttt.data \ --input data/example.jsonl \ --output runs/example_shards \ --tokenizer byte \ --shar… | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-16_packing_and_masks-P032 / L85 | Code path and a remote smoke command / The command is a data-preparation example, not a claim that a local workstation has the required dependencies.… | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P033 / L87 | Checks and limits / ## Checks and limits | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P034 / L89 | Checks and limits / For a tiny hand-check, print one sampled window and verify four facts: the labels are one position ahead, the … | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P035 / L91 | Checks and limits / The next chapter turns token files, masks, hashes, and sampler state into a recoverable dataset object. | 补写/P1 | T13；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P036 / L93 | Summary and next question / ## Summary and next question | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-16_packing_and_masks-P037 / L95 | Summary and next question / Packing defines the window, label shift defines the prediction target, and the loss mask defines which targets… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/en/17_dataset_and_manifest.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| en-17_dataset_and_manifest-P001 / L1 | 11. Datasets, shards, and manifests / # 11. Datasets, shards, and manifests | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P002 / L3 | Why a directory is not a data version / ## Why a directory is not a data version | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P003 / L5 | Why a directory is not a data version / Suppose a directory contains &lt;code&gt;part-0.npy&lt;/code&gt;, &lt;code&gt;part-1.npy&lt;/code&gt;, and a temporary file. A script … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P004 / L7 | Why a directory is not a data version / In this book, a **shard** is a storage fragment, while a **dataset** is the logical ordered stream plus the ru… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P005 / L9 | A minimal manifest / ## A minimal manifest | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P006 / L11 | A minimal manifest / The current loader accepts a version-one manifest with the following essential shape: | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P007 / L13 | A minimal manifest / ~~~ { "format_version": 1, "vocab_size": 259, "shards": [ { "tokens": "v-00000.npy", "mask": "v-00000.mask.npy… | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-17_dataset_and_manifest-P008 / L29 | A minimal manifest / &lt;code&gt;tokens&lt;/code&gt; and &lt;code&gt;mask&lt;/code&gt; are filenames relative to the manifest directory. The loader rejects… | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P009 / L31 | A minimal manifest / The vocabulary field matters even when every integer is in range. A file produced with one tokenizer can conta… | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P010 / L33 | What the loader verifies / ## What the loader verifies | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P011 / L35 | What the loader verifies / &lt;code&gt;TokenDataset.__init__&lt;/code&gt; performs several checks with different purposes: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P012 / L37 | What the loader verifies / 1. It checks &lt;code&gt;format_version&lt;/code&gt; and &lt;code&gt;vocab_size&lt;/code&gt;, so a new schema or model vocabulary cann… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P013 / L44 | What the loader verifies / A hash proves that bytes match the manifest; it does not prove that the bytes represent the right corpus. That… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P014 / L46 | A worked cross-shard window / ### A worked cross-shard window | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P015 / L48 | A worked cross-shard window / Use two tiny shards: | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P016 / L50 | A worked cross-shard window / &#124; Physical shard &#124; Token contents &#124; Count &#124; Cumulative end &#124; &#124;---&#124;---&#124;---:&#124;---:&#124; &#124; &lt;code&gt;v-00000.npy&lt;/code&gt; &#124; … | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P017 / L55 | A worked cross-shard window / These ends use half-open code-index intervals: shard 0 covers global positions [0,4), and shard 1 covers [4,9)… | 改写/P0 | F27、T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P018 / L57 | A worked cross-shard window / ~~~text global positions: 2, 3 &#124; 4, 5, 6 physical shard: v-0 &#124; v-1 local offsets: 2, 3 &#124; 0, 1, 2 tokens: 12,13… | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-17_dataset_and_manifest-P019 / L64 | A worked cross-shard window / For a next-token window of length 5, the training path reads one additional token before shifting labels. A sh… | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P020 / L66 | Three identities to keep / ## Three identities to keep | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P021 / L68 | Three identities to keep / There are three useful layers of identity: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P022 / L70 | Three identities to keep / - A **shard digest** detects replacement or corruption of one physical file. - A **manifest digest** detects c… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P023 / L74 | Three identities to keep / If someone reorders two shards, every shard digest still matches but the training sequence changes. The manife… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P024 / L76 | Three identities to keep / &lt;code&gt;prepare_sharded_data&lt;/code&gt; currently assigns a random UUID-like version and refuses to overwrite an exi… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P025 / L78 | Global offsets and windows / ## Global offsets and windows | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P026 / L80 | Global offsets and windows / Let shard lengths be $(n_1,n_2,\ldots,n_K)$ and define cumulative ends | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P027 / L82 | Global offsets and windows / $$ e_j=\sum_{k=1}^{j} n_k. $$ | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-17_dataset_and_manifest-P028 / L86 | Global offsets and windows / A global position $p$ belongs to the first shard with $e_j&gt;p$. &lt;code&gt;TokenDataset.rows&lt;/code&gt; performs this se… | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P029 / L88 | Global offsets and windows / The same distinction matters for document boundaries: physical storage is not semantic structure. If document … | 补写/P1 | T14；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P030 / L90 | Dataset state and sampler state / ## Dataset state and sampler state | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P031 / L92 | Dataset state and sampler state / The dataset answers “what tokens exist?” The sampler answers “which window comes next?”. &lt;code&gt;BatchStream&lt;/co… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P032 / L94 | Dataset state and sampler state / Restoring only the seed is insufficient. A pseudo-random generator has an internal state that advances after e… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P033 / L96 | Dataset state and sampler state / &lt;code&gt;sample_rows(step, indices)&lt;/code&gt; derives independent windows from a &lt;code&gt;SeedSequence&lt;/code&gt; built fro… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P034 / L98 | Remote inspection example / ## Remote inspection example | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P035 / L100 | Remote inspection example / After preparing a sharded dataset on the experiment machine, inspect the manifest and digest before requesting… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P036 / L102 | Remote inspection example / ~~~ python -c "import json,hashlib,pathlib; p=pathlib.Path('runs/example_shards/manifest.json'); print(json.du… | 改写/P0 | F31；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-17_dataset_and_manifest-P037 / L106 | Remote inspection example / The command records the manifest itself; it does not replace the loader's per-shard checks. Save its output wi… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P038 / L108 | Limits / ## Limits | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P039 / L110 | Limits / Memory mapping reduces host memory pressure but random windows can still be storage-bound. Very small shards i… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P040 / L112 | Limits / The next chapter separates file layout from the statistical mixture of data sources. | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P041 / L114 | Summary and next question / ## Summary and next question | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-17_dataset_and_manifest-P042 / L116 | Summary and next question / A dataset is a logical token stream plus a sampling protocol; a shard is only a storage fragment; the manifest… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/en/18_data_mixture.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| en-18_data_mixture-P001 / L1 | 12. Data mixtures and deterministic sampling / # 12. Data mixtures and deterministic sampling | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P002 / L3 | File size is an accidental training decision / ## File size is an accidental training decision | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P003 / L5 | File size is an accidental training decision / Imagine 900 GB of web text, 50 GB of code, and 50 GB of books. Concatenating files gives approximately 90%, 5%… | 改写/P0 | F28；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P004 / L7 | File size is an accidental training decision / The opposite mistake is to sample documents uniformly. A short code snippet and a 200-page book chapter then r… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P005 / L9 | A recipe is a probability distribution plus a budget / ## A recipe is a probability distribution plus a budget | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P006 / L11 | A recipe is a probability distribution plus a budget / Write the intended mixture as data, for example: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P007 / L13 | A recipe is a probability distribution plus a budget / ~~~ { "sources": [ {"name": "web", "weight": 0.70}, {"name": "code", "weight": 0.20}, {"name": "books", "weigh… | 改写/P0 | F29；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-18_data_mixture-P008 / L27 | A recipe is a probability distribution plus a budget / Here &lt;code&gt;weight&lt;/code&gt; is the intended probability before exhaustion and filtering. &lt;code&gt;target_tokens&lt;/cod… | 改写/P0 | F29；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P009 / L29 | A recipe is a probability distribution plus a budget / Keep two tables. The **recipe table** records the target distribution. The **consumption table** records actua… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P010 / L31 | Temperature needs a declared formula / ## Temperature needs a declared formula | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P011 / L33 | Temperature needs a declared formula / “Temperature sampling” is ambiguous unless the exponent is written down. One common convention is | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P012 / L35 | Temperature needs a declared formula / $$ p_i=\frac{w_i^{1/\tau}}{\sum_j w_j^{1/\tau}}, $$ | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-18_data_mixture-P013 / L39 | Temperature needs a declared formula / while another uses $p_i\propto w_i^{\tau}$. They have opposite behavior for $\tau&lt;1$. With weights $(0.8,0.15,… | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P014 / L41 | Temperature needs a declared formula / The mixture distribution is not the same as the final loss distribution. If each sampled window has a mask $m$… | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P015 / L43 | Temperature needs a declared formula / $$ N_i^{\mathrm{effective}}=\sum_{t\in i}m_t, $$ | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-18_data_mixture-P016 / L47 | Temperature needs a declared formula / not simply the number of windows selected from that source. | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P017 / L49 | Determinism across ranks / ## Determinism across ranks | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P018 / L51 | Determinism across ranks / Distributed training adds a choice that is easy to miss. Do we select a source globally and then partition its… | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P019 / L53 | Determinism across ranks / A clear design derives a source and window from a global sample index, seed, and recipe version, then assigns … | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P020 / L55 | Determinism across ranks / $$ (\text{source},\text{window})=f(\text{seed},\text{recipe},\text{global\_index}). $$ | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-18_data_mixture-P021 / L59 | Determinism across ranks / Restoring a run then means restoring the global step and recipe fingerprint, rather than hoping that each rank… | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P022 / L61 | Determinism across ranks / The current &lt;code&gt;BatchStream&lt;/code&gt; samples windows from one logical token stream. It does not implement sour… | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P023 / L63 | A validation plan / ## A validation plan | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P024 / L65 | A validation plan / Before training, generate a fixed sample plan with a fixed seed and count planned source tokens. During traini… | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P025 / L67 | A validation plan / Also report uncertainty over a finite window. If the target is 70/20/10, a 1,000-token window can deviate subs… | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P026 / L69 | Reproducible command skeleton / ## Reproducible command skeleton | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P027 / L71 | Reproducible command skeleton / The repository does not yet expose a source-mixture command. A reproducible experiment can still store the pro… | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P028 / L73 | Reproducible command skeleton / ~~~ python -c "import json; p='runs/recipe.json'; print(json.dumps(json.load(open(p, encoding='utf-8')), inden… | 改写/P0 | F31、T15；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-18_data_mixture-P029 / L77 | Reproducible command skeleton / Treat this as an inspection step until a sampler implementation is added. A recipe file without a sampler and … | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P030 / L79 | Reproducible command skeleton / The next chapter turns these contracts into a preflight gate before compilation and a long training run. | 补写/P1 | T15；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P031 / L81 | Summary and next question / ## Summary and next question | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-18_data_mixture-P032 / L83 | Summary and next question / A mixture is an explicit probability distribution and token budget, not a side effect of file size. State the … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/en/19_data_preflight.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| en-19_data_preflight-P001 / L1 | 13. Data preflight / # 13. Data preflight | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P002 / L3 | Why check before a long run? / ## Why check before a long run? | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P003 / L5 | Why check before a long run? / A four-step tiny run can expose an import error quickly. A multi-day job can spend hours downloading data and … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P004 / L7 | Why check before a long run? / Preflight is not a quality score. It does not prove that the model will learn useful language. It proves that … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P005 / L9 | Eight checks, with the reason for each / ## Eight checks, with the reason for each | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P006 / L11 | 1. Vocabulary alignment / ### 1. Vocabulary alignment | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P007 / L13 | 1. Vocabulary alignment / The manifest vocabulary size, tokenizer vocabulary size, and model configuration must agree: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P008 / L15 | 1. Vocabulary alignment / $$ V_{\mathrm{manifest}}=V_{\mathrm{tokenizer}}=V_{\mathrm{model}}. $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| en-19_data_preflight-P009 / L19 | 1. Vocabulary alignment / Special-token IDs must lie in $[0,V)$, and the BOS/EOS/PAD insertion rules used to create the shards must matc… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P010 / L21 | 2. File completeness and integrity / ### 2. File completeness and integrity | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P011 / L23 | 2. File completeness and integrity / Every token and mask shard must exist. Recomputed SHA256 digests must equal the manifest, &lt;code&gt;count&lt;/code&gt; m… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P012 / L25 | 3. Trainable windows / ### 3. Trainable windows | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P013 / L27 | 3. Trainable windows / For sequence length $L$, the dataset must provide at least $L+1$ tokens. Tokens must be in range, the returned… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P014 / L29 | 4. Boundary semantics / ### 4. Boundary semantics | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P015 / L31 | 4. Boundary semantics / The report must state whether packing is independent-document or continuous-stream, whether windows can cross … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P016 / L33 | 5. Split and contamination evidence / ### 5. Split and contamination evidence | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P017 / L35 | 5. Split and contamination evidence / Training and validation document groups must be disjoint according to the split policy. Record duplicate or ne… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P018 / L37 | 6. Sampling and resume semantics / ### 6. Sampling and resume semantics | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P019 / L39 | 6. Sampling and resume semantics / Record seed, recipe version, shuffle or sampling protocol, rank assignment, initial position, and checkpoint r… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P020 / L41 | 7. Model and data interface / ### 7. Model and data interface | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P021 / L43 | 7. Model and data interface / Check &lt;code&gt;sequence_length &lt;= max_position_embeddings&lt;/code&gt;, valid &lt;code&gt;ttt_chunk_size&lt;/code&gt;, compatibl… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P022 / L45 | 8. Output safety / ### 8. Output safety | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P023 / L47 | 8. Output safety / The output directory must be empty, or the run must explicitly declare resume or initialization from a checkpo… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P024 / L49 | A report a human can audit / ## A report a human can audit | 补写/P1 | T16；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P025 / L51 | A report a human can audit / Each item should contain its inputs, observations, status, and reason: | 补写/P1 | T16；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P026 / L53 | A report a human can audit / ~~~ { "name": "vocab_alignment", "status": "pass", "inputs": { "manifest_sha256": "&lt;sha256&gt;", "config_sha256":… | 补写/P1 | T16；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-19_data_preflight-P027 / L70 | A report a human can audit / Use &lt;code&gt;pass&lt;/code&gt;, &lt;code&gt;warning&lt;/code&gt;, and &lt;code&gt;fail&lt;/code&gt; deliberately. A warning states the human de… | 补写/P1 | T16；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P028 / L72 | A complete pass, warning, and fail example / ## A complete pass, warning, and fail example | 改写/P0 | F30；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P029 / L74 | A complete pass, warning, and fail example / Here is a shortened report for a tiny run. File and window checks pass, continuous windows across EOS receive … | 改写/P0 | F30；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P030 / L76 | A complete pass, warning, and fail example / ~~~ { "checks": [ {"name": "file_integrity", "status": "pass", "message": "2 shards and masks match their hash… | 改写/P0 | F30；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-19_data_preflight-P031 / L87 | A complete pass, warning, and fail example / The fail should be repaired before submitting a long job. Either the model must use the 259 positions required… | 改写/P0 | F30；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P032 / L89 | A complete pass, warning, and fail example / ~~~ python -c "from ttt.dataset import TokenDataset; TokenDataset('runs/example_shards/manifest.json', 259); p… | 改写/P0 | F30、F31；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-19_data_preflight-P033 / L93 | A complete pass, warning, and fail example / This command checks implemented file and range invariants. It does not replace split, contamination, or distri… | 改写/P0 | F30；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P034 / L95 | What the repository provides today / ## What the repository provides today | 补写/P1 | T16；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P035 / L97 | What the repository provides today / There is not yet a standalone &lt;code&gt;ttt.preflight&lt;/code&gt; command. The current minimum gate is the validation p… | 补写/P1 | T16；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P036 / L99 | What the repository provides today / ~~~ python -c "from ttt.dataset import TokenDataset; TokenDataset('runs/example_shards/manifest.json', 259); p… | 改写/P0 | F31、T16；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-19_data_preflight-P037 / L103 | What the repository provides today / This command checks the loader's implemented invariants; it does not perform split contamination analysis, sou… | 补写/P1 | T16；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P038 / L105 | The boundary of the claim / ## The boundary of the claim | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P039 / L107 | The boundary of the claim / Passing preflight establishes input consistency and a recoverable experiment identity. It does not establish u… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P040 / L109 | Summary and next question / ## Summary and next question | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-19_data_preflight-P041 / L111 | Summary and next question / Preflight must label each check as pass, warning, or fail. Passing establishes consistency of the input object… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |

### docs/book/en/20_baseline_model.md

| Paragraph ID / 行 | 小节 / 首句 | 处理结论 | 问题、修复与证据 | 未核验 |
|---|---|---|---|---|
| en-20_baseline_model-P001 / L1 | 15. Dense causal-language-model baseline / # 15. Dense causal-language-model baseline | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P002 / L3 | From logits to loss / ## From logits to loss | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P003 / L5 | From logits to loss / Logits are scores, not probabilities. If a vocabulary contains `cat`, `sat`, and `dog`, scores $(2,1,0)$ becom… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P004 / L7 | From logits to loss / $$ p_v=\frac{\exp z_v}{\sum_u\exp z_u}. $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| en-20_baseline_model-P005 / L11 | From logits to loss / If the correct token is `sat`, this example gives probability about $0.245$ and negative log-likelihood about … | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P006 / L13 | From logits to loss / The dense baseline is the control for TTT, Loop, and MoE. It fixes tokenizer, manifests, sequence length, effe… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P007 / L15 | From logits to loss / For logits \(z\) at position \(t\) and target \(y_t\): | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P008 / L17 | From logits to loss / $$ \ell_t=-\log\frac{\exp z_{t,y_t}} {\sum_v \exp z_{t,v}}. $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| en-20_baseline_model-P009 / L22 | From logits to loss / The repository can compute cross-entropy in token and vocabulary blocks, preserving the same objective while r… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P010 / L24 | From logits to loss / The baseline disables experts, fast MLP, inner updates, and loop mode. It can still choose sliding or full att… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P011 / L26 | From logits to loss / Each training step samples a window, runs sequence_loss, computes outer gradients, applies Optax, records loss… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P012 / L28 | From logits to loss / The gradient with respect to one logit has a simple form: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P013 / L30 | From logits to loss / $$ \frac{\partial\ell_t}{\partial z_{t,v}} =p_{t,v}-\mathbf{1}[v=y_t]. $$ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| en-20_baseline_model-P014 / L35 | From logits to loss / The correct token receives $p_{t,y_t}-1$, which pushes its score up; every other token receives its current pr… | 改写/P0 | F36；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P015 / L37 | From logits to loss / We can derive it by expanding the loss: | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P016 / L39 | From logits to loss / $$ \ell_t=-z_{t,y_t}+\log\left(\sum_u\exp z_{t,u}\right), \qquad \frac{\partial\ell_t}{\partial z_{t,v}} =-\ma… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 渲染未验；推导背景见T项 |
| en-20_baseline_model-P017 / L46 | From logits to loss / The second term is exactly the softmax probability, which gives the form $p_{t,v}-\mathbf{1}[v=y_t]$. | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P018 / L48 | One minimal gradient update / ## One minimal gradient update | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P019 / L50 | One minimal gradient update / Use the same logits example to isolate the direction of the error signal. Let \(\mathbf z=(2,1,0)\), with the … | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P020 / L55 | One minimal gradient update / $$ \mathbf g\approx(0.665,-0.755,0.090). $$ | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-20_baseline_model-P021 / L59 | One minimal gradient update / If logits were temporarily treated as directly updateable variables, a plain SGD step with \(\eta=0.1\) would … | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P022 / L62 | One minimal gradient update / $$ \mathbf z'=\mathbf z-\eta\mathbf g \approx(1.9335,1.0755,-0.009). $$ | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 渲染未验；推导背景见T项 |
| en-20_baseline_model-P023 / L67 | One minimal gradient update / The correct score rises from 1 to about 1.076, and its softmax probability rises from about 0.245 to about 0.2… | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P024 / L74 | What the baseline disables / ## What the baseline disables | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P025 / L76 | What the baseline disables / ~~~text num_experts = 0 use_fast_mlp = false inner.enabled = false loop_mode = none ~~~ | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 未执行；仅核对静态入口/协议 |
| en-20_baseline_model-P026 / L83 | What the baseline disables / These switches remove routing, extra fast state, and depth recurrence. Attention may still be full or sliding,… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P027 / L85 | One training step and one recovery step / ## One training step and one recovery step | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P028 / L87 | One training step and one recovery step / The training path creates the AdamW schedule from the resolved configuration, samples a window and mask, compu… | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P029 / L89 | One training step and one recovery step / Recovery is part of the baseline. A resumed run must restore parameters, optimizer statistics, global step, sa… | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P030 / L91 | What a falling loss does and does not show / ## What a falling loss does and does not show | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P031 / L93 | What a falling loss does and does not show / For a teaching run, check finite loss, nonzero and bounded gradients, no immediate NaNs, continuity after rest… | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P032 / L95 | Parameters, activation memory, and measured cost / ## Parameters, activation memory, and measured cost | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P033 / L97 | Parameters, activation memory, and measured cost / Parameter count is the number of leaves in the model tree, but it is not activation memory. TTT may retain or … | 改写/P0 | F34；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P034 / L99 | Parameters, activation memory, and measured cost / For each later comparison, report data and tokens, total and active parameters, optimizer and inner-loop setti… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P035 / L101 | Minimal run, evidence, and failure diagnosis / ## Minimal run, evidence, and failure diagnosis | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P036 / L103 | Minimal run, evidence, and failure diagnosis / Start with the sample tokens and tiny configuration: | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P037 / L105 | Minimal run, evidence, and failure diagnosis / ~~~bash bash scripts/prepare_sample.sh outputs/sample-tokens.npy JAX_PLATFORMS=cpu bash scripts/run_train.sh \… | 改写/P0 | F05、T17；具体修复、证据及双语问题见上表 | 未执行；仅核对静态入口/协议 |
| en-20_baseline_model-P038 / L113 | Minimal run, evidence, and failure diagnosis / The expected directory contains &lt;code&gt;experiment.json&lt;/code&gt;, &lt;code&gt;environment.json&lt;/code&gt;, &lt;code&gt;metrics.jso… | 补写/P1 | T17；具体修复、证据及双语问题见上表 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P039 / L115 | Summary and next question / ## Summary and next question | 保留/定位 | 标题只作位置标识，不构成该节通过 | 网页/远端/外链全文未验 |
| en-20_baseline_model-P040 / L117 | Summary and next question / The dense baseline fixes data identity, budget, configuration, and recovery protocol. Logits, masked loss, gra… | 保留/静态 | 未发现该块独立硬错误；仅限正文静态审读 | 网页/远端/外链全文未验 |
