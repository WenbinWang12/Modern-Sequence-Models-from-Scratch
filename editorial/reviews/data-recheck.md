# 数据主线修订后的交叉复审

审阅日期：2026-11-11。审阅者：rewrite_frontier_lab_parity（与本批数据章节作者不同的子代理）。对照记录：[main-review.md](main-review.md)，其原始基线为 567fef3；本次复审对象是根代理正在修改的工作区，不把修订后的内容追记为原快照通过。

范围：00、notation、10、13、14、15、16、17、18、19、20 的中英文；重点检查 F01–03、F16–24、F26–34、F36。先完整复读 10/18/19，再核对其他页的相关段、推导、命令和实际源码。首次审稿时15/16/17与notation仍在编辑，下面保留首句/式子作为历史定位。第二轮增量复审后，经根代理授权，仅补写17中英文的末尾实践段；没有修改其他章节，也没有执行本地或远端测试。

结论：若干原始硬错误已实质修正，但不能给整批“出版通过”。未执行命令、视觉/数学渲染、实际 tokenizer 行为和完整训练/评估/恢复链均不在本轮验收范围。这里的“静态关闭”只关闭编号所述事实错误，不关闭该章其他教学缺口。

## 最终限定复核：当前状态更新

按根代理指定范围，再次核对15中英、notation中英与Frontier解释器行；本次只更新既有发现，不扩大审稿范围。以下更新覆盖第二轮表中R05/R10的旧状态。

| 对应问题 | 最终静态判断 | 核对证据 |
|---|---|---|
| F24、R05 | 所述入口前提问题静态关闭 | 15两版明确两条命令是替代实验；corpus.jsonl须为已清洗、已切分且每行含text的上游文件。第二条使用上文训练得到的本地outputs/tokenizer-32k，输出分为data/pretrain-byte与data/pretrain-bpe，两条均调用.venv/bin/python。首跑不再隐含需要下载Llama tokenizer。文件是否实际存在、tokenizer是否训练成功仍待运行证据。 |
| R10：字节与batch符号 | 静态关闭所述冲突 | 15两版改用N_bytes和T/N_bytes，与notation一致，B继续表示batch大小。 |
| R10：token/chunk/outer轴 | 静态关闭所述冲突 | notation两版将快权重公式及状态表改为phi_c，明确outer step k、chunk c和token位置t。SGD式限定不考虑梯度裁剪，不把简化公式说成完整实现。 |
| Frontier解释器一致性 | 静态关闭裸python问题 | 中英线性例、Loop配方生成、长上下文validate及配对eval的直接Python命令全部为.venv/bin/python；索引页仍调用已有bash wrapper，没有新增裸python入口。这里只核对文本，不代表这些命令已运行。 |

继续开放：F22对应的实际tokenizer训练脚本仍未显式加入完整initial_alphabet；未见字符与NFKC解码行为没有实测。17/19/Frontier新命令均未在本轮运行，图形与数学网页渲染未验，20完整held-out与恢复证据链未完成。以上静态关闭不能推导出版通过或实验通过。

## 第二轮增量复审快照

根代理修订后重新阅读相关段落，并与真实实现核对。以下状态更新优先于后面的首次审稿快照，避免把已修改的问题继续列作当前未修复。

| 对应问题 | 当前判断 | 增量证据及仍保留的边界 |
|---|---|---|
| R01、R02 | 静态关闭 | 10两版将ttt.data限定为生成分片，另指Dataset/preflight读回；表改为Teaching array/教学数组，并明确loader返回tokens/masks、sequence_loss构造X/Y。 |
| F26、R03 | 静态关闭所述语义错误 | 16英文改为boundary-target masking；末尾检查限定固定inputs/mask/初态，只改最后masked target。新增真实ByteTokenizer例：A/B字节65/66→ID68/69，流[1,68,2,1,69,2]，目标mask[1,1,0,1,1]；两版表格和代码对应。零mask不保证后文无法读取A，正文已指出。片段尚未执行，完整attention/state隔离实现仍缺。 |
| F27 | 静态关闭 | 17两版均写side='right'，新增p=4→分片1、局部offset0；与TokenDataset.rows逐全局位置searchsorted对应。 |
| F31、R04 | 所指路径/入口缺口静态关闭 | 16两版均使用examples/sample.jsonl与outputs/packing-sample。17两版现有相同自包含片段：检查或生成outputs/dataset-sample，真实构造TokenDataset并调用rows，保存rows-check.json。旧runs/example_shards不再作为隐含前提。 |
| R06 | 静态关闭假产物主张 | 14两版明确编码只产两个manifest，split/contamination报告须上游另产；不再靠manifest摘要不等推断文档不重叠。真正执行分组切分与污染报告的T11教学路径仍待实现。 |
| R07 | 已补失败协议，执行未验 | 19两版在同一manifest先以259读取，随后用32000触发ValueError，并显示预期源码消息。异常分支真实，当前没有远端失败日志，不声称已测。 |
| R08 | 过度结论静态关闭；完整实验链仍开放 | 20两版将四步dense命令限为训练产物与有限loss，明确不完成恢复/heldout；真实tiny_dense关闭fast、suffix和inner。独立验证数据、eval与恢复证据仍缺，T17不能关闭。 |
| R09 | 静态关闭方向歧义 | 19中文改为tokens[:-1]/tokens[1:]；与英文next-token shift一致。 |
| R10 | 部分修复 | notation两版以n_exp/k_sel表示expert数量/选择数，区分embedding E，并定义N_bytes。15正文仍用B表示字节，notation仍用phi_t描述chunk/session；这些跨章记号一致性问题继续开放。 |
| F22–24、R05 | 保留原边界 | API已有decoder/alphabet，NFKC不可逆说明正确；实际训练脚本仍缺完整alphabet，未见字符行为未验。15远端tokenizer入口的访问条件、固定本地版本与语料前提仍待补；不由其他修订推导该路径可复现通过。 |

17新增实践片段以JSONL fixture生成分片；已有manifest绝不覆盖，先核对源文件SHA256、byte模式与supervised标志，再由TokenDataset检查词表及文件。它读取首片边界前两位和恰好落在边界的两个五token窗口，与直接切片拼接的数组比较，并打印X/Y/M。报告中的cumulative_ends为loader计算结果，不是虚构manifest字段。它只检查两个窗口的token顺序，mask仅打印供核对，不能推广到全语料语义验收。

该片段中英文代码一致；源码接口和形状经过静态核对，命令没有执行。git diff --check只检查文本格式，不构成NumPy、loader、tokenizer或网页验收。出版结论仍保持未通过，不自动关闭原记录的全部T项。

## 首次审稿快照：原始问题的逐项复核

| 问题 | 当前静态判断 | 修复证据及限制 |
|---|---|---|
| F01 | 所检范围内旧 TXT→sharded 错误已消除 | 00/notation/10/15/16中文/17中文/19 改用新 examples/sample.jsonl；其4行各有 text。普通 prepare_sample.sh 保持TXT→单NPY是合法路径。英文16/17仍有另一套未知上游路径，见R04；写出分片不等于loader已验证。 |
| F02 | 静态关闭 | notation 两版说明 M=(1,1,0) 屏蔽EOS；加55后得到 X=(17,42,31,2)、Y=(42,31,2,55)、M=(1,1,1,0)，确实屏蔽跨文档目标。 |
| F03 | 静态关闭 | notation 两版用100与900 token文档，先选文档再取单token得1/200与1/1800；全流均匀为1/1000。限定单token示例，未混入窗口边界。 |
| F16 | 静态关闭核心错误 | 10两版定义 A/K 的期望与 E[A]/E[K]，说明 K>0；例子窗口均值2.5与token均值1.75正确。补有限batch比值估计偏差和跨batch总量聚合；training.py的counts加权、eval_retrieval.py的loss×count累计与之对应。 |
| F17 | 静态关闭 | 10先固定清洗去重后的集合，再定义 wi/sum(w)；删除一条后三等概率剩下各1/2，删掉不适用的简单pushforward说法。无需保留抽象记号强行证明非逐条去重。 |
| F18 | 静态关闭 | 10两版都限定文档局部条件分布，并另外定义拼接流起点a=1..T−L与指示函数分布，涵盖跨文档窗口。需T≥L+1，实际loader使用时仍要校验长度。 |
| F19 | 静态关闭 | 两版表后明确book token=100/1100≈9.09%，均匀选文档后单窗=10000/960000≈1.04%，均匀选来源=1/3；不再写无条件“提高”。 |
| F20 | 静态关闭 | 14两版假设独立分配，推导1−(1−p)^10−p^10；p=.01约9.56%、p=.1约65.13%正确。 |
| F21 | 静态关闭 | 13两版最小检查明确890+80+30=1000；near_reviewed=50为与结果重叠的活动计数，不再加总。 |
| F22 | 文档事实修复，行为保障未闭合 | 15 API补ByteLevel decoder及initial_alphabet；实际scripts/train_tokenizer.py仅有decoder，尚无完整alphabet。两版新增段如实承认脚本可能UNK，未声称已实现无损fallback。未运行未见字符例，不能把这记为训练脚本覆盖检查通过。 |
| F23 | 静态关闭归一化错误 | 15两版将round trip目标改为normalize(text)，给Ａ→A与①→1解释不可逆。对于可能UNK的脚本路径仍须先验证覆盖，不能承诺每个输入都等于normalize(text)。 |
| F24 | 输出覆盖错误关闭；入口前提仍需补 | 两版byte与Llama改为data/pretrain-byte、data/pretrain-llama，避免依次覆盖同manifest。但命令仍使用未知corpus.jsonl、远端模型名和普通python；需标为替代方案，并说明输入与授权/固定本地文件前提，见R05。 |
| F26 | 独立文档定义已修；相邻例子需继续审 | 16两版定义现需attention/state隔离，不再以一个loss mask保证隔离。审阅时英文mask例仍将M称isolated-document policy，中文末检查对mask与TTT状态的说法不清，见R03。 |
| F27 | 部分修复 | 中文不再要求manifest有start/end，已说明count累积。但两版手算仍写searchsorted([4,9],2)，缺side='right'与p=4端点；仅p=2无法区分左右查找。待修后复核。 |
| F28 | 静态关闭 | 18两版明确900/50/50 GB仅能推出字节90%；重新声明800M/150M/50M token算例。 |
| F29 | 静态关闭 | 两版用alpha幂指数、给pi/pj=(wi/wj)^alpha，明确T=1/alpha；JSON probability为最终概率，不再重复温度变换。 |
| F30 | 静态关闭虚假实测表述 | 19改为由真实入口生成小报告，scope限定loader及一个窗口；不填伪造hash/计数，未称已运行。代码读取rows([0],17)返回(1,17) tokens、(1,16) masks，字段与实现吻合。 |
| F31 | 18/19关闭，17尚未全闭合 | 18两版共同创建outputs/mixture-example/recipe.json且明确未接入trainer；19先生成outputs/preflight-example后读取同路径。英文17仍读runs/example_shards，与中文outputs/dataset-sample不齐，见R04。 |
| F32 | 限定后的静态关闭 | 19删除initialize-from必须同数据的误述；当前表中一致性限定“恢复”。initialize-from与resume区别应继续由恢复章解释；没有额外声称实际恢复已比较。 |
| F33 | 静态关闭 | 20中文softmax分母改v=0..V−1。英文sum_v不与0-based ID冲突，但写相同上下界更便于初学者核对。 |
| F34 | 静态关闭 | 20英文改叶张量元素数之和，用16×32矩阵=1叶、512参数说明；与profile.py的sum(value.size)对应。 |
| F36 | 静态关闭首次无条件主张 | 00/notation/20中英均在梯度方向说明处限定直接优化独立logits，并说明共享参数/其他样本/优化器不能保证逐样本单调。手算更新z'≈(1.9335,1.0755,−.009)和p正确项≈.271一致。 |

## 首次审稿快照：新发现或需保留的具体问题

| 编号 | 级别 | 文件、首句或代码定位 | 具体问题与所需修复 |
|---|---|---|---|
| R01 | P0证据边界 | 10两版，“本章的最小检查…” / “The smallest checkpoint…” | 文字说命令证明输入可被loader读取，但命令只有ttt.data分片写入，没有构造TokenDataset。把结论限为生成token/mask/manifest，或追加真实loader调用。不要把“生成成功”当作所有loader不变量通过。 |
| R02 | P0代码对应 | 10两版，“一份原始文档如何变成训练信号”中的表；英文明确Code field input_ids/labels | 实际TokenDataset/BatchStream返回(tokens,masks)，sequence_loss内部shift。notation与中文16已修正，但10仍像真实返回字段；改称教学数组名并给真实返回映射。 |
| R03 | P0/P1待作者再修 | 16中文“若最后一列mask为0，它不应改变loss或TTT状态”；英文isolated-document例 | 设置mask由1变0当然改变loss/梯度。若要测试屏蔽，应限定固定inputs与mask，只更改被屏蔽且不再作为输入的末尾target ID。英文“isolated”需改为仅屏蔽跨边界目标，另需attention/state隔离。实际输出打印和失败对照也未给出，T13仍开放。 |
| R04 | P0双语协议 | 16英文data/example.jsonl→runs/example_shards；17英文读取runs/example_shards；中文分别outputs/packing-sample、outputs/dataset-sample | 两版不是同一可复现链，上游data/example.jsonl未在本章产生。统一已有fixture与输出，并提供实际rows读取；否则F31只部分关闭，T14继续开放。 |
| R05 | P1实践前提 | 15“数据准备入口是”/Prepare data with | corpus.jsonl需明确已有上游语料；Llama入口需要获得访问且固定文件版本，当前CLI没有revision参数。优先给本地固定tokenizer目录作为可复现实验，再将远端名称列为有前提的替代。解释器仍应使用环境章的.venv。 |
| R06 | P0证据边界 | 14两版最小检查，“预期产物是两个独立manifest和一份split/contamination报告” | 两个ttt.data命令只编码输入，不运行choose_split或污染分析，不能产生该报告。分开“此命令输出”与“上游必须提供的报告”，或补真正生成split与报告的教学命令。T11未关闭。 |
| R07 | P1教学实作 | 19“从已有样例生成一份小报告” | 正例代码真实，但只有成功路径；原T16要求成功/失败对照仍缺。可在同一manifest上以错误vocab构造TokenDataset并确认特定异常，记录其scope。不要把该条留白写为全项通过。 |
| R08 | P1实验链 | 20最小运行与“恢复后下一批…”/held-out结论 | 新tiny_dense确实关闭fast、suffix、inner，解决baseline身份；但本章命令仍只做4步sample训练，没恢复比较、独立验证数据或eval调用。结果表只能期待训练产物，不能声称这些命令证明恢复/heldout。T17继续开放。 |
| R09 | P2术语/双语 | 19中文“目标由窗口右移一位取得”与16“向左错开一位” | 用tokens[:-1]/tokens[1:]或“目标取原序列下一位置”消除左右视角歧义；英文“shifted by one”不含方向。 |
| R10 | P2记号待作者完成 | notation及15/16 | 审阅时仍有E既表示expert数又embedding、phi_t同时称chunk/session、15用B表示字节、英文16流用x/z的旧写法。根代理正处理，需按最终稿再验，不能由F02/03关闭推断统一记号完成。 |

## 抽查过的推导与没有检验的部分

18的新推导链可以静态成立：固定来源选择概率pj，给定该来源的平均有效位置kj，长期有效位置比例qj=pjkj/sum(piki)。每窗有效数相等时才能把选择概率称token比例。sqrt(.8,.15,.05)归一化约(.5942,.2573,.1485)；书籍预算148.5M/50M≈2.97。这里是长期比率/预算估计，不是每个有限随机批次都恰好满足。

18的比例方差推导使用独立且同概率的Bernoulli选择，Var(p_hat)=p(1−p)/n；p=.2,n=100的标准差=.04正确。它不能用于把窗内1000 token当作1000独立来源选择；正文已明确限制。来源级确定性sample plan仍仅设计，T15的真实恢复抽样示例尚未完成。

10的两种平均数与14的重复簇泄漏概率已经手算核对。00新增概率链式法则与取负log的路径较原稿完整，但零基础读者所需的单变量导数/链式求导基础不因这几处修复就全部补齐；T01等教材深度问题不能自动关闭。

本次没有执行任何doc命令，因此不存在新的run报告、tokenizer未见字符结果、loader异常结果、恢复比对或质量分数。图片本体、响应式页面、数学渲染亦未验。完成事实勘误后仍应沿“同一示例→手算→代码→真实输出→失败解释”的路径逐节继续审稿。
