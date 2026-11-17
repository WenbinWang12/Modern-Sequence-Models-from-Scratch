## 3 实验设置

### 3.1 四个模型

主实验训练四个 12 层、宽度 768 的模型。它们使用同一个 tokenizer、同一份数据、完全相同的 batch 顺序、同样的优化器和同样的训练 token 数，只在“怎样使用上下文”上不同。

| 模型 | 注意力 | 带 MLP′ 的块 | 测试时更新 | 总参数 | 非嵌入参数 |
|:--|:--|:--|:--|--:|--:|
| 全注意力 | 可见全部 4,096 个位置 | – | – | {{ M(r.training_cost.full.parameters, 2) }} | {{ M(r.training_cost.full.parameters - 50257 * 768, 2) }} |
| 滑窗注意力 | 最近 \(w=512\) 个位置 | – | – | {{ M(r.training_cost.swa.parameters, 2) }} | {{ M(r.training_cost.swa.parameters - 50257 * 768, 2) }} |
| 无更新对照 | 最近 \(w=512\) 个位置 | 最后 3 块 | 不更新 | {{ M(r.training_cost.ck.parameters, 2) }} | {{ M(r.training_cost.ck.parameters - 50257 * 768, 2) }} |
| TTT-E2E | 最近 \(w=512\) 个位置 | 最后 3 块 | 每 \(C=256\) 个 token 一步 | {{ M(r.training_cost.ttt.parameters, 2) }} | {{ M(r.training_cost.ttt.parameters - 50257 * 768, 2) }} |

每个对照回答一个问题：

- **全注意力**是上下文内信息无损可见时的参照。它的成本随上下文线性增长，本报告把它看作质量上的目标，而不是成本上的对手。
- **滑窗注意力**是成本与 TTT-E2E 同量级的基线。它的每一层只能直接看到最近 512 个位置；更早的信息只能靠逐层接力间接传递（12 层在理论上最远可达约 6,100 个位置），模型没有专门的长程记忆。
- **无更新对照**与 TTT-E2E 结构相同、参数量相同（都比前两者多 {{ M(r.training_cost.ttt.fast_parameters, 2) }} 个 MLP′ 参数），训练和推理时都不做内循环。它用来区分“多出来的参数”和“测试时更新”各自的作用。
- 评测时还会把训练好的 TTT-E2E **关闭更新**再测一次，用来观察同一组慢参数在不更新时的表现。

其余结构四者相同：\(d=768\)，12 个注意力头，\(d_{\mathrm{ff}}=2048\)，输入输出嵌入共享，RoPE 基数 \(5\times10^5\)，QK 归一化。

### 3.2 数据

| 语料 | 用途 | 文档数 | token 数 | 字节/token | 来源与许可 |
|:--|:--|--:|--:|--:|:--|
| FineWeb-Edu `sample/10BT`，分片 000–002 | 训练 | {{ n(r.audits.corpora["fineweb-edu-train"].documents) }} | {{ B(r.audits.corpora["fineweb-edu-train"].tokens) }} | {{ f2(r.audits.corpora["fineweb-edu-train"].bytes_per_token) }} | HuggingFaceFW/fineweb-edu，revision `87f0914`，ODC-By 1.0[^fineweb] |
| PG-19 train | 训练 | {{ n(r.audits.corpora["pg19-train"].documents) }} | {{ B(r.audits.corpora["pg19-train"].tokens) }} | {{ f2(r.audits.corpora["pg19-train"].bytes_per_token) }} | DeepMind 发布的 Project Gutenberg 长篇书籍，文件列表 revision `4d28bd7`，Apache-2.0[^pg19] |
| PG-19 validation | 训练过程中的验证、消融 | {{ n(r.audits.corpora["pg19-validation"].documents) }} | {{ M(r.audits.corpora["pg19-validation"].tokens, 2) }} | {{ f2(r.audits.corpora["pg19-validation"].bytes_per_token) }} | 同上，官方划分 |
| PG-19 test | 最终评测 | {{ n(r.audits.corpora["pg19-test"].documents) }} | {{ M(r.audits.corpora["pg19-test"].tokens, 2) }} | {{ f2(r.audits.corpora["pg19-test"].bytes_per_token) }} | 同上，官方划分 |
| FineWeb-Edu `sample/10BT`，分片 013 的前 20,000 篇 | 最终评测 | {{ n(r.audits.corpora["fineweb-edu-heldout-dedup"].documents) }} | {{ M(r.audits.corpora["fineweb-edu-heldout-dedup"].tokens, 2) }} | {{ f2(r.audits.corpora["fineweb-edu-heldout-dedup"].bytes_per_token) }} | 同 FineWeb-Edu；已剔除与训练分片全文重复的 {{ n(r.audits.corpora["fineweb-edu-heldout-dedup"].excluded_duplicate_documents) }} 篇 |

两种语料互补：FineWeb-Edu 是经过质量分类器筛选的网页，文档短（平均约 1,000 个 token），决定模型的一般语言能力；PG-19 的每篇文档是一整本书（平均约 10 万个 token），一个 4,096 token 的窗口几乎总是落在同一本书内部，是真正需要长上下文的数据。

**Tokenizer。** 全部使用 GPT-2 的 BPE 词表（\(V=50{,}257\)）。每篇文档编码为 `<|endoftext|>` 加正文，文档首尾相接成一条 token 流。选用现成词表而不是自己训练，是为了能和公开模型在同一词表下直接比较损失。

**混合与采样。** 每个训练样本先以 50% 的概率选择来源，再从该来源的 token 流里均匀随机取一个长度为 \(L+1\) 的窗口（有放回）。窗口可以跨文档边界。按训练时的随机种子重放采样器，实际比例和覆盖情况如下：

{% table sampler %}

整个训练只用到每个语料的一小部分，绝大多数训练窗口只出现一次。

### 3.3 训练

| 项目 | 取值 |
|:--|:--|
| 序列长度 \(L\) / batch | 4,096 / 16 条序列（每步 65,536 个 token） |
| 训练步数 / token 数 | 12,000 / {{ n(r.training.swa.tokens_seen) }} |
| 优化器 | AdamW，\(\beta_1=0.9\)，\(\beta_2=0.95\)，权重衰减 0.1（只作用于矩阵） |
| 学习率 | 峰值 \(6\times10^{-4}\)；前 480 步线性预热，之后余弦衰减到 0 |
| 外循环梯度裁剪 | 全局范数 1.0 |
| 内循环（仅 TTT-E2E） | \(\eta=1.0\)，\(\tau=1.0\)；\(\eta\) 在前 1,200 步从 0.1 线性升到 1.0 |
| 精度 | 参数 float32，矩阵乘法 bfloat16 |
| 随机种子 | 0（四个模型的初始化种子和数据顺序都相同） |
| 硬件 | 每个模型独占一张 NVIDIA H200（143,771 MiB），不做多卡并行 |

超参数没有针对任何一个模型单独调过：四个模型共用上表，内循环的 \(\eta\) 和 \(\tau\) 取实现的默认值。训练 token 数约为原论文 125M 模型基础配方（25 亿 token）的 31%，受限于两张卡一晚上的预算；这是本报告最主要的规模限制。

### 3.4 评测协议

**Held-out 语言建模。** PG-19 test 的每本书从头切成互不重叠的 4,096 token 窗口（丢弃末尾不足一个窗口的部分），共 {{ n(r.fig_context_4k.arms.swa.windows) }} 个窗口、{{ M(r.fig_context_4k.arms.swa.tokens, 2) }} 个预测位置。每个窗口从空状态开始：KV 缓存清空，快权重回到 \(\phi_0\)。于是窗口内第 \(t\) 个位置的损失，就是“恰好有 \(t\) 个 token 上下文时”的预测损失，按位置分桶即可得到损失随上下文长度变化的曲线。FineWeb-Edu held-out 的文档很短，按训练时的方式把文档首尾相接，取前 2,048 个窗口。我们报告每 token 的平均负对数似然（NLL，单位 nat）、困惑度和 bits/byte。

**统计。** 四个模型评测的是同一批窗口，因此模型之间的差值按配对方式估计：PG-19 以书为单位、FineWeb-Edu 以窗口为单位做 2,000 次 bootstrap 重采样，给出 95% 置信区间。这个区间反映评测集的抽样误差，不包含重新训练带来的随机性；后者用第 4.9 节的多随机种子实验单独估计。

**超出训练长度。** 把窗口加长到 32,768（训练长度的 8 倍），其余协议不变，不做任何微调。

**公开 benchmark。** 所有任务都是零样本、按对数似然评分：多选题对每个候选项计算 \(\log p(\text{候选}\mid\text{题干})\) 并取最大者，提示模板、候选项拼接方式和长度归一化都与 lm-evaluation-harness[^lmeval] 的同名任务一致。为了确认这一点，我们用自己的评测脚本和 lm-evaluation-harness 分别评测 GPT-2，逐项对比（第 4.7 节）。不使用 few-shot 示例，不针对任何 benchmark 做训练、提示词搜索或检查点挑选；所有模型都只评测训练结束时的最后一个检查点。

**公开参考模型。** 为了给出绝对水平的参照，用同一套脚本评测了一组规模相近的公开模型。它们的训练数据量比本报告的模型大一到三个数量级，这组数字用于定位，不是匹配预算的比较。
