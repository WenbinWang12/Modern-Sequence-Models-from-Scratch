# 评估与统计协议：怎样让 loss 支持一个结论

## 评估不是训练日志的最后一列

训练 loss 回答“模型是否在拟合它反复看到的训练分布”。它不能单独回答模型是否能处理未参与更新的数据、是否记住了重复文本、是否能完成一个任务，或某个新模块是否值得它增加的计算成本。

本章把质量证据分成数据、指标、统计和成本四层。后面的消融与 Frontier 页面都应复用这里的协议。当前仓库的 tiny 运行和 retrieval probe 只建立入口，不自动产生正式质量结论。

## 先固定评估对象

| 对象 | 能回答的问题 | 不能单独回答的问题 |
|---|---|---|
| train | 优化器是否降低了训练目标 | 泛化、污染、真实能力 |
| validation | 配置和训练阶段之间如何选择 | 最终无偏报告 |
| test | 已冻结方案的最终比较 | 反复调参 |
| task probe | 某个任务格式下的表现 | 所有语言能力 |
| long-context probe | 指定上下文协议中的记忆/检索 | 一般推理能力 |

测试集一旦参与模型选择，就已经承担 validation 的角色。每份评估都要记录数据版本、tokenizer 身份、上下文长度、mask 规则和是否允许 TTT 更新。

## Token-weighted NLL 与 perplexity

对第 \(t\) 个目标，负对数似然是

$$
\ell_t=-\log p_\theta(y_t\mid x_{\le t}).
$$

令 \(m_t\in[0,1]\) 为监督权重，\(W=\sum_t m_t\)。一个评估集合的 token-weighted NLL 为

$$
\mathcal L_{\mathrm{tok}}
=\frac{\sum_t m_t\ell_t}{\sum_t m_t}
\qquad W>0.
$$

perplexity 只是它的指数变换：

$$
\operatorname{PPL}=\exp(\mathcal L_{\mathrm{tok}}).
$$

不能先对每个窗口取平均，再把窗口平均值简单平均，除非每个窗口的有效权重完全相同。应保存 loss 分子、权重分母、窗口数、正 mask 位置数和监督权重总和；分数 mask 要报告权重总和，不要把它伪装成整数 token 数。

## 文档边界和污染

连续 token 流可以让窗口跨过 EOS，也可能让训练和评估共享相邻内容。评估协议必须说明：

1. 文档是否独立编码；
2. EOS、BOS 和 padding 是否参与 loss；
3. 窗口是否可以跨文档；
4. TTT 快权重是否在文档之间重置；
5. 评估 tokenizer 是否来自训练集合并保持冻结。

一份验证集内部没有精确重复，不等于它没有训练镜像或近重复。报告至少应保存全文匹配、规范化匹配、长子串或近似检索中实际执行了哪些方法，以及哪些污染检查没有执行。

## 生成、检索和长上下文任务

生成任务必须固定 prompt、采样参数、最大新 token 数、停止 token 和随机种子。温度、top-k 和 top-p 会改变输出分布，因此不能只报告“同一个模型生成得更好”。

检索探针可以使用 exact match、规范化后的字符串匹配或 token-level F1，但必须写清楚答案如何规范化。答案不能在训练数据或 tokenizer 训练文本中泄漏；自定义的少量 probe 只能回答这个 probe 的问题，不能冒充完整 benchmark。

长上下文的 TTT 评估应使用同一个模型和 checkpoint 做成配对实验：同一上下文分别运行关闭更新和开启更新的路径，只比较后续未见目标或任务得分的差异，并同时记录额外内循环时间、更新次数和状态大小。这回答的是“更新对该 checkpoint 和该协议的影响”，不等于两个方法都从零训练后的最终质量比较。

## 多 seed 与不确定性

一次运行的差异可能来自初始化、数据顺序或设备归约。至少要把 seed、数据指纹、配置摘要和硬件记录在结果中。若报告多个 seed，应同时给出每个原始结果和聚合统计：

$$
\bar x=\frac1n\sum_{i=1}^{n}x_i,
\qquad
s^2=\frac1{n-1}\sum_{i=1}^{n}(x_i-\bar x)^2.
$$

均值和标准差描述 seed 间变化，不等于每个 token 的置信区间。对文档级或样本级指标，可以按样本重采样构造 bootstrap 区间；重采样单位应与独立性假设一致，不能把同一窗口中的相关 token 当作独立文档。

## 统一结果记录

下面是建议的结果 schema；它是待接入规范，当前 runner 不会自动写出其中所有字段。一个最小结果对象应能让读者回到输入和代码：

~~~json
{
  "source_commit": "...",
  "config_sha256": "...",
  "data_manifest_sha256": "...",
  "tokenizer": {"name": "...", "revision": "...", "digest": "..."},
  "split": "validation",
  "sequence_length": 2048,
  "adaptation": {"enabled": false, "inner_steps": 0},
  "metrics": {
    "nll_numerator": 0.0,
    "supervision_weight": 0.0,
    "perplexity": 0.0
  },
  "seed": 0,
  "status": "not_checked"
}
~~~

这里的零值只是 schema 示例，不能当成测量结果。正式产物还应保存每个 seed 的原始 JSON、日志摘要、失败原因和成本字段，例如稳态 step time、有效 token/s、峰值显存和内循环计算量。

## 当前仓库的边界

当前的 eval_retrieval.py 和 Frontier 页面提供 checkpoint loss、少量 retrieval probe 及长上下文配对评估的入口。它们没有实现完整 RULER、统一多 seed 聚合、污染数据库或自动 bootstrap 报告。实验章的 scaling 拟合也不会生成训练质量结果。

因此，当前代码可以支持“按声明的入口运行一个评估片段”的结论；它还不能支持“模型在正式 benchmark 上更强”或“某个结构在相同成本下普遍更好”的结论。缺少证据时，结果状态应写成 not_checked，而不是填入预测数字。

## 本章小结

评估先固定对象和污染边界，再定义按有效权重聚合的指标，随后记录任务协议、多个 seed、不确定性和成本。只有这些字段与配置、数据和 tokenizer 身份一起保存，loss 才能成为可审计的实验结果。
