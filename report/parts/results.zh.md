## 4 结果

### 4.1 训练过程

<figure markdown="span" class="report-figure">
  ![训练损失与 held-out 损失随训练 token 数的变化](assets/report/fig_training.light.svg#only-light)
  ![训练损失与 held-out 损失随训练 token 数的变化](assets/report/fig_training.dark.svg#only-dark)
  <figcaption>图 2　训练过程。左：训练损失（指数滑动平均）。中、右：每 1,000 步保存的检查点在 PG-19 validation（每本书前 5 个窗口，共 250 个）和 FineWeb-Edu held-out（256 个窗口）上的损失。四个模型在每一步读到的 batch 完全相同。</figcaption>
</figure>

{% include parts/results_4_1.zh.md %}

### 4.2 主结果：4K 上下文下的 held-out 损失

<div class="report-table" markdown>

{% table main %}

</div>

表 1　训练结束时的检查点在两个 held-out 集上的损失（越低越好）。差值是与同一批窗口上全注意力的配对差，方括号内为 95% 置信区间。bits/byte 按语料的平均字节/token 换算。

{% include parts/results_4_2.zh.md %}

### 4.3 损失随上下文长度的变化

<figure markdown="span" class="report-figure">
  ![PG-19 test 上损失随上下文长度的变化](assets/report/fig_context_4k.light.svg#only-light)
  ![PG-19 test 上损失随上下文长度的变化](assets/report/fig_context_4k.dark.svg#only-dark)
  <figcaption>图 3　PG-19 test 上，损失随窗口内已读 token 数的变化（每 64 个位置取平均）。左：各模型的损失。右：各模型与全注意力的配对差，阴影是按书重采样的 95% 置信区间。滑窗注意力的窗口是 512，TTT-E2E 每 256 个 token 更新一次。</figcaption>
</figure>

<div class="report-table" markdown>

{% table ranges %}

</div>

表 2　PG-19 test 上按上下文位置分段的损失，以及模型之间的配对差（95% 置信区间）。

{% include parts/results_4_3.zh.md %}

### 4.4 收益来自哪里

{% include parts/results_4_4.zh.md %}

### 4.5 超出训练长度：32K 上下文

<figure markdown="span" class="report-figure">
  ![超出训练长度时的损失](assets/report/fig_context_32k.light.svg#only-light)
  ![超出训练长度时的损失](assets/report/fig_context_32k.dark.svg#only-dark)
  <figcaption>图 4　把评测窗口加长到 32,768 个 token（训练长度的 8 倍）时 PG-19 test 上的损失，每 256 个位置取平均，竖线标出训练长度。虚线是不做任何截断、直接外推的全注意力；“只看最近 4K”是把同一个全注意力模型的可见范围限制在训练时见过的长度内。右图是各模型与滑窗注意力的配对差。</figcaption>
</figure>

<div class="report-table" markdown>

{% table long %}

</div>

表 3　32K 窗口内按位置分段的损失与配对差（95% 置信区间）。所有模型都只在 4,096 长度上训练过。

{% include parts/results_4_5.zh.md %}

### 4.6 压缩丢掉了什么：逐字回忆

<figure markdown="span" class="report-figure">
  ![逐字回忆探针](assets/report/fig_recall.light.svg#only-light){ width="70%" }
  ![逐字回忆探针](assets/report/fig_recall.dark.svg#only-dark){ width="70%" }
  <figcaption>图 5　逐字回忆探针。把窗口内第 128–191 个 token 原样复制到更靠后的位置，纵轴是重复段中间 32 个 token 上的损失比未改动文本降低了多少；值越大说明模型越能利用此前出现过的原文。</figcaption>
</figure>

<div class="report-table" markdown>

{% table recall %}

</div>

表 4　重复段上的损失降低量（nat，方括号内为 95% 置信区间；PG-19 test 每本书前 4 个窗口，共 400 个）。

<div class="report-table" markdown>

{% table niah %}

</div>

表 5　单针检索：在干草堆文本里藏一句“One of the special magic numbers for &lt;键&gt; is: &lt;7 位数字&gt;.”，然后提问这个键对应的数字。表中是答案的每 token 对数似然比“针被换成无关键值对”时降低了多少（nat，越大说明越用到了针）。干草堆为 PG-19 test 的书籍正文，针均匀放在 10 个深度上；提示词和针的句式取自 RULER。

{% include parts/results_4_6.zh.md %}

### 4.7 公开 benchmark 与公开模型

<div class="report-table" markdown>

{% table benchmarks_ours %}

</div>

表 6　本报告四个训练配置的零样本准确率（%），并额外列出关闭测试时更新的 TTT 检查点。HellaSwag、PIQA、ARC、OpenBookQA 报告按候选项长度归一化的准确率，其余为普通准确率；“平均”是十项的算术平均。

<div class="report-table" markdown>

{% table reference_models %}

</div>

表 7　用于对照的公开模型。训练数据一列摘自各自的模型卡片或论文（2026 年 11 月核对）；它们的训练量比本报告的模型大一到三个数量级。

<div class="report-table" markdown>

{% table benchmarks_reference %}

</div>

表 8　用同一脚本、同一协议评测的公开模型（%）。参数量由加载的权重直接统计。

<div class="report-table" markdown>

{% table perplexity %}

</div>

表 9　语言建模指标（越低越好）。WikiText-103 test 和 PG-19 test 按各模型的最大上下文切成互不重叠的窗口计算，与 lm-evaluation-harness 的做法相同；词级困惑度按空白分词的词数归一化，可以跨 tokenizer 比较。

{% include parts/results_4_7.zh.md %}

### 4.8 消融

<figure markdown="span" class="report-figure">
  ![消融实验](assets/report/fig_ablations.light.svg#only-light)
  ![消融实验](assets/report/fig_ablations.dark.svg#only-dark)
  <figcaption>图 6　TTT-E2E 的四项设计选择（8 层、宽度 512、上下文 2,048 的模型，每次只改变一项）。横线是同规模的全注意力和滑窗注意力；最右一栏里滑窗注意力随窗口一起变化。</figcaption>
</figure>

<div class="report-table" markdown>

{% table ablations %}

</div>

表 10　消融结果（PG-19 validation，上下文 2,048）。“关闭更新时”是同一个检查点在评测时不做测试时更新的损失。

{% include parts/results_4_8.zh.md %}

### 4.9 随机种子与模型规模

<div class="report-table" markdown>

{% table seeds %}

</div>

表 11　三个随机种子（初始化和数据顺序同时改变）下的 PG-19 validation 损失。同一种子下各模型读到的数据相同，所以差值按种子配对。

<div class="report-table" markdown>

{% table scaling %}

</div>

表 12　三种模型规模在相同的 1.64 亿训练 token 下的 PG-19 validation 损失（上下文 2,048，窗口 256）。

<figure markdown="span" class="report-figure">
  ![模型规模与训练量](assets/report/fig_scaling.light.svg#only-light)
  ![模型规模与训练量](assets/report/fig_scaling.dark.svg#only-dark)
  <figcaption>图 7　左：表 10 的三种规模。右：主实验训练过程中，PG-19 validation 上模型之间的损失差随训练 token 数的变化。</figcaption>
</figure>

{% include parts/results_4_9.zh.md %}

### 4.10 成本

<div class="report-table" markdown>

{% table cost %}

</div>

表 13　单张 H200 上的训练成本（batch 16 × 4,096 token，随机输入，取 4–5 步的中位数）。全注意力和无更新对照开启了逐层梯度检查点，否则全注意力超出单卡显存。

<figure markdown="span" class="report-figure">
  ![训练与前向吞吐](assets/report/fig_cost.light.svg#only-light)
  ![训练与前向吞吐](assets/report/fig_cost.dark.svg#only-dark)
  <figcaption>图 8　左：表 11 的训练吞吐。右：不同上下文长度下的前向吞吐（整段读入并计算每个位置的损失，每批固定 32,768 个 token；TTT-E2E 包含测试时更新）。</figcaption>
</figure>

<div class="report-table" markdown>

{% table sessions %}

</div>

表 14　单个推理会话在读入不同长度的上下文之后保存的状态，以及随后逐 token 解码的速度（批大小 1）。状态大小由实际分配的数组统计。

{% include parts/results_4_10.zh.md %}

## 5 数据审计

<div class="report-table" markdown>

{% table overlap %}

</div>

表 15　评测题目与训练语料的逐字重合检查。对每道题取规范化（小写、压缩空白）后的题干前 80 个字符，在全部训练原文（{{ GB(r.audits.corpora["fineweb-edu-train"].utf8_bytes + r.audits.corpora["pg19-train"].utf8_bytes) }} GB）中做精确匹配。

{% include parts/results_5.zh.md %}

## 6 讨论与局限

{% include parts/discussion.zh.md %}
