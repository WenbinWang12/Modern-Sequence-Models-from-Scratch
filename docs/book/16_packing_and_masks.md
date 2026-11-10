# 10　Packing、标签与 Mask：一个 batch 是怎样构造出来的

## 模型真正训练的不是文档

模型接收整数数组。为解释 next-token 任务，先把输入、目标和监督位置分别记为 \(X,Y,M\)，示例变量名使用 <code>input_ids</code>、<code>labels</code> 和 <code>loss_mask</code>。当前仓库的 loader 实际返回 <code>(tokens, masks)</code>，由 <code>sequence_loss</code> 再移位构造输入和目标；并不是返回这三个 key 的字典。

先用四个可读的符号代替 token：原序列 `[A, B, C, D]` 的输入是 `[A, B, C]`，标签是向左错开一位的 `[B, C, D]`。模型在看到 `A` 后预测 `B`，看到 `A,B` 后预测 `C`，看到 `A,B,C` 后预测 `D`；这就是“下一个 token”而不是复制当前 token。

\[
\begin{aligned}
\text{tokens}&=(s_1,s_2,s_3,s_4),\\
X&=(s_1,s_2,s_3),\\
Y&=(s_2,s_3,s_4),\\
M&=(1,1,1).
\end{aligned}
\]

输入位置 \(i\) 的预测目标是原序列的 \(i+1\)。如果把 input 和 label 取成相同序列，模型就不再执行标准的 next-token 任务；如果 shift 多一位，模型会学习错位目标，却可能仍然产生有限 loss。

在当前实现中，`TokenDataset.rows(starts, length)` 的 `length` 指原始 token 数：返回 `length` 个 token 和去掉第一列后的 `length - 1` 个目标权重。因此 `rows(starts, 5)` 对应 4 个预测位置。模型配置中的 `sequence_length` 指输入位置数，`BatchStream.next_batch_with_masks()` 会请求 `sequence_length + 1` 个原始 token，再把窗口交给训练器。阅读数据代码时，先确认这一处 shift，再看模型的 loss。

如果 `[A,B,C,D]` 中的 `D` 是 padding，第三个预测位置的 mask 应为 0；如果 `B` 来自 prompt，prompt 的 mask 也可以为 0，但它仍然可以作为上下文被模型看到。mask 控制“是否计入损失”，不等于“是否从输入中删除”。

## 独立文档与连续流

有两种常见 packing 语义。

**独立文档 packing** 要阻止注意力和携带的模型状态读取前一篇文档，并明确处理边界处的 next-token 目标。仅把一个 loss mask 置零不能实现这种隔离：它删除一个损失项，却不会删除上下文。实现时需要文档感知的 attention 边界，或分开调用模型并重置状态。

**连续流 packing** 将多个文档连接成一个 token 流，训练效率高，但必须定义 EOS 和边界。若只加入 EOS 而仍允许 attention 跨越它，模型可以学习跨文档的统计关系；这可能是有意的，也可能泄漏文档组织信息。

当前仓库的 `prepare_data` 把数据写成单个 NPY，`prepare_sharded_data` 写成分片和 manifest。二者都将保留的文档编码，并在 tokenizer 定义了 EOS 时追加 EOS。`TokenDataset.rows` 按全局 token offset 取窗口，因此一个窗口可能跨越 EOS 和文档边界；它没有实现“遇到 EOS 就重置注意力”的 block mask。这两条路径都保留连续上下文，不能描述成独立文档 packing。

两条路径还有必须记录的监督差别：单 NPY 没有独立 mask，loader 默认所有目标权重为 1；分片路径会将每篇文档开头的 BOS 权重置零。用 byte tokenizer 编码两篇文档 A、B，流为 `[1,68,2,1,69,2]`，其中 1 是 BOS、2 是 EOS。取整个流时，单 NPY 的五个目标权重为 `[1,1,1,1,1]`，分片路径为 `[1,1,0,1,1]`；第三个目标就是“前文 EOS 预测后文 BOS”。后者仍允许 B 的预测读取 A，并没有隔离上下文。另外，分片路径跳过纯空白文本，单 NPY 路径仍编码它。因此切换这两条准备入口可能改变监督和 token 流，不能只记录为存储格式变化。

## Loss mask 与 padding

mask 的含义是“这个 label 是否参与 loss”，不是“这个 token 是否存在”。prompt-response 监督可以让 prompt 进入上下文但 mask 为 0；padding 也可以 mask 为 0。两个 0 的因果意义不同，后续评估和数据统计可能需要区分来源。

`token_cross_entropy` 会将 mask 为 0 的 target 替换成安全 ID，然后把对应 loss 置零。替换是为了避免无效 ID 触发索引错误，不是让模型学习这个安全 ID。这里假设模型的 logits 有限；mask 不是修复模型数值故障的工具。

### 从单个位置的损失，到加权平均

假设模型为正确目标分配了概率 \(p_i\)。这个位置的交叉熵是 \(\ell_i=-\log p_i\)：正确答案获得的概率越大，代价越小。例如，概率为 \(1/2\) 时损失约为 \(0.693\)，概率只有 \(1/4\) 时约为 \(1.386\)。取自然对数，所以单位是 nat；这不是正确率，两个位置都猜中或猜错时，置信程度仍会改变损失。

仓库通常生成 \(m_i\in\{0,1\}\) 的 mask；loader 也接受 \([0,1]\) 中的分数，表示该目标的相对权重。令总权重 \(W=\sum_{i=1}^{L}m_i\)，定义

$$
\mathcal L=
\begin{cases}
\displaystyle\frac{\sum_{i=1}^{L}m_i\ell_i}{W},&W>0,\\[6pt]
0,&W=0.
\end{cases}
$$

例如两个位置的损失为 \(2,4\)，权重为 \(1/4,1/2\)。加权和为 \(2.5\)，总权重为 \(0.75\)，所以平均为 \(2.5/0.75=10/3\)。把权重一起乘 2，得到 \(1/2,1\)，平均仍是 \(10/3\)。整体缩放权重不应改变平均；只有各位置之间的相对权重影响结果。分数权重的和不是“有几个 token”，而是**总监督权重**。

实现只在 \(W=0\) 时选用安全分母 1。不能写成 \(\max(W,1)\)：上面的 \(W=0.75\) 会被错误改为 1，得到 2.5。全零 mask 没有可用监督，因此按约定返回零语言建模 NLL；数据检查仍应发现这种窗口，不能把零值解释成完美预测。这一结论只涉及 NLL，不自动约束 MoE 等模型的辅助项。即使 NLL 梯度为零，带动量或权重衰减的优化器也可能更新参数，因此“全零 mask”并不等于“整个训练步骤没有变化”。

跨样本时，必须先还原各自的加权和，再除以整个 batch 的权重总和。若第一条样本有 100 个有效目标、平均 loss 为 2，第二条只有 2 个有效目标、平均 loss 为 10，正确合并是 \((100\times2+2\times10)/102\approx2.157\)，不是 \((2+10)/2=6\)。跨设备遵循同样原则：先聚合分子与分母，不能让几乎没有监督的设备与拥有大量监督的设备各占一半权重。`tests/test_weighted_loss.py` 为分数权重、外循环梯度、全零权重和两设备稀疏监督写了回归检查；本轮尚未执行。

## 一个可以手算的例子

设教学用 token 序列为 \(s=(10,11,12,2,20)\)，其中 2 是 EOS；这些 ID 不是一条真实文本的编码。长度为 4 的训练窗口是 \(X=(10,11,12,2)\)、\(Y=(11,12,2,20)\)。

如果我们把 EOS 后的 20 视为下一篇文档的第一个 token，最后一项是否训练取决于 packing 约定。若希望文档之间不互相监督，mask 可以是 \(M=(1,1,1,0)\)；若把整个 token 流当作连续语料，则是 \(M=(1,1,1,1)\)。两种设置训练的是不同条件分布。

这就是为什么“加了 EOS”还不足以说明边界语义。必须同时写出 input、label、mask 和 attention mask。

## 用真实的 byte ID 看一次边界

在目标机器的仓库根目录运行下面的小例子。它按当前普通分片路径的规则，分别编码两篇单字符文档，再拼接；这里显式构造数组，是为了让每一步都能看见。

~~~bash
.venv/bin/python - <<'PY'
from ttt.data import ByteTokenizer

tokenizer = ByteTokenizer()
tokens, stored_mask = [], []
for text in ("A", "B"):
    document = tokenizer.encode(text) + [tokenizer.eos_token_id]
    tokens.extend(document)
    stored_mask.extend([0] + [1] * (len(document) - 1))
print("tokens:", tokens)
print("X:", tokens[:-1])
print("Y:", tokens[1:])
print("M:", stored_mask[1:])
PY
~~~

按 ByteTokenizer 的定义，字节 65、66 分别映射为 ID 68、69，BOS 为 1、EOS 为 2。因此预期的 token 流为 <code>[1,68,2,1,69,2]</code>，移位后为：

| 代码索引 | 输入 \(X\) | 目标 \(Y\) | \(M\) | 含义 |
|---|---:|---:|---:|---|
| 0 | 1 | 68 | 1 | BOS 后预测 A |
| 1 | 68 | 2 | 1 | A 后预测 EOS |
| 2 | 2 | 1 | 0 | 不为下一篇的 BOS 计损失 |
| 3 | 1 | 69 | 1 | 第二篇 BOS 后预测 B |
| 4 | 69 | 2 | 1 | B 后预测 EOS |

第三行没有直接损失，但第四行仍可以通过因果注意力读取之前的 A。由此可以看见两个不同问题：边界目标是否被监督，以及后文是否能访问前文。mask 只回答第一个问题。

## 当前训练路径

`ttt/inner_loop.sequence_loss` 接收 `tokens` 和可选的 `loss_mask`，将序列拆成 ttt_chunk_size 的块；尾部 padding 的位置 mask 为 0。`training.py` 按每个样本的监督权重和聚合外循环 NLL；在二值 mask 下，这个和才等于有效目标数。

这段代码解决了 token-level mask 和 TTT chunk 的关系，但没有提供按文档边界重置 KV cache 的通用接口。若要加入独立文档 packing，需要在数据和模型状态之间新增契约，而不是只修改 mask。

下一章将 token 分片、mask 分片和 manifest 放进一个可恢复的 Dataset；这样我们可以检查文件内容，而不只是检查 batch shape。

## 本章最小检查、预期证据与失败诊断

~~~bash
.venv/bin/python -m ttt.data \
  --input examples/sample.jsonl \
  --output outputs/packing-sample \
  --sharded --tokenizer byte --shard-tokens 256
~~~

随后可按上面的短例检查输入、目标和 mask 是否分别有 \(L\) 个位置。一个更严格的屏蔽目标检查是：保持全部输入、mask 和初始状态不变，只改变最后一个被屏蔽的目标 ID；它不应通过损失项影响结果。不能把这句话推广为“任意被屏蔽 token 都不影响状态”，因为同一个 token 还可能是后续预测的输入。

## 本章小结与下一问

Packing 定义窗口，label shift 定义预测目标，loss mask 定义哪些目标有监督；EOS 只表达边界，不自动决定边界之后的监督。下一章把这些数组写入带 hash 和 offset 的 Dataset manifest。
