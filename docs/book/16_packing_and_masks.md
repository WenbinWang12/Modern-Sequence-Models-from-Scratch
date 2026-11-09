# 9　Packing、标签与 Mask：一个 batch 是怎样构造出来的

[English](en/16_packing_and_masks.md)

## 模型真正训练的不是文档

训练循环看到的是 `input_ids`、`labels` 和 `loss_mask`，不是“这是一篇文章”。一个简单的 next-token 样本把：

~~~text
tokens:     [x0, x1, x2, x3]
input_ids:  [x0, x1, x2]
labels:     [x1, x2, x3]
mask:       [ 1,  1,  1]
~~~

输入位置 `i` 的预测目标是原序列的 `i+1`。如果把 input 和 label 取成相同序列，模型就不再执行标准的 next-token 任务；如果 shift 多一位，模型会学习错位目标，却可能仍然产生有限 loss。

在当前实现中，`TokenDataset.rows(starts, length)` 返回长度为 `length` 的 token 窗口，并返回去掉第一列后的 mask。`BatchStream.next_batch_with_masks()` 再把窗口交给训练器。阅读数据代码时，先确认这一处 shift，再看模型的 loss。

## 独立文档与连续流

有两种常见 packing 语义。

**独立文档 packing** 在每篇文档后添加 EOS，并保证注意力或 loss 不跨越边界。它容易解释：文档结束后，模型不会把下一篇文档当成上一篇的自然延续。

**连续流 packing** 将多个文档连接成一个 token 流，训练效率高，但必须定义 EOS 和边界。若只加入 EOS 而仍允许 attention 跨越它，模型可以学习跨文档的统计关系；这可能是有意的，也可能泄漏文档组织信息。

当前仓库的 `prepare_data` 和 `prepare_sharded_data` 会把每个输入文档编码后追加 EOS，再把 token 依次写入分片。`TokenDataset.rows` 按全局 token offset 取窗口，因此一个窗口可能跨越 EOS 和文档边界；它没有实现“遇到 EOS 就重置注意力”的 block mask。正文应把这个事实写出来，不能把当前实现描述成独立文档 packing。

## Loss mask 与 padding

mask 的含义是“这个 label 是否参与 loss”，不是“这个 token 是否存在”。prompt-response 监督可以让 prompt 进入上下文但 mask 为 0；padding 也可以 mask 为 0。两个 0 的因果意义不同，后续评估和数据统计可能需要区分来源。

`token_cross_entropy` 会将 mask 为 0 的 target 替换成安全 ID，然后把对应 loss 置零。替换是为了避免无效 ID 触发索引错误，不是让模型学习这个安全 ID。平均 loss 的分母是有效 mask 的和；如果全部 mask 为 0，代码用 1 防止除零，但这个 batch 应被数据 preflight 发现。

跨设备或跨样本聚合时，也要用有效 token 数作权重。直接平均每个样本的 loss，会让一条有效 token 很少的样本和一条有效 token 很多的样本拥有相同权重。

## 一个可以手算的例子

设 token 序列为 `[10, 11, 12, 2, 20]`，其中 2 是 EOS。长度为 4 的训练窗口是：

~~~text
input_ids = [10, 11, 12, 2]
labels    = [11, 12,  2, 20]
~~~

如果我们把 EOS 后的 20 视为下一篇文档的第一个 token，最后一项是否训练取决于 packing 约定。若希望文档之间不互相监督，mask 可以是 `[1,1,1,0]`；若把整个 token 流当作连续语料，则是 `[1,1,1,1]`。两种设置训练的是不同条件分布。

这就是为什么“加了 EOS”还不足以说明边界语义。必须同时写出 input、label、mask 和 attention mask。

## 当前训练路径

`ttt/inner_loop.sequence_loss` 接收 `tokens` 和可选的 `loss_mask`，将序列拆成 ttt_chunk_size 的块；尾部 padding 的位置 mask 为 0。`training.py` 按每个样本有效 token 数聚合外循环 loss。

这段代码解决了 token-level mask 和 TTT chunk 的关系，但没有提供按文档边界重置 KV cache 的通用接口。若要加入独立文档 packing，需要在数据和模型状态之间新增契约，而不是只修改 mask。

下一章将 token 分片、mask 分片和 manifest 放进一个可恢复的 Dataset；这样我们可以检查文件内容，而不只是检查 batch shape。
