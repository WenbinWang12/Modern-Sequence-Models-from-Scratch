# 5. 内循环与外循环

## 两种优化发生在哪里

TTT 的训练包含两种时间尺度。内循环在一个序列块内部更新快权重，使模型适应当前上下文；外循环更新静态模型参数，使模型学会如何进行这种适应。两者使用的状态、损失和优化器不能混为一谈。

对一个 chunk 的简化表示是：

~~~text
fast_0 ──forward(prefix)──> loss_0 ──SGD──> fast_1
fast_1 ──forward(suffix)──> loss_1
                              │
                              └── outer loss → static parameters
~~~

如果某个 token 的标签在快权重更新前就被使用，它会改变之后的预测；如果标签被提前暴露给同一个预测，则会产生数据泄漏。实现中应明确 predict → loss → update 的顺序。

## 梯度语义

内循环更新可以选择保留或截断高阶梯度。保留高阶梯度更接近严格的 meta-learning 目标，但显存和编译成本更高；截断更新更适合先验证系统路径。无论采用哪一种，都要在实验记录中写明，因为两者不是同一个算法。

外循环通常使用 AdamW，负责静态参数、归一化参数和可能的路由参数。快权重可以使用简单 SGD 或专用的局部优化器。它们的学习率、weight decay 和梯度裁剪应分别记录。

## 代码入口

- ttt/inner_loop.py 实现块级预测、token loss、SGD 和 scan。
- ttt/loss.py 计算分块交叉熵，并处理 padding 与有效 token mask。
- ttt/training.py 管理外循环 AdamW、梯度裁剪和训练状态。
- ttt/checkpoint.py 同时保存静态参数、优化器和采样状态；会话快权重由 runtime 单独管理。

一个实用的阅读顺序是先看 inner_loop.py 的单块纯函数，再看 training.py 如何把它嵌入 batch，最后看 checkpoint 如何恢复两种状态。这样能把算法正确性与调度细节分开。

## 对照实验

设置 inner.enabled=false 可以得到静态的 C 路径；设置为 true 是 F 路径。两者必须使用相同的 tokenizer、架构宽度、训练 tokens 和数据顺序。只改变内循环开关，才能把差异解释为测试时更新带来的变化。

在远端 GPU 作业中，验证报告应至少记录：内循环步数、更新前后快权重范数、有限 loss、外循环梯度范数以及恢复后的下一步 loss。tests/test_meta_gradients.py 和 tests/test_grouped_gradients.py 覆盖这些边界；若只检查最终 loss，可能遗漏梯度路径错误。

## 下一章要连接什么

内循环让模型拥有会变化的状态，注意力缓存和 Gated Delta 则决定这些状态如何跨 token、跨 chunk 和跨请求传递。下一章讨论训练图之外的流式运行时。
