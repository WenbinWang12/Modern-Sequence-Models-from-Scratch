表 13 把单步训练吞吐、峰值显存和参数量放在一起，表 14 则统计会话读取不同长度上下文后保留的实际数组状态以及逐 token 解码速度。全注意力、滑窗和 TTT 的成本随上下文的变化方式不同，单一的 tokens/s 不能概括这些差异。

{% if r.training_cost.ttt and r.training_cost.swa %}在表 13 所列的 4K 训练设置下，TTT-E2E 每秒处理的输入 token 数是滑窗注意力测得速率的 {{ f2(r.training_cost.ttt.tokens_per_second / r.training_cost.swa.tokens_per_second) }} 倍。这个比值绑定于表 13 中的 batch、序列长度、检查点策略和 H200 软件环境。{% endif %}

TTT-E2E 的额外成本来自计算内循环梯度、存储快权重和执行更新。实际部署还需要计入编译缓存、批大小、并发会话、检查点读写和失败恢复。这里的 H200 单卡测量用于复现实验的相对比较，不代表所有硬件和服务栈的绝对吞吐。
