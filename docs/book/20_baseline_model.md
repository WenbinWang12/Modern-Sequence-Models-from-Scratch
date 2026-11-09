# 14　Dense Causal LM 基线：先证明训练系统能学习

## 基线不是最小的玩具

我们要比较 TTT、Loop 和 MoE，必须先有一个不会把数据、目标或恢复问题隐藏起来的 dense baseline。它不追求结构新颖，而要把输入 token 变成下一个 token 的 logits，并在固定数据上稳定降低 loss。

基线至少固定：tokenizer 和特殊 token、训练/验证 manifest、sequence length、有效 token 预算、初始化、优化器、学习率计划、随机种子、精度、硬件和 checkpoint 格式。之后的结构实验只改变声明的架构变量。

## 从 logits 到 loss

模型产生 `z_{t,v}`，其中 `t` 是位置、`v` 是词表 ID。交叉熵为：

$$
\ell_t=-\log\frac{\exp z_{t,y_t}}
{\sum_{v=1}^{V}\exp z_{t,v}}.
$$

直接对大词表计算完整 softmax 可能占用显存，因此仓库的 `loss.py` 提供按 token 和 vocab block 处理的路径。数学目标不因分块而改变；实现只是把求和拆成可执行的块。

有效平均 loss 为：

$$
\mathcal L=
\frac{\sum_t m_t\ell_t}{\max(\sum_t m_t,1)},
$$

其中 `m_t` 是 mask。最后的 max 防止全零 mask 时除零，但全零样本本身应该被 preflight 拒绝。

## 基线配置到底关闭了什么

~~~text
num_experts = 0
use_fast_mlp = false
inner.enabled = false
loop_mode = none
~~~

这些开关表达“没有额外动态状态、没有路由、没有深度循环”。注意力仍然可以是 sliding 或 full；如果要研究注意力类型，需把它作为独立变量，不能把所有默认项混成一个 baseline 名称。

仓库的 configs/tiny.json 是训练配置包装，model 子对象最终由 TTTE2EConfig 解析。configs/125m 目录下的配置是配方和消融候选，125M 是档位名，不代表启用 MLP′、专家或大词表后的精确总参数。

## 训练一步如何发生

`ttt/training.py` 先根据配置创建 AdamW 和学习率 schedule。每步：

1. BatchStream 取窗口与 mask；
2. `sequence_loss` 对每个样本计算 forward、内循环和有效 loss；
3. 外循环对静态参数求梯度；
4. 梯度裁剪并由 Optax 更新参数和优化器状态；
5. 记录 loss、梯度范数、inner lr、快权重范数和更新时间；
6. 按 checkpoint_every 保存状态。

当 `inner.enabled=false` 时，`sequence_loss` 仍沿用同一条路径，只是不更新 fast weights。这个共享路径很重要：C/F 对照改变的是更新行为，而不是另一套模型实现。

## 用 loss 判断“可学习”

教学运行只需要观察方向和数值健康：

- loss 是有限值；
- 多步运行没有立即变成 NaN；
- 梯度范数不是持续为零或无界增长；
- 重复恢复后，下一步与未中断运行接近；
- held-out loss 的计算使用独立 manifest。

“loss 下降”本身不是能力证明。随机数据也可能在有限步内被记忆，极小样本更容易过拟合。tiny baseline 的任务是验证接口和优化，不是证明架构可扩展。

可以记录训练 loss 与 held-out loss 的差值。如果训练下降而 held-out 上升，先检查数据切分、重复和采样，再讨论正则化。数据量太小时，曲线只能作为排错信号。

## 参数、激活和实际成本

参数量可由模型树叶元素数量统计，但不等于每步激活内存。TTT 还会保存或重算快权重、梯度和 chunk 状态；MoE 的总参数可能很大，但每个 token 激活的专家较少。

`ttt/profile.py` 当前提供参数量、fast MLP 参数量、prefill/decode 时间、状态字节、设备峰值内存和 XLA cost analysis。cost analysis 是估计，不是硬件实测；稳态吞吐还需要真实设备运行并排除首步编译时间。

## 基线通过后的比较纪律

对于任一新模型，实验表至少包含：

| 维度 | 必须固定或报告 |
|---|---|
| 数据 | manifest、有效 token、来源与污染状态 |
| 模型 | 总参数、激活参数、层结构、词表 |
| 算法 | optimizer、学习率、内循环、seed |
| 系统 | GPU、设备数、精度、编译与通信 |
| 结果 | train/held-out loss、吞吐、显存、恢复 |

如果 TTT 只在更长训练或更多 token 下胜出，应报告预算差异；如果 Loop 每步更慢但质量更好，应同时报告质量-计算曲线。这样 baseline 才是研究参照，而不是一个被快速越过的占位页。

下一部分进入模型内部：先逐层追踪一个 token，再解释 TTT 的快权重究竟在何时改变。
