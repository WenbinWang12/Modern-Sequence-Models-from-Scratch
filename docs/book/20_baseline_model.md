# 20. 标准 Causal LM 基线

## 基线为什么先于前沿结构

TTT、Linear Attention、Loop 和 MoE 的收益都需要相对于一个可靠的 dense baseline 判断。基线不是“最简单能跑的模型”，而是数据、tokenizer、优化器、训练预算和评估方式都固定的一条主线。

本章的目标是得到一个没有快权重、没有专家路由、没有额外循环的 next-token predictor。它先证明数据和训练系统能够学习，再成为后续结构消融的参照。

## 基线配置

~~~text
num_experts = 0
use_fast_mlp = false
inner.enabled = false
loop_mode = none
~~~

至少验证：

- next-token loss 能下降并保持有限；
- 梯度范数没有持续爆炸或变成零；
- checkpoint 能恢复参数、优化器和采样状态；
- batch、sequence length 和 mask 变化不会破坏 shape；
- 训练和推理的 causal 输出一致；
- held-out 数据的统计与训练数据不同且可追踪。

baseline 的参数量、训练 tokens、学习率和 warmup 必须写进实验记录。只报告最终 loss 而不报告有效 token 数，无法比较不同序列长度和 batch。

## 代码路径

模型注册和 dense block 位于 ttt/models；训练状态由 ttt/training.py 和 ttt/checkpoint.py 管理；配置示例在 configs/tiny.json 和 configs/125m。数据输入沿用前面章节的 manifest 和 preflight，不为 baseline 单独写一套数据读取逻辑。

先用 tiny 配置完成有限步训练，再使用同一代码路径扩大到 125M。扩大规模时只改变明确的配置字段，并重新记录编译时间、稳态吞吐、峰值显存和 checkpoint 大小。

## 公平比较

TTT、Loop 和 MoE 对照应固定：参数初始化规则、tokenizer、数据 manifest、训练 tokens、优化器、随机种子和硬件。参数量相同不等于计算量相同，因此还要记录实际 FLOPs、激活参数、通信、编译和推理成本。

若某个方法需要额外的内循环或 cache，报告要把这些成本计入，而不是只比较静态参数的数量。

## 进入下一部分

baseline 稳定后，才进入前向顺序、快权重和内外循环。这样当 TTT 的 loss 或长上下文表现发生变化时，我们知道变化来自结构，而不是数据管线或训练入口的漂移。
