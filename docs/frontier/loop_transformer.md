# Looped Transformer：深度方向的前沿专题

**快照日期：2026-11-07。状态：复现中。**

## 它研究什么

Looped Transformer 通过重复使用同一组层，增加有效计算深度，而不按循环次数增加独立参数。相关工作把它用于 latent reasoning、recurrent depth 和可变推理计算；早期 looped Transformer 结果已经显示，小模型重复执行共享层可能接近更深的非循环模型，但收益依赖监督方式、循环次数和公平的计算预算。[Looped Transformer 方向](https://arxiv.org/abs/2502.17416)

## 它和 TTT 的关系

两者作用在不同轴上：

```text
TTT：时间轴上的状态学习，token/block 到来时更新快权重
Loop：深度轴上的重复计算，同一层或同一 block 被多次执行
```

可以形成四类对照：

| 模型 | 时间更新 | 深度循环 |
|---|---:|---:|
| Dense baseline | 否 | 否 |
| Looped Transformer | 否 | 是 |
| TTT-E2E | 是 | 可选 |
| TTT + Loop | 是 | 是 |

## 必须控制的变量

- unique parameters；
- total training FLOPs；
- effective depth；
- loop count；
- sequence length 和训练 tokens；
- 是否对每次 loop 提供监督；
- residual 和 norm 的位置；
- 训练与推理时是否允许改变 loop count。

只比较参数量会高估 loop 的收益；只比较一次 forward 又会低估循环模型的实际计算成本。

## 实验路径

先使用当前配置里的 `loop_mode` 做功能验证，再增加真正的 shared-layer recurrent depth 配置。每个 loop count 单独记录编译时间、稳态时间、显存和 loss。最后再把 Loop 与 TTT 的快权重更新组合起来，验证它们是互补还是互相干扰。
