# TTT-E2E 与长上下文

**快照日期：2026-11-07。状态：主线方向。**

TTT-E2E 把长上下文建模描述为 continual learning：模型在读取测试上下文时继续通过 next-token 目标更新，把上下文压缩进权重状态。论文报告了 3B 规模和长上下文方向的实验，并将 TTT 与 Transformer、Mamba 2、Gated DeltaNet 等序列模型比较。[TTT-E2E 论文](https://arxiv.org/abs/2512.23675)

## Claim：快权重能否压缩长上下文

核心假设是：在相同的训练 token、隐藏宽度和实际计算预算下，测试时更新的快权重可以保存比固定 KV cache 更有用的上下文统计，同时保持可接受的首 token 延迟和稳态吞吐。这个假设还不是本仓库已经证明的结论。

## Relation：它在主线中的位置

这部分在本书中属于主线，但必须和前沿专题分开记录：

- 功能验证：内循环、缓存、梯度和恢复是否正确；
- 规模验证：固定硬件下首步编译和稳态吞吐；
- 长上下文验证：32K/128K 的真实数据、显存、稳定性和质量；
- 质量结论：RULER 或其他正式评测。

当前仓库已经覆盖前两层的小规模验证，正式 3B、128K 和完整 RULER 仍需要外部权重、数据和硬件条件。

## Controls：比较时固定什么

- tokenizer、数据 manifest、sequence length 和有效训练 token；
- hidden size、层数、初始化、优化器和精度；
- 实际 FLOPs、batch、chunk size 和硬件；
- KV cache、fast weights、session snapshot 的状态格式。

## Evidence：需要记录什么

每次运行至少记录前向数值、因果性、首 token 延迟、稳态 tokens/s、显存、快权重范数、恢复后的下一步 loss，以及长上下文任务的质量指标。32K/128K 结果必须注明真实数据、编译时间和是否发生 OOM；不能用 tiny 的形状检查代替长上下文质量结论。

## Decision：何时进入主线

只有当功能、成本和质量三层证据都能在固定协议下复现，TTT-E2E 才能作为主线架构结论；如果只通过形状检查或小规模 loss，结论应保持为“功能已验证，质量未验证”。

## Next step

先完成 dense、KV cache 和 TTT 的同预算对照，再扩展到 32K，最后才申请 128K 和 RULER 资源。每一步都把配置、数据摘要和报告提交到仓库。
