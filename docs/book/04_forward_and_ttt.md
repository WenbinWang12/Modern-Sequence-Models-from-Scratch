# 4. 前向顺序与快权重

## TTT-E2E 改变了什么

TTT-E2E 的核心不是简单替换一个 MLP，而是在部分后段层中额外加入一个可在测试时更新的 MLP′。原始 MLP 继续承担稳定的静态表示变换，MLP′ 通过内循环吸收当前上下文中的局部信息。两条路径必须在代码中明确分开，否则无法判断收益来自结构本身还是来自额外参数。

一个简化的 block 可以写成：

~~~text
x
 ├─ causal attention ──┐
 ├─ static MLP ─────────┼─ residual output
 └─ fast MLP′ ── inner update ─┘
~~~

训练时，模型在一个 block 内先用更新前的快权重产生预测，再根据标签计算内循环损失并更新快权重。下一块或下一个 chunk 是否使用新状态，由配置中的更新边界决定。

## 前向中的边界

代码需要同时表达三个边界：

1. **计算边界**：哪些层包含 MLP′，哪些层只保留标准 FFN。
2. **状态边界**：快权重、KV cache、未完成块和随机状态属于哪个 session。
3. **梯度边界**：内循环更新是否保留高阶梯度，静态参数如何接收外循环梯度。

如果只在张量形状上验证前向，可能仍然把 suffix 的标签泄漏给 prefix。因果性测试必须检查修改后半段 token 不会改变前半段已经产生的 logits。

## 代码入口

- ttt/models/ttt_e2e/modeling_ttt_e2e.py 组织完整模型和层级前向。
- ttt/models/ttt_e2e/modeling_ttt_e2e_train.py 提供训练路径和内循环接口。
- ttt/models/ttt_e2e/modeling_ttt_e2e_infer.py 提供推理路径，避免把训练图带入 decode。
- ttt/runtime.py 保存 session 的快权重、KV/Delta 状态、未完成 chunk 和随机状态。

阅读代码时建议从 __call__ 向下追踪，而不是先看 kernel。先确认输入、残差和状态的语义，再看 JAX scan 或 grouped GEMM 如何实现它们。

## 实验与证据

在有 GPU 的作业中运行：

~~~bash
bash scripts/validate_gpu.sh outputs/gpu-validation.json
~~~

报告应同时包含整段计算与流式计算的 logits 差异、因果性检查、状态形状和恢复结果。整段与流式只在相同的初始状态、位置和 mask 下比较；否则差异不能归因于 TTT。

## 下一章要回答的问题

前向路径说明“在哪里使用快权重”，但还没有说明“什么时候更新”。下一章把内循环和外循环写成明确的计算图，解释标签、SGD、AdamW 和梯度截断分别属于哪一层。
