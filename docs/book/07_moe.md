# 18　MoE：路由把模型变成一条动态计算图

[English](en/07_moe.md)

## 为什么“复制 FFN”不是 MoE

如果把一个 FFN 复制八份，然后每个 token 都执行八份，参数量增加了，激活计算也增加了，这只是更大的 dense 模型。MoE 的关键是 router 根据 hidden state 选择少数专家，使总参数容量和每 token 的激活计算分离。

一个 token 的动态路径是：

~~~text
h
→ router logits
→ top-k expert ids and gate weights
→ token dispatch
→ expert FFN
→ restore original order
→ weighted sum + residual
~~~

任何一步出错都可能保留相同 shape，却改变模型。比如 token dispatch 后没有正确恢复原顺序，loss 仍然会是一个有限数字，只是每个位置接收了别人的专家输出。

## Router 的数学对象

设有 E 个专家，router 为位置 t 产生 `r_t∈R^E`。经过 softmax 或 top-k 选择后，得到专家集合 `S_t` 和权重 `g_{t,e}`。稀疏专家输出可以写成：

$$
y_t=\sum_{e\in S_t}g_{t,e} f_e(h_t).
$$

top-k 约束了每个 token 的激活数，但不保证每个专家负载均匀。一个专家收到太多 token，可能超过容量而丢弃 token、填充无效位置或等待通信；一个专家几乎没有 token，则参数没有被充分训练。

仓库的 moe.py 还包含 shared experts。shared expert 对所有 token 提供稳定路径，稀疏专家提供容量和分工。shared expert 不是免费组件：它增加每 token 计算和参数，并改变路由收益的解释。

## 路由负载和辅助损失

常用路由辅助目标鼓励 token 使用更均衡的专家。仓库把 auxiliary loss、z loss 和 loads 作为统计返回到 sequence_loss，再在 training.py 中加入总 loss。

这几个量不能被简单合并成“MoE loss”。语言模型 nll 衡量 token 预测，aux loss 衡量路由行为，z loss 约束 router logits 的规模。报告中应分开记录，否则一个总 loss 下降可能只是辅助项变化。

如果使用 load bias 动态修正路由，bias 更新也应记录。它不是普通 AdamW 参数更新，而是根据统计量的额外规则。不同的 bias update rate 会改变训练算法。

## 容量、token drop 与公平性

一个 expert capacity 通常与每步 token 数、top-k 和 capacity factor 有关。允许 token drop 可以限制内存，却会让部分 token 没有经过完整专家路径；不允许 drop 可能导致 padding、通信或 OOM。

比较 dense 与 MoE 时，至少报告：

- 总参数和 active 参数；
- 每个 token 的 expert 数；
- expert token counts 的均值、最大值和分位数；
- drop 或 padding token 数；
- all-to-all 时间；
- 每个设备的峰值显存。

只比较总参数会低估 MoE 的活跃成本；只比较 active 参数又可能忽略 checkpoint 和 optimizer 的存储。

## Grouped GEMM 的角色

专家计算通常希望把属于同一专家的 token 聚合成一组矩阵乘法。`grouped_gemm.py` 提供参考路径和高阶实现路径，但 JAX、批处理和高阶导可能限制可用 kernel。

因此需要区分：

**数学路径**：token 按专家聚合，计算 expert FFN，再按原顺序还原。

**参考实现**：便于验证，可能逐专家执行矩阵乘。

**优化实现**：减少 kernel launch 和 padding，可能对 shape 有更严格要求。

参考路径正确，不代表优化路径等价；性能报告也不能拿参考路径的时间声称优化 kernel 的吞吐。

## 专家并行和 mesh

`ttt/distributed.py` 使用 data/expert 轴表达 batch 与专家的设备布局。expert parallel size 改变专家参数和 token 的所在设备，all-to-all 通信成为每步计算的一部分。

多卡验证需要检查全局和本地 shape。一个 rank 看到的 expert 参数可能只是 shard；直接用本地参数量和全局模型比较，会得出错误结论。checkpoint 恢复也必须恢复到目标 mesh，而不是先拼成单卡数组再假设等价。

## 最小两卡实验

~~~bash
bash scripts/run_gpu_train.sh \
  --config configs/runs/mesh_tiny.json \
  --data outputs/sample-tokens.npy \
  --output outputs/mesh-tiny

bash scripts/validate_mesh.sh \
  outputs/mesh-validation.json \
  outputs/mesh-checkpoint
~~~

这不是性能结论，而是系统门槛：专家 shape、路由计数、通信、两步 loss 和 restore 后的下一步都要有记录。先用不丢 token 的小配置验证路径，再探索容量和 kernel 优化。

下一章建立消融矩阵，把 TTT、Loop、MoE 的质量与计算成本放进同一张实验表。
