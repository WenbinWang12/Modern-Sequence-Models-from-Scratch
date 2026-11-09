# 20　GPU、多卡与检查点：从能运行到可扩展

[English](en/09_gpu_validation.md)

## 三个“通过”不是同一个通过

一个 tiny 模型在 GPU 上得到 loss，说明 Python、JAX 和基本计算图能运行。它没有证明多卡 sharding 正确，也没有证明 125M 模型能放进目标显存，更没有证明长时间训练可以恢复。

因此把 GPU 验证拆成三级：

| 级别 | 回答的问题 | 典型证据 |
|---|---|---|
| 功能级 | 单设备计算是否正确 | finite loss、causality、forward/streaming |
| 系统级 | mesh、通信、shard、恢复是否正确 | global/local shape、all-to-all、restore |
| 规模级 | 目标配置是否可运行 | compile、稳态吞吐、显存、稳定性 |

任何一级失败，都不应被另一级的成功覆盖。

## 功能级：先验证语义

`scripts/validate_gpu.sh` 运行架构扩展、因果性和运行时一致性检查。它应比较整段和流式输出，并记录状态 shape、初始参数和误差阈值。

对 TTT，功能级还要检查：

- inner update 确实改变了快权重；
- 关闭 inner update 时 C 路径稳定；
- 修改未来 token 不改变过去 logits；
- 尾 chunk 的 padding 不改变 fast state；
- session snapshot 之后恢复同样的状态。

这些检查可以使用小 shape，但必须覆盖真实的 layer_types、cache 和 adaptation target。

## 系统级：JAX mesh 不是“把数组复制到多卡”

mesh 定义全局设备轴，sharding 规则决定每个参数和激活在哪个设备。data parallel 复制或切 batch，expert parallel 分布专家，二者组合后通信路径改变。

系统验证应输出：

- global shape 和 local shard shape；
- 每个 rank 的设备、进程和 coordinator；
- 参数、optimizer state 和 checkpoint 的 sharding；
- route token counts 和 all-to-all 时间；
- 失败 rank 和恢复后的下一步。

如果只在 rank 0 打印 loss，可能掩盖另一个 rank 的 shape 或通信错误。

## 规模级：性能要排除编译污染

JAX 第一次执行通常包含 tracing 和 XLA 编译。报告至少分开：

$$
T_{\mathrm{job}}=
T_{\mathrm{setup}}+
T_{\mathrm{compile}}+
T_{\mathrm{steady}}+
T_{\mathrm{eval}}+
T_{\mathrm{checkpoint}}.
$$

吞吐应使用稳定区间的有效 token 数除以 steady time。若把 checkpoint 和评估混进窗口，结果反映的是作业调度而不是模型 kernel。

还要记录显存峰值。预分配设置、编译缓存和异步执行会让“当前 allocated”与“peak bytes in use”不同；性能报告需要说明测量接口。

## 调度层与模型层分离

`scripts/sutd_*.pbs` 负责申请资源、设置进程和共享目录；`ttt/validate.py` 与 `ttt/validate_mesh.py` 负责模型验证。换到另一个集群时，调度参数会改变，验证语义不应改变。

多机启动需要 coordinator address、process count、process id 和共享路径。它们必须来自作业分配，而不是写死在 Python 模块中。每个 rank 都应知道自己的全局 step 和数据分片规则。

## 检查点恢复的边界

规模训练最常见的事故是“文件保存成功，但不能在下一台作业恢复”。恢复测试要覆盖：

1. 同设备、同 mesh 的正常 resume；
2. 进程重启后的 resume；
3. 同拓扑但不同 batch 的拒绝；
4. 数据 manifest 变化时的拒绝；
5. 目标设备布局变化时的显式转换或拒绝；
6. session snapshot 与训练 checkpoint 不混用。

恢复成功的定义是下一步状态和指标符合预期，而不是仅仅能够打开一个文件。

## 什么时候进入正式实验

只有功能级和系统级通过，才提交长时间的规模级作业。只有规模级报告有 compile、steady、memory、checkpoint 和失败记录，才把某个变体放入正式质量比较。

官方权重、长上下文和 RULER 还需要额外数据与硬件条件。前沿实验可以先用 tiny 验证方向，但结论必须标记为 hypothesis、reproduced、partially reproduced 或 unresolved。

下一部分回到 Frontier Lab：主线已经固定数据、状态和成本协议，热点研究才能在同一坐标系中比较。
