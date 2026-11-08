# 6. 注意力、缓存与流式推理

## 同一个模型有两种时间轴

训练通常一次处理一个完整 batch，推理却经常一次接收一个 token。完整前向可以直接构造因果 mask；流式推理必须把过去的信息放进 KV cache、Delta 状态和 TTT 快权重中。两条路径的数学结果应尽量一致，但内存布局和更新时机不同。

~~~text
prefill:  x[0:L]  → hidden states + cache + fast weights
decode:   x[L]     → read cache → produce y[L] → append state
~~~

prefill 负责建立上下文，decode 负责逐 token 增量更新。若 decode 重新计算全部历史 token，结果可能正确，但无法代表真实的服务成本；若 cache 的位置或 mask 错一位，结果会静默偏离。

## 缓存的三类状态

| 状态 | 作用 | 生命周期 |
|---|---|---|
| KV cache | 保存过去 token 的 key/value | 一个请求或会话 |
| Delta/GDN 状态 | 维护递归的线性注意力状态 | 一个流式会话 |
| TTT 快权重 | 由内循环更新的局部参数 | 一个实验定义的 session |

这些状态都不是静态模型参数。它们应该有明确的 batch、层、head、位置和 dtype，并且能被快照、恢复和回滚。HTTP 服务中的请求取消不能留下半更新状态。

## 代码入口

- ttt/models/ttt_e2e/attention.py 实现因果注意力、GQA 和 KV cache。
- ttt/models/ttt_e2e/gated_delta.py 实现递归状态和门控更新。
- ttt/runtime.py 组合 prefill、decode、chunk 边界和随机状态。
- ttt/session_store.py 为会话快照提供原子保存、恢复和容量限制。
- ttt/serving.py、ttt/serve.py 把运行时状态暴露给服务入口。

## 运行时验证

GPU 作业中的验证命令为：

~~~bash
bash scripts/validate_gpu.sh outputs/gpu-validation.json
~~~

报告至少应比较：完整前向与 prefill+decode 的 logits、位置 0、chunk 边界、随机流式输入、快照恢复后继续生成，以及超出容量后的错误处理。比较时要固定初始 cache 和 session 状态；“看起来差不多”不能替代数值阈值。

tests/test_causality.py、tests/test_runtime_consistency.py 和 tests/test_session_store.py 分别覆盖因果性、整段/流式一致性和快照语义。正式实验还应保留运行时报告，而不是只保留测试通过的布尔值。

## 下一章要扩大哪一种状态

单个 dense block 的缓存边界明确后，MoE 会把状态扩展到专家路由、共享专家和跨设备 shard。下一章先固定路由语义，再讨论通信和吞吐。
