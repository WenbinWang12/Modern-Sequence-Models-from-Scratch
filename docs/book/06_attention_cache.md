# 17　Attention、Cache 与流式推理

[English](en/06_attention_cache.md)

## 训练中的矩阵和推理中的状态

训练可以一次看到长度为 L 的整段序列，并构造完整的 causal mask。生成时，模型通常已经处理了前缀，只收到一个新 token。若每次重新计算全部历史，算法可能正确，但成本随历史增长；KV cache 让新 token 复用过去的 key/value。

TTT 又增加了一层状态：过去的上下文不仅保存在 KV 或 Delta 矩阵里，还可能改变 fast weights。因此推理状态不再是一个 cache 数组，而是：

~~~text
session
├── position
├── KV cache or Delta state
├── TTT fast weights
├── incomplete chunk
└── random state
~~~

这些字段必须一起推进或一起回滚。只恢复 KV 而不恢复 fast weights，会得到另一个模型状态；只恢复 fast weights 而位置错位，后续 RoPE 和 mask 也会错。

## KV cache 的形状和生命周期

对 GQA，query heads 可以多于 key/value heads。cache 保存的是 KV heads 的历史表示，不是每个 query head 都复制一份。配置中的 num_key_value_heads 和 num_attention_heads 共同决定形状。

`attention.py` 的 cache 包含 key、value、位置和容量信息。full attention 通常需要完整历史；sliding attention 只保留窗口内历史；gated_delta 不使用同样的 KV 数组，而保留递归状态。

cache 的 capacity 不是 max_position_embeddings 的同义词。它是一次 session 分配的存储空间；如果需要延长，`grow_cache` 要处理旧值复制、滑窗截断和新位置。服务应在容量不足时明确报错或回收，而不是静默覆盖历史。

## Prefill 和 decode

prefill 把一段 token 写入状态，并输出最后位置的 logits。decode 读取当前状态，处理一个或一小段新 token，然后追加状态。

~~~text
prefill(x[0:L])
  → cache at position L
  → fast state after chosen chunk boundary

decode(x[L])
  → read cache at L
  → predict x[L+1]
  → append key/value and possibly update fast state
~~~

TTT chunk size 大于 1 时，decode 不能把每个 token 都当作完整 inner update；它可能需要把 token 放进 incomplete chunk，待标签和边界满足后更新。`runtime.py` 保存未完成块，就是为了不把半个更新误当成完整状态。

## 整段与流式为什么可能不一致

常见原因包括：

- position 起点不同；
- decode 使用了错误的 cache index；
- sliding window 在边界处移位；
- prefix 的 KV 被重新计算成不同精度；
- TTT 更新在整段和 decode 中发生在不同 token；
- 采样使用了不同随机状态；
- session snapshot 漏掉一个字段。

因此验证不能只比较最终生成字符串。应比较每个位置的 logits 或 top-k、cache 长度、fast weight norm 和 position。误差阈值要按 dtype 和 kernel 说明。

## Gated Delta 的另一种状态

线性注意力或 gated delta 不保存完整的历史 token，而维护递归统计量。它的状态可能是矩阵、卷积缓冲和门控变量。这样的状态占用与上下文长度关系不同，但表达能力、数值稳定性和更新语义也不同。

这正是 Frontier Lab 里 TTT 与线性注意力需要对照的原因：二者都把历史压缩进状态，却不代表状态更新规则相同。比较时要固定状态尺寸、训练预算和实际 FLOPs。

## Session snapshot 与 checkpoint

`session_store.py` 的快照应该原子写入，带模型版本、session ID、位置、状态摘要和协议版本。恢复前先检查模型参数版本与更新策略；恢复后再继续生成。

训练 checkpoint 还包含 optimizer 和 sampler，因此不能直接作为 HTTP session snapshot。两种文件都可以使用 Orbax 或其他存储后端，但 schema 和生命周期不同。

## 验证清单

在 GPU 作业中，至少验证：

1. 同一个前缀的 prefill 与 token-by-token decode；
2. chunk 边界前后的位置与状态；
3. 修改 suffix 不影响 prefix logits；
4. snapshot 后恢复生成；
5. 超过容量时的错误；
6. 多 session 之间状态隔离。

下一章把“一个状态”扩展成“多个专家的状态路径”：MoE 的路由和专家并行会把通信也纳入模型定义。
