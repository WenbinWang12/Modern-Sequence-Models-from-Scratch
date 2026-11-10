# 18　Attention、Cache 与流式推理

## 先区分训练时的矩阵和推理时的状态

训练时，模型通常一次看到长度为 \(L\) 的整段序列。它可以为所有位置构造 causal mask：位置 \(t\) 只能读取 \(1,\ldots,t\)，不能读取未来位置。生成时，模型往往已经处理过很长的前缀，现在只收到一个新 token。如果每次都重新计算全部历史，结果可以正确，但会反复计算同样的 key 和 value；KV cache 保存这些历史结果，让新 token 直接复用它们。

TTT 又增加了一层状态。过去的上下文不仅保存在 KV 或 Delta 矩阵中，还可能改变 fast weights。因此一次推理请求的状态是一个有边界的集合：

~~~text
session
├── position
├── KV cache or Delta state
├── TTT fast weights
├── incomplete chunk
└── random state
~~~

这些字段必须一起推进或一起回滚。只恢复 KV 而不恢复 fast weights，会得到另一个模型状态；只恢复 fast weights 而位置错位，后续 RoPE 和 mask 也会错。

## Attention 到底在算什么

先用两个 token 做一个最小例子。设每个 token 的 query、key、value 维度都是 \(2\)，位置 1 的 query 为
\(q_1=(1,0)\)，两个 key 为
\(k_1=(1,0)\)、\(k_2=(0,1)\)，两个 value 为
\(v_1=(10,0)\)、\(v_2=(0,10)\)。缩放点积为

$$
s_{t,j}=\frac{q_t k_j^\mathsf{T}}{\sqrt{d_h}}.
$$

在位置 1，因果规则只允许读取位置 1，所以分数向量只有 \((1/\sqrt{2})\)，softmax 后权重为 \(1\)，输出就是 \(v_1=(10,0)\)。

假设位置 2 的 query 也是 \(q_2=(1,0)\)。它可以读取两个 key，分数为

$$
\left(\frac{1}{\sqrt{2}},0\right)\approx(0.707,0).
$$

softmax 后约为 \((0.67,0.33)\)，所以输出约为

$$
0.67v_1+0.33v_2\approx(6.7,3.3).
$$

这个例子说明三件事：query 决定“我在找什么”，key 决定“每个历史位置和我有多匹配”，value 承载“真正要取回的内容”；causal mask 决定哪些 key 根本不能参与 softmax。实现中通常把它们堆成形状 \([B,H,L,d_h]\) 的张量，而不是逐个 Python 向量循环。

## KV cache 的形状与显存

对 GQA，query heads 可以多于 key/value heads。例如 \(H_q=4\)、\(H_{kv}=2\) 时，每个 KV head 服务两个 query head。cache 保存的是 KV heads 的历史表示：

| 对象 | 形状 | 含义 |
|---|---|---|
| query | \([B,H_q,L_{\mathrm{new}},d_h]\) | 当前要计算的位置 |
| key cache | \([B,H_{kv},L_{\mathrm{past}},d_h]\) | 已保存的历史匹配向量 |
| value cache | \([B,H_{kv},L_{\mathrm{past}},d_h]\) | 已保存的历史内容 |
| position | \([B,L_{\mathrm{past}}]\) 或计数器 | RoPE 与 mask 的位置依据 |

单层 KV cache 的近似字节数是

$$
\mathrm{bytes}
=2\cdot B\cdot H_{kv}\cdot L_{\mathrm{past}}\cdot d_h\cdot s,
$$

其中第一个 \(2\) 表示 key 和 value 两份，\(s\) 是每个元素的字节数。若 \(B=1\)、\(H_{kv}=2\)、\(L_{\mathrm{past}}=4096\)、\(d_h=64\)、dtype 为 bfloat16（\(s=2\)），单层约占

$$
2\cdot1\cdot2\cdot4096\cdot64\cdot2
=2{,}097{,}152\text{ bytes}\approx2\text{ MiB}.
$$

12 层都保存时约为 24 MiB，还没有计入激活、fast weights、分配碎片和其他状态。这个计算解释了为什么上下文长度、KV head 数和 dtype 会直接影响推理显存。

<code>attention.py</code> 的 cache 还要保存当前长度、容量和位置。full attention 通常保留完整历史；sliding attention 只保留窗口内历史；gated delta 不使用同样的 KV 数组，而保留递归状态。

## Prefill 和 decode 的生命周期

prefill 把一段 token 一次写入状态，并输出最后位置的 logits。decode 读取当前状态，处理一个或一小段新 token，然后追加状态。

| 阶段 | 输入 | 读取的状态 | 写入的状态 | 典型输出 |
|---|---|---|---|---|
| prefill | \(x_{1:L}\) | 空或旧 session | \(K,V\) 到长度 \(L\)，以及 chunk 边界后的 fast state | 位置 \(L\) 的 logits |
| decode | \(x_{L+1}\) | 长度 \(L\) 的 cache/state | 追加一个位置，必要时更新 fast state | 位置 \(L+1\) 的 logits |

TTT 的 chunk size 大于 1 时，decode 不能把每个 token 都当作完整 inner update。它可能需要把 token 放进 incomplete chunk，待标签和边界满足后再更新。<code>runtime.py</code> 保存未完成块，就是为了不把半个更新误当成完整状态。

一个长度为 6、chunk size 为 4 的例子是：

~~~text
prefill: x1 x2 x3 x4  → 用 φ0 处理并得到 φ1
decode:  x5 x6        → 先放入未完成块，不能假设已经完成第二次更新
~~~

如果服务在 \(x_6\) 后保存 snapshot，恢复时必须同时恢复位置 6、KV 长度、未完成块和当前 fast state；只保存 token 文本不能重建同一个内部状态。

## 整段和流式为什么可能不一致

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

<code>session_store.py</code> 的快照应该原子写入，带模型版本、session ID、位置、状态摘要和协议版本。恢复前先检查模型参数版本与更新策略；恢复后再继续生成。

训练 checkpoint 还包含 optimizer 和 sampler，因此不能直接作为 HTTP session snapshot。两种文件都可以使用 Orbax 或其他存储后端，但 schema 和生命周期不同。

## 代码映射与验证清单

阅读代码时可以按这条路径走：

1. 在 <code>attention.py</code> 找 Q/K/V 投影、causal mask、cache 读写和容量检查；
2. 在 <code>runtime.py</code> 找 session state、incomplete chunk 和 snapshot；
3. 在 <code>inner_loop.py</code> 找 fast state 如何随 chunk 更新；
4. 用验证脚本比较 dense/full-sequence 与 prefill/decode 的 logits。

在 GPU 作业中，至少验证：

1. 同一个前缀的 prefill 与 token-by-token decode；
2. chunk 边界前后的位置与状态；
3. 修改 suffix 不影响 prefix logits；
4. snapshot 后恢复生成；
5. 超过容量时明确报错；
6. 多 session 之间状态隔离。

下一章把“一个状态”扩展成“多个专家的状态路径”：MoE 的路由和专家并行会把通信也纳入模型定义。
