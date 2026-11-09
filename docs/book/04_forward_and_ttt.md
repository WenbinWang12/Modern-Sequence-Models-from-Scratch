# 16　前向顺序：一个 token 如何穿过 TTT-E2E

## 先不要从 kernel 读模型

TTT-E2E 看起来包含很多新名词：prefix、suffix、fast weights、inner loss、MLP′、KV cache。最稳妥的读法不是先看 JAX scan 或 grouped GEMM，而是拿一个 token，沿着残差流走一遍。

模型首先把 token ID 查成 embedding。随后每一层将隐藏状态送进 attention 或 gated delta，再送入 MLP。后段层可能多一条 fast MLP′。最后的 hidden state 经过归一化和词表投影得到 logits。

~~~text
token ids
   ↓
embedding
   ↓
attention / gated delta + residual
   ↓
fast MLP′ + residual     （选择的后段层）
   ↓
static MLP or MoE + residual
   ↓
final norm → lm_head → logits
~~~

每个箭头都是形状、mask、cache 和梯度的接口。将 MLP′ 放在普通 MLP 前后，得到的算法并不相同；将更新发生在 token 内、chunk 末尾或下一个 chunk，状态语义也不同。

## 当前代码的前向顺序

`TTTE2EForCausalLM._block` 先执行 attention branch。若层类型是 full_attention 或 sliding_attention，则调用 attention；若是 gated_delta，则调用递归状态更新。输出加回 hidden，形成 attention residual。

若参数中包含 fast_mlp，代码取当前 fast weight 初始化的 MLP′，对 hidden 做 norm、SwiGLU 和可选 post norm，再加回 residual。之后才执行普通 MLP 或 MoE 分支。

这意味着当前实现的概念顺序是：

~~~text
Attention / Delta
→ Fast MLP′
→ Static MLP or MoE
~~~

这与“用 MLP′ 替换原 FFN”不同。普通 FFN 仍然存在，MLP′ 是附加的适应路径。若 fast weights 不存在，配置和参数树也不会凭空生成一条快路径。

## Prefix 与 suffix 的作用

配置将层分成 prefix 和 suffix。prefix 一次性运行静态层，产生每个位置的 hidden；suffix 使用这些 hidden 和可更新快权重继续计算。

这种划分减少了内循环需要重复的计算，但也带来一个约束：prefix 的 cache 和 hidden 必须与 suffix 的位置、mask 对齐。后缀不是一段独立文本，它接收的是同一序列中对应位置的表示。

`sequence_loss` 先把 token 序列按 ttt_chunk_size 补齐并 reshape，再运行 model.prefix。suffix 在 scan 中逐 chunk 处理。快权重从 initial_fast_weights 开始，下一块使用上一块的更新结果；尾块不足完整长度时，代码保留旧 fast weights，不让 padding token 改变状态。

## 为什么快权重不是普通参数

普通参数由外循环 AdamW 更新，跨 batch、跨训练 step 保存。快权重在一个序列内由 inner loss 更新，可以在下一个 chunk 变化；是否跨 request 保留，则由 runtime session 决定。

可以用两个时间索引区分：

$$
\theta_{k+1}=\mathrm{AdamW}(\theta_k,\nabla_\theta \mathcal{L}_{\mathrm{outer}}),
$$

$$
\phi_{t+1}=\phi_t-\eta_{\mathrm{inner}}
\nabla_\phi \mathcal{L}_{\mathrm{inner}}(x_t;\theta,\phi_t).
$$

$\theta$ 是静态参数，$\phi$ 是快权重。外循环学习的是“怎样初始化和使用 $\phi$”，内循环执行的是“当前上下文如何改变 $\phi$”。如果把 $\phi$ 直接放进 optimizer state，TTT 的时间尺度就被错误合并了。

## 梯度边界和因果性

一个安全的内循环顺序是：

~~~text
使用 fast_t 预测 y_t
→ 用已观测 label 计算 inner loss
→ 更新得到 fast_{t+1}
→ 后续 chunk 使用 fast_{t+1}
~~~

如果 suffix 的 label 在同一次预测前已经用于更新，模型可以通过标签泄漏获得不应有的信息。因果性测试应修改后半段 token，再比较前半段 logits；前半段不能因为未来标签改变。

当前实现的 inner_step 对 fast_weights 求 value_and_grad，再调用 sgd_update。外循环对 sequence_loss 的输出求导，因此是否保留高阶梯度由这条计算图和 checkpoint 重算共同决定。阅读代码时要区分“更新了 fast state”和“是否对更新过程求了高阶导”。

## 一个手算的 chunk 例子

设 sequence length 为 8，chunk size 为 4。输入位置 0–7 被分成两块：

~~~text
chunk 0: positions 0,1,2,3
chunk 1: positions 4,5,6,7
~~~

chunk 0 使用 $\phi_0$ 产生 $\mathcal{L}_0$，并更新成 $\phi_1$。chunk 1 使用 $\phi_1$ 产生 $\mathcal{L}_1$。若最后一个块只有 2 个真实位置，另外 2 个是 padding，mask 将后两个 label 置零，状态更新也不应因 padding 发生。

这个例子解释了为什么 chunk size 不是单纯的 batch 参数。它改变了快权重观察上下文的频率，也改变了内循环反向传播和 cache 的边界。

## 如何验证前向不是“形状正确但语义错误”

至少做四种检查：

1. dense C 路径与 fast MLP 关闭时的输出形状和数值；
2. 修改 suffix token 不影响 prefix logits 的因果性；
3. 完整序列前向与 prefill/decode 的输出在容忍范围内一致；
4. 把 chunk size 改变时，明确记录这是算法变量还是仅实现变量。

GPU 验证脚本会生成架构和运行时报告，但报告中的数值必须关联到 config 与初始参数。一次通过不说明所有长度和所有 layer_types 都通过。

下一章把前向中的更新写成内循环与外循环，解释损失如何在两个时间尺度之间流动。
