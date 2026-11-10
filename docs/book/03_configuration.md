# 14　配置：用一份架构合同约束所有张量

## 本章导读：让每个数字都有来源

在 <code>tiny-contract-001</code> 中，词表大小是 259、隐藏宽度是 16、attention 有 2 个 query head。读者应该能够从这三个数字一路算到 embedding、每个 head、残差流和 logits 的形状；如果某个数字只能在代码深处找到，配置就还不是实验合同。

读完本章，你应该能加载一份配置，列出解析后的默认值，手算一个 batch 的 shape，并解释为什么 tokenizer 的词表大小会固定分类头最后一维。本章不讨论训练是否有效，下一章才会把这些 shape 接到 loss。

## 预备知识和 tiny 目标

你只需要知道矩阵可以把一组数字映射成另一组数字。贯穿输入为 <code>input_ids</code>，形状是 <code>[B,L]=[2,4]</code>；本章的完成标准是写出它经过 embedding、attention/MLP 和 <code>lm_head</code> 后的每个形状。

## 为什么模型不应该“猜”参数

如果模型代码在 forward 里根据张量形状猜 head 数、词表或快权重层位置，实验配置就会变成隐藏状态。一个 JSON 看起来写了 hidden_size，实际模型却在别处覆盖它，最后得到的是无法审计的结构。

配置对象应该是架构合同：它写明所有会影响参数形状、计算图、状态和比较公平性的字段；加载时立即校验；保存时包含补全后的默认值。模型类只消费这份合同。

## 从词表到分类头

设：

- \(V\)：词表大小；
- \(d\)：hidden size；
- \(n_h\)：attention heads；
- \(d_{\mathrm{ff}}\)：普通 MLP 中间维度；
- \(d'_{\mathrm{ff}}\)：快 MLP 中间维度。

输入 embedding：

$$
E\in\mathbb R^{V\times d}.
$$

每个位置的隐藏状态位于 $\mathbb R^d$。SwiGLU 的中间计算可以使用 $d_{\mathrm{ff}}$，但残差输出必须回到 $d$。最后：

$$
\mathbf z_{b,t}=\mathbf h_{b,t}W_{\mathrm{out}},
\quad W_{\mathrm{out}}\in\mathbb R^{d\times V}.
$$

当前实现的 `lm_head` 使用 `bias=False`；如果以后显式启用输出偏置，可以在公式中再加 \(\mathbf b_{\mathrm{out}}\in\mathbb R^V\)。如果 tied embeddings，\(W_{\mathrm{out}}=E^\mathsf{T}\)；如果不共享，就额外分配一个 \(d\times V\) 矩阵。无论是否共享，logits 最后一维都是 \(V\)。这正是 tokenizer 选定后分类头输出维度基本固定的原因。

这也解释了为什么不能只把 MLP 中间层扩大来“增加词表能力”。中间层的宽度改变计算和表示容量，词表投影的输出空间仍由 tokenizer 决定。

## 四类字段和它们的边界

**模型尺寸字段**包括 <code>vocab_size</code>、<code>hidden_size</code>、<code>intermediate_size</code>、<code>num_hidden_layers</code>、<code>num_attention_heads</code> 和 <code>num_key_value_heads</code>。它们决定参数树的形状。

**序列字段**包括 max_position_embeddings、sliding_window、ttt_chunk_size 和 layer_types。它们决定 cache 容量、分块边界和合法位置。

**自适应字段**包括 use_fast_mlp、ttt_suffix_length、adaptation_target、inner.enabled 和 inner learning rate。它们决定是否有可更新状态，但不能偷偷改变 C/F 的静态架构。

**系统字段**包括 dtype、attention_backend、loss block size、parallelism 和 expert_parallel_size。它们影响运行方式，部分字段也会改变可比较的成本。

仓库的 TTTE2EConfig 将这些字段集中在 ttt/models/ttt_e2e/configuration_ttt_e2e.py；训练配置则由 ttt/config.py 和 ttt/training.py 管理。区分模型配置与训练配置，避免把“这层存在”与“这次更新它”混在一起。

## 默认值也属于实验输入

`ttt_suffix_length=None` 在启用 fast MLP 或 <code>adaptation_target="router"</code> 时取最后四分之一层，至少一层，其他情况下为 0；`num_key_value_heads=None` 默认等于 query heads；`head_dim=None` 根据 hidden size 除以 query head 数推导。默认值很方便，但如果报告只保存原始 JSON，读者不知道最终值。

因此 `to_dict()` 将补全后的字段写入实验记录。配置审计要问两个问题：原始输入是什么，解析后的实际值是什么。迁移到新版本时，默认规则改变也会改变模型；配置文件不变不代表实验不变。

## 校验为什么要早

配置类检查正整数、head 整除性、旋转位置编码的偶数维度、层类型数量、suffix 范围、dtype、特殊 token ID 和 attention 后端。

例如：

- hidden_size 不能被 num_attention_heads 整除，无法推导默认 head_dim；
- num_attention_heads 不能被 num_key_value_heads 整除，GQA 无法复制 KV；
- 启用 fast MLP 或 router 适应，且层类型包含 sliding_attention 时，ttt_chunk_size 大于 sliding_window；
- 在默认 <code>adaptation_target="mlp"</code> 下，use_fast_mlp=false 却指定非零 suffix；router 适应允许关闭 fast MLP 而保留 suffix，不能把两种情况混为一谈；
- vocab 中的 eos_token_id 越界，数据生成会产生模型无法索引的标签。

这些错误在 Python 初始化阶段失败，远比编译十分钟后才发现好。

## AutoConfig 与 AutoModel

`AutoConfig.from_pretrained` 读取配置，并兼容早期没有 model_type 的训练配置；`AutoModelForCausalLM` 根据 mode 选择 base、train 或 infer 类。这样的注册接口让训练脚本不需要知道每个模型的具体类名。

但自动接口也需要边界。它不能从一个 ttt_e2e checkpoint 自动推断你想用 training 还是 inference 语义；调用方必须显式选择。加载参数不等于加载 optimizer 或 session state，API 文档需要说清楚返回对象包含什么。

这里的自动注册只处理架构。<code>AutoConfig</code> 返回模型配置，不携带完整实验的 <code>inner</code> 和 <code>training</code>；<code>mode="train"</code> 只选择模型类，也不等于已经关闭或开启内更新。训练时仍需用 <code>load_experiment</code> 读取完整实验，并将其中的 inner 配置传给训练或序列损失路径。

## 一次配置审计

审计 tiny 配置时，依次核对：

~~~text
vocab_size = 259
hidden_size = 16
num_attention_heads = 2
num_key_value_heads = 1
num_hidden_layers = 2
ttt_suffix_length = 1
intermediate_size = 32
fast_intermediate_size = 32
inner.enabled = false    # tiny_control；tiny.json 的对应值为 true
sequence_length = 16
ttt_chunk_size = 4
~~~

上述是从配置摘出的审计表，不是可以直接载入的 JSON。模型字段属于 <code>model</code> 对象，<code>sequence_length</code> 属于 <code>training</code>，更新开关属于 <code>inner</code>。下面把这些数字连接到张量，而不仅是背诵字段名。

### 先用两维向量理解查表和投影

向量是一列有顺序的数；矩阵把多列这样的数排列起来。为了直接算出每一项，暂取四个 token、每个 token 两个数的 embedding：

$$
\mathbf E=\begin{pmatrix}1&0\\0&1\\2&1\\-1&1\end{pmatrix}.
$$

输入 ID 为 0 和 2 时，查表得到第 0 行 \((1,0)\) 和第 2 行 \((2,1)\)，组成形状为 \([2,2]\) 的数组。第一个维度数位置，第二个维度数每个位置的特征。若使用共享输出权重，隐藏向量 \(\mathbf h=(1,0)\) 与四行 embedding 分别做内积，得到

$$
\mathbf h\mathbf E^{\mathsf T}
=(1\times1+0\times0,\ 1\times0+0\times1,\ 1\times2+0\times1,\ 1\times(-1)+0\times1)
=(1,0,2,-1).
$$

两个特征产生四个候选 token 的分数。多个位置一起计算仍用同一个投影；再加入 batch 轴，就得到 \([B,L,d]\to[B,L,V]\)。这个例子仅演示查表和输出投影，省略了中间模型层。

### 用一个 tiny shape 走完整条路径

使用 tiny 的模型宽度 \(V=259\)、\(d=16\)、\(d_{\mathrm{ff}}=32\)，另取教学 batch \(B=2\)、位置数 \(L=4\)；这两个演算维度不改变实际训练配置的 \(B=1,L=16\)。输入是整数张量
\[
\texttt{input\_ids}\in\{0,\ldots,258\}^{2\times4}.
\]
查 embedding 后得到
\[
\texttt{hidden}\in\mathbb R^{2\times4\times16}.
\]
普通 SwiGLU 可以先把最后一维从 16 投影到 32，再通过 down projection 回到 16，因此残差相加前后的形状都保持 \([2,4,16]\)。最后的 tied <code>lm_head</code> 使用 \(E^\mathsf T\in\mathbb R^{16\times259}\)，所以 logits 是
\[
\texttt{logits}\in\mathbb R^{2\times4\times259}.
\]
第一个位置的预测不是一个长度任意的向量，而是 259 个候选 token 的分数；目标标签只是其中一个整数，例如 \(y=42\)。这一步把“词表大小决定分类头最后一维”变成了可手算的 shape 事实。

这些形状说明残差与分类头怎样连接；下面再核对注意力头和 MLP 内部宽度。

### 头维度和 MLP 中间维度

上面的批量布局用于理解多个样本如何叠放。当前模型 <code>forward</code> 实际处理一条序列：输入是 \([L]\)，embedding 后为 \([L,d]\)，输出为 \([L,V]\)。训练代码通过 <code>jax.vmap</code> 对 batch 中各序列应用同一个函数，才形成 \([B,L,V]\)；不能把教学表中的 \([B,L]\) 直接当成单序列接口输入。

对 tiny 的 \(d=16,n_h=2,n_{\mathrm{kv}}=1\)，默认每头宽度 \(d_h=8\)。Query 投影产生 \(2\times8=16\) 个数，reshape 后为 \([L,2,8]\)；Key 与 Value 各产生 \(1\times8=8\) 个数，形状为 \([L,1,8]\)。两个 query 头共享一个 KV 头，这就是本例的分组查询注意力（GQA）。它减少的是 KV 投影与存储，不把输出残差宽度减半；注意力原理在缓存章展开。

SwiGLU 中的“中间宽度 32”也不是把输出改成 32。对每个行向量 \(h\in\mathbb R^{16}\)，先作两个投影，再逐元素门控，最后投影回来：

$$
g=hW_g,\quad u=hW_u,\quad
\operatorname{SwiGLU}(h)=(\operatorname{SiLU}(g)\odot u)W_d,
$$

其中 \(W_g,W_u\in\mathbb R^{16\times32}\)，\(W_d\in\mathbb R^{32\times16}\)，\(\odot\) 表示逐元素乘法，\(\operatorname{SiLU}(a)=a/(1+e^{-a})\) 逐元素计算。两个中间向量都是 32 维，最终回到 16 维，才能与残差相加。本节先确定计算和形状；为什么选择这种门控属于模型设计消融的问题。

## 最小配置审计

在远端节点上，先确认配置可以被解析，再启动训练：

~~~bash
.venv/bin/python -c "from ttt.config import load_experiment; import json; print(json.dumps(load_experiment('configs/book/tiny_control.json').to_dict(), ensure_ascii=False, indent=2))"
~~~

预期输出中应同时出现 <code>model</code>、<code>inner</code>、<code>training</code> 和 <code>tokenizer</code>，并包含补全后的派生字段。若出现 head 不整除、特殊 token 越界或 sequence length 超过位置上限，先修配置，不要等待 XLA 编译后再查。这个命令只证明配置合同可以解析，不证明 forward 或训练已经正确。

## 本章小结

- 配置合同同时约束参数 shape、状态边界、成本和比较公平性；
- \(V\) 决定 embedding 和 logits 的词表轴，\(d\) 决定残差流宽度；
- 默认值必须写入解析后的实验记录；
- tiny shape walk 把抽象字段连接到真实张量；
- 下一章在同一份配置上计算 logits、交叉熵和梯度。
