# 14　配置：用一份架构合同约束所有张量

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

**模型尺寸字段**包括 vocab_size、hidden_size、intermediate_size、layers、attention heads 和 KV heads。它们决定初始化树的 shape。

**序列字段**包括 max_position_embeddings、sliding_window、ttt_chunk_size 和 layer_types。它们决定 cache 容量、分块边界和合法位置。

**自适应字段**包括 use_fast_mlp、ttt_suffix_length、adaptation_target、inner.enabled 和 inner learning rate。它们决定是否有可更新状态，但不能偷偷改变 C/F 的静态架构。

**系统字段**包括 dtype、attention_backend、loss block size、parallelism 和 expert_parallel_size。它们影响运行方式，部分字段也会改变可比较的成本。

仓库的 TTTE2EConfig 将这些字段集中在 ttt/models/ttt_e2e/configuration_ttt_e2e.py；训练配置则由 ttt/config.py 和 ttt/training.py 管理。区分模型配置与训练配置，避免把“这层存在”与“这次更新它”混在一起。

## 默认值也属于实验输入

`ttt_suffix_length=None` 会根据层数和 use_fast_mlp 推导后缀长度；`num_key_value_heads=None` 默认等于 query heads；`head_dim=None` 根据 hidden size 推导。默认值很方便，但如果报告只保存原始 JSON，读者不知道最终值。

因此 `to_dict()` 将补全后的字段写入实验记录。配置审计要问两个问题：原始输入是什么，解析后的实际值是什么。迁移到新版本时，默认规则改变也会改变模型；配置文件不变不代表实验不变。

## 校验为什么要早

配置类检查正整数、head 整除性、旋转位置编码的偶数维度、层类型数量、suffix 范围、dtype、特殊 token ID 和 attention 后端。

例如：

- hidden_size 不能被 num_attention_heads 整除，无法推导默认 head_dim；
- num_attention_heads 不能被 num_key_value_heads 整除，GQA 无法复制 KV；
- ttt_chunk_size 大于 sliding_window，后缀层无法覆盖更新块；
- use_fast_mlp=false 却指定非零 suffix，C 路径和配置语义矛盾；
- vocab 中的 eos_token_id 越界，数据生成会产生模型无法索引的标签。

这些错误在 Python 初始化阶段失败，远比编译十分钟后才发现好。

## AutoConfig 与 AutoModel

`AutoConfig.from_pretrained` 读取配置，并兼容早期没有 model_type 的训练配置；`AutoModelForCausalLM` 根据 mode 选择 base、train 或 infer 类。这样的注册接口让训练脚本不需要知道每个模型的具体类名。

但自动接口也需要边界。它不能从一个 ttt_e2e checkpoint 自动推断你想用 training 还是 inference 语义；调用方必须显式选择。加载参数不等于加载 optimizer 或 session state，API 文档需要说清楚返回对象包含什么。

## 一次配置审计

审计 tiny 配置时，依次核对：

~~~text
vocab_size = 259
hidden_size = 16
heads = 2, kv_heads = 1
layers = 2
ttt_suffix_length = 1
fast_intermediate_size = intermediate_size
inner.enabled = true or false
sequence_length = 16
ttt_chunk_size = 4
~~~

再画出 embedding、每层 attention、MLP/MLP′ 和 lm_head 的 shape。只要其中一个数字没有来源，就说明配置还没有成为真正的实验合同。

### 用一个 tiny shape 走完整条路径

取 \(V=259\)、\(d=16\)、\(d_{\mathrm{ff}}=32\)、batch \(B=2\)、序列长度 \(L=4\)。输入是整数张量
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

下一章沿着这张 shape 图追踪一个 token，理解 Attention、MLP′、普通 FFN 和残差的先后顺序。
