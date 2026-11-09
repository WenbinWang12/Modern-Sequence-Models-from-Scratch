# 19　消融、成本与 Scaling：怎样知道收益来自哪里

## 先写不变量

一个实验名称不能替代实验协议。假设我们比较 dense、TTT、Loop 和 MoE。如果 TTT 使用了更多训练 token，Loop 使用了更大 batch，MoE 使用了不同 tokenizer，那么最终 loss 的差异无法归因于结构。

每次实验先列出不变项：

~~~text
data manifest
tokenizer and vocabulary
effective training tokens
sequence length and mask
initialization and seed
optimizer and schedule
hardware and precision
evaluation code
~~~

再列出唯一改变项，例如 use_fast_mlp、inner.enabled、loop_mode、num_experts 或 layer_types。若研究问题本来就是多个变量的组合，则把组合写成明确的因子，而不是事后解释。

## 质量与成本是二维结果

模型质量常用 train loss、held-out loss 和任务指标衡量。系统成本至少包括参数、active 参数、FLOPs、编译、稳态吞吐、峰值显存、通信、checkpoint 大小和推理延迟。

可以用一个结果行表达：

~~~json
{
  "variant": "F2",
  "tokens": 1000000000,
  "parameters": 125000000,
  "active_parameters": 125000000,
  "held_out_loss": null,
  "compile_seconds": null,
  "steady_tokens_per_second": null,
  "peak_memory_bytes": null,
  "status": "pending"
}
~~~

`null` 不等于零，表示该实验还没有对应证据。没有实际 profiler 或硬件运行时，不应把理论估计填进实测字段。

## 参数量、FLOPs 和 tokens 的关系

对 dense Transformer，训练计算量常用参数与 token 的乘积估算量级。TTT 还包含 inner gradient、fast weight update 和可能的 post loss；MoE 还包含路由、dispatch、padding 和 all-to-all。因此统一用一个乘数会隐藏真正差异。

更好的做法是把成本拆成：

- 静态模型 forward；
- 内循环 forward 与 backward；
- 路由和专家矩阵乘；
- cache/session 状态；
- 通信与 checkpoint；
- 编译与稳态执行。

`ttt/profile.py` 可以报告参数量、fast MLP 参数量、prefill/decode 时间、状态字节和 XLA cost analysis。XLA 的 cost analysis 是编译器估计，不能代替目标硬件上的稳态计时。

## 消融矩阵怎样分层

**功能消融**在 tiny 上回答结构是否能运行：关闭 fast MLP、关闭 inner update、换 attention backend、改变 chunk size。

**质量消融**在固定数据和 token budget 上回答 held-out loss 是否变化：MLP′ 层位置、内循环步数、Loop 位置、MoE top-k。

**系统消融**回答效率：不同 batch、sequence length、mesh、cache、dtype 和 kernel。

三层不要混在一张未经说明的表里。功能通过不代表质量有效，质量更好也不代表系统更快。

## N/D scaling 的问题

Scaling 不是把一条 loss 曲线延长，而是要在多个模型规模 N 和数据量 D 上拟合趋势。一个简单形式可以写成：

$$
L(N,D)=L_\infty+A N^{-\alpha}+B D^{-\beta}.
$$

拟合需要独立样本、统一评估和足够范围；少量 pending 行不能被当成观测。`ttt/scaling.py` 读取 JSONL 并可做 bootstrap，但输出仍然依赖输入数据质量。

如果 TTT 的额外参数和 inner FLOPs 没有计入 N 或 D 的定义，scale 曲线会把架构差异折叠进变量。报告要说明 N 是总参数还是 active 参数，D 是原始 token 还是有效监督 token。

## 失败实验也放进矩阵

OOM、编译超时、通信失败或 loss NaN 不应被从表里删除。它们定义了方法的可用边界。记录失败的配置、设备、阶段和原因，后续才知道某个模型是在质量差，还是根本无法完成目标规模。

主线实验报告可以用 success、pending、failed、blocked 四种状态；状态转换需要写原因和时间。

## 一个可执行的最小矩阵

~~~bash
python3 -m ttt.experiments \
  --output configs/ablations/generated \
  --sizes 125m 760m \
  --include-moe

python3 -m ttt.profile \
  --checkpoint CHECKPOINT \
  --context-length 1024 \
  --decode-tokens 16 \
  --training-flops
~~~

先生成配置和成本清单，再提交训练。配置生成器不会自动保证公平；公平性仍由 manifest 和实验表检查。

下一章把实验矩阵放到 GPU 现实中，区分功能、分布式恢复和目标规模三个验证门槛。
