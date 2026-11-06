# 14. 标准 Causal LM 基线

## 问题

TTT 和 MoE 的收益必须相对于一个可靠的 dense baseline 判断。基线先固定数据、tokenizer、优化器和训练 tokens，再开始改变模型结构。

## 基线配置

```text
num_experts = 0
use_fast_mlp = false
inner.enabled = false
loop_mode = none
```

基线至少需要验证：

- next-token loss 能下降；
- 梯度有限；
- checkpoint 能恢复；
- batch 和 sequence length 变化不会破坏形状；
- 训练和推理的 causal 输出一致。

## 公平比较

TTT、Linear Attention、Loop 和 MoE 的对照应固定：参数量、训练 tokens、数据、tokenizer、优化器、随机种子和硬件。参数量相同不等于计算量相同，正式结果还要记录实际 FLOPs、编译时间、稳态吞吐和峰值显存。

## 下一步

基线稳定后，才进入 [前向顺序与快权重](04_forward_and_ttt.md) 和内循环章节。
