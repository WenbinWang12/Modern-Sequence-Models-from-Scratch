# 16. Token Packing 与 Mask

## 问题

文档 token 还不是训练 batch。需要决定文档边界、EOS、padding、跨文档 packing、position IDs 和监督 mask。

## 训练样本构造

```text
documents
  → append EOS
  → concatenate or keep boundaries
  → split into fixed sequence_length
  → build labels and loss_mask
  → batch
```

预训练通常监督所有有效 next-token；SFT 可以只监督 response 和 EOS。提示段是否参与 TTT 内循环，需要作为训练配方明确记录。

## 必须检查

- labels 是否是输入右移一位；
- padding 是否不会贡献 loss；
- 文档边界是否符合 attention mask；
- mask 是否和 labels 形状一致；
- 训练和验证的 packing 规则是否一致。

## 练习

手工构造两个短文档，分别比较独立样本和跨文档 packing 的 labels、mask 和 position IDs。
