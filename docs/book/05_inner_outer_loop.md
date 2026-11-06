# 5. 内循环与外循环

## 问题

TTT 的实验核心是两层优化：内循环更新快权重，外循环更新静态模型参数。更新边界和梯度路径必须写清楚，否则训练结果无法解释。

## 代码

- `ttt/inner_loop.py`：块级预测、token loss、SGD、完整序列 scan；
- `ttt/training.py`：AdamW 外循环、梯度裁剪和训练状态；
- `ttt/loss.py`：分块词表交叉熵，避免一次分配完整 logits。

每个块使用更新前的快权重产生预测；块内的 next-token 标签全部观测后才更新，下一块使用新快权重。

## 实验

```bash
python3 -m pytest -q tests/test_meta_gradients.py tests/test_grouped_gradients.py
```

配置 `inner.enabled=false` 可以生成静态 C 对照；配置 `inner.enabled=true` 是 F 路径。两者应该保持相同模型架构。

## 证据

测试覆盖解析梯度、有限差分、二阶梯度、快权重更新范数和 C/F 配对配置。

## 下一步

注意力缓存和 Gated Delta 会决定长序列训练与流式推理的状态边界。
