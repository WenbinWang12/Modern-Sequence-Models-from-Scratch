# 19. 训练前数据验证

## 问题

数据错误应该在训练启动前暴露，而不是训练数小时后才通过 NaN 或 loss 异常暴露。

## 启动前检查

```text
schema
→ manifest
→ SHA256
→ vocab range
→ mask shape
→ split overlap
→ tokenizer/config match
→ sequence length
→ batch and parallelism divisibility
```

训练命令应在这些检查失败时直接退出，并输出具体文件、样本或字段，而不是继续运行。

## 结果文件

建议生成：

```text
data-preflight.json
data-statistics.json
data-manifest.json
```

这些文件和训练配置一起进入 run directory，成为 checkpoint 的数据上下文。

## 练习

人为修改一个 shard 的 token、一个 manifest 的 SHA256 和一个 mask 的形状，确认三类错误分别被报告。
