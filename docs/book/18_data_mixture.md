# 18. 数据混合与确定性采样

## 问题

多来源语料不能只依赖文件拼接顺序。不同来源的采样比例会改变模型看到的分布，也会影响最终结论。

## 混合配置

```json
{
  "sources": [
    {"name": "web", "manifest": "data/web/manifest.json", "weight": 0.5},
    {"name": "books", "manifest": "data/books/manifest.json", "weight": 0.3},
    {"name": "code", "manifest": "data/code/manifest.json", "weight": 0.2}
  ],
  "temperature": 1.0,
  "seed": 1234
}
```

需要明确：

- source 权重如何归一化；
- 是否使用 temperature sampling；
- 每个 global step 如何选择来源；
- resume 后采样序列是否保持一致；
- source 权重是否随训练阶段变化。

## 练习

用两个可区分的 toy source 运行固定 seed 的采样器，确认连续执行和从 checkpoint 恢复后的 source 序列一致。
