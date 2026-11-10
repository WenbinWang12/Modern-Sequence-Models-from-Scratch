# 书中使用的 tiny 配置 / Tiny configurations for the book

这三份配置均使用 byte tokenizer（259 个 ID）、两层、隐藏宽度 16、4 个训练 step。它们用于检查数据和状态路径，不能用于判断模型质量。

| 配置 / Configuration | MLP′ | 内循环 / Inner update | 用途 / Purpose |
|---|---|---|---|
| `tiny_dense.json` | 无 / absent | 关闭 / disabled | 最小普通语言模型 / Minimal ordinary language model |
| `tiny_control.json` | 有 / present | 关闭 / disabled | C：保留适应分支的对照 / Control with adaptation branch retained |
| `../tiny.json` | 有 / present | 开启 / enabled | F：执行适应 / Adaptive variant |

C 与 F 的输入配置只有 `inner.enabled` 不同。Dense 与 C 的参数树不同，不能混称为同一个基线，也不能使用要求参数树一致的 `--initialize-from` 在它们之间直接迁移。

All three configurations use the byte tokenizer (259 IDs), two layers, width 16, and four training steps. They check data and state paths; they do not establish model quality. C and F differ only in `inner.enabled`. Dense and C have different parameter trees, so the current `--initialize-from` cannot transfer weights directly between them.
