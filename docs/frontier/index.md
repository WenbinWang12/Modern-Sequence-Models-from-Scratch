# Frontier Lab：前沿专题

这一部分不是按固定教材顺序推进的章节，而是一个带日期的研究雷达。每个专题都必须回答三个问题：

1. 论文或社区当前提出了什么新主张；
2. 这个主张与 TTT-E2E 当前代码有什么关系；
3. 我们能否在固定规模、固定数据和固定算力下复现或证伪它。

## 热点专题的生命周期

```text
Radar → Paper reading → Minimal reproduction → Compute-matched comparison
  ↑                                               ↓
Archive ← No evidence / topic cooled ← Decision record
```

每个专题都记录快照日期、论文版本、代码状态和复现实验状态。热点可以从主线进入，也可以在证据不足时归档，不把未经验证的宣传结论写成教材事实。

## 当前专题

| 专题 | 核心问题 | 当前状态 |
|---|---|---|
| TTT 与线性注意力 | TTT 的状态更新是否可以看成线性注意力、KV binding 或 delta rule | 复现中 |
| TTT-E2E 与长上下文 | 测试时学习能否以权重状态压缩上下文，并保持可扩展性 | 主线方向 |
| Looped / recurrent-depth Transformer | 重复使用同一层增加有效深度，是否比增加独立参数更划算 | 复现中 |
| TTT × Loop | 时间方向的快权重更新与深度方向的循环是否互补 | 待设计对照 |

## 专题固定模板

```text
Snapshot：日期、论文版本、代码版本
Claim：论文主张是什么
Relation：与当前 TTT-E2E 哪些组件相关
Controls：必须保持不变的模型、数据、tokens、FLOPs
Minimal reproduction：最小可运行实验
Decision：支持、部分支持、未复现或证伪
Next step：是否进入主线架构或归档
```

专题代码和配置放在 `frontier/<topic>/`，正式结果放在 `docs/reports/`。前沿章节引用主线代码，但不把临时研究分支直接混入稳定训练入口。
