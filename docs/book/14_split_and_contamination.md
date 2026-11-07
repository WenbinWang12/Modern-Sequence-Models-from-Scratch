# 14. 数据切分与污染检查

## 问题

随机切分并不总是可靠。相同文档、同一作者的连续文章或 benchmark 原文可能跨越 train、validation 和 test。

## 切分策略

根据任务选择：

- 文档级切分；
- source 级切分；
- 时间切分；
- domain 切分；
- 版本切分。

切分前先做去重，切分后再对 train/validation/test 做交叉 hash 检查。

## 污染报告

报告应说明：

- benchmark 样本是否出现在训练数据；
- benchmark 的近似匹配数量；
- 过滤阈值和匹配算法；
- 哪些样本被删除以及原因。

评测集污染不是一个可以由 loss 自动发现的问题，需要单独的数据审查流程。

## 练习

为一个小型 validation 集生成 n-gram hash，检查它和训练 manifest 是否存在重叠。
