# 10. 数据契约

## 问题

训练数据不是一个文件路径，而是一份需要被验证、追踪和复用的输入协议。数据契约先规定每条样本的结构和元数据，再决定清洗规则。

## 最小格式

```json
{"id":"doc-0001","text":"...","source":"dataset-a","license":"CC-BY"}
```

至少需要明确：

- `id` 是否稳定且唯一；
- `text` 是否为非空字符串；
- `source` 和 `license` 是否可追溯；
- 语言、时间、领域等字段是否参与过滤；
- 原始文件和处理后文件的 SHA256；
- 训练、验证、测试的切分规则。

## 实验产物

数据处理程序应该同时生成：

```text
cleaned.jsonl
manifest.json
cleaning_report.json
```

`manifest.json` 记录文件路径、样本数、token 数、SHA256 和处理版本。语料本身不提交到 Git 仓库。

## 练习

为 `examples/retrieval.jsonl` 增加稳定 ID、数据来源和许可字段，并写一个校验器拒绝重复 ID 和空文本。
