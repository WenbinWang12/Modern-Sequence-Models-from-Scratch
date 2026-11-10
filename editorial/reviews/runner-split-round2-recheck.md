# 统一检查入口与切分输入边界的最终静态复核

日期：2026-11-11。复核者：`dedup_teaching_revision`。根代理授权直接修改统一运行器、材料 README 和切分数值边界；根代理随后独立逐段阅读了修改后的运行器，报告未见新的静态阻断。**执行和视觉验收均待完成，用户已允许推迟无法连接的远端验证；本轮不再尝试 SSH，也没有本地运行 Python／测试。**

## 检查范围与实际数量

完整阅读：

- `examples/book/run_checks.py` 和 `examples/book/README.md`；
- `examples/book/data_checks.py`、`dedup_checks.py`；
- `tests/test_tokenizer_training.py`，以及它实际使用的 `scripts/train_tokenizer.py`；
- `tests/conftest.py` 和 pytest／Ruff 配置；
- `examples/book/split_walkthrough.py` 及第14章两版相关输入、输出、边界说明。

当前默认入口是 **7 条子进程命令**：两个边界检查文件，加获取、清洗、去重、切分、BPE五个教学实验。当前标准库边界文件分别有10和7个 unittest 测试方法，另含多个 subTest 输入变体。此前说“8个标准库检查”是新增边界前的数量，不能继续当作当前计数。

`--tokenizer` 增加依赖预检和一个 pytest 子命令。测试文件有5个测试函数，其中一个按BPE/Unigram参数化，合计6个用例：三个教学BPE用例、两个正式训练器未见字节／NFKC往返用例、一个训练输入变化拒绝用例。`--lint` 再加一个 Ruff 子命令。README已区分命令数、测试方法数与参数化/子测试数，不把这些数字写成实测执行总数。

## 发现与修改

| 项目 | 原问题 | 本次修复 |
|---|---|---|
| pytest部分跳过 | 依赖预检能拦常见缺包，但退出码0仍可能伴随skip；原报告只检查returncode | 加 `--junitxml`，保存XML，读取 tests/failures/errors/skipped，要求测试数大于0且其余为0；缺文件、缺属性、无suite、负计数、testcase数量不一致等都拒绝 |
| 退出码与验收原因 | 单一退出码不能区分“pytest成功退出但报告不合要求” | 保留真正的 `returncode`，另写 `validation_error`／`validation_errors`；运行器自身退出非零，`all_requested_commands_passed=false` |
| 无完成状态 | 中断或运行器异常时只有一个false值，未记录已计划和当前命令 | 保存 `planned_commands`、逐条running状态和时间；记录completed/failed/interrupted终态，无法完整结束的running报告不当作通过 |
| 依赖范围 | tokenizer单文件测试会加载 `tests/conftest.py`，导入JAX/模型，超出当前单文件测试来源范围 | 使用 `--noconftest`；当前测试仅依赖pytest内置tmp_path和monkeypatch，不用该模型fixture |
| 外部pytest筛选 | `PYTEST_ADDOPTS=-k ...` 或 `--collect-only` 可悄悄改变检查内容 | 对tokenizer测试子进程清除此变量，并在报告记录策略；显式使用 `pyproject.toml` |
| 来源摘要 | 原来仅在开始时读摘要，运行期间修改来源仍可能给出通过 | 结束时重新扫描来源和输入，记录首尾摘要及changed_sources；不一致或终态读取失败均不通过；增加pyproject.toml和uv.lock |
| split非有限元数据 | JSON默认接受NaN/Infinity；合法JSON数字1e999也可变为inf；保留额外metadata后重序列化会写出非标准Infinity | 增加非标准常量拒绝与有限浮点parse回调，所有文档/报告输出加allow_nan=False；双语14说明数字范围限制及整次拒绝、不创建目录 |
| 文档过宽声明 | 命令成功容易被理解为生产、GPU、模型、污染或视觉保证 | README明确排除这些范围，区别独立教学语料与真实端到端流水线 |

pytest选项和JUnit结构查阅了作者的[输出文档](https://docs.pytest.org/en/stable/how-to/output.html#creating-junitxml-format-files)与[命令行参考](https://docs.pytest.org/en/stable/reference/reference.html)。这只是API核对，未冒充实际命令执行。

来源首尾相等只证明两次读取所得字节一致，不排除中途改动后又恢复；它不等于文件系统快照、环境锁定或对所有第三方代码的摘要。README要求在无并发修改的checkout执行。依赖实际版本由tokenizer预检stdout保存，完整环境命令继续使用锁文件。

## 与另一代理的协调

`bpe_teaching_revision` 负责新增 `data_checks.py` 的切分回归方法，复核者没有并行编辑该文件。已静态读到的新增用例：

- 直接metadata中的 `NaN`、`Infinity`、`-Infinity`、`1e999`、`-1e999`；
- 嵌套scores/value中的正负溢出；
- 每种无效输入要求整体抛出ValueError，且输出目录不存在；
- 普通0.125／-2.5嵌套元数据在三份实际分片JSONL中保持一致。

用例沿用12条切分fixture加元数据，因此保留“至少一个多文档组、至少两个非空集合”的故障探针条件，不因错误样本过小而测错分支。既有清洗非有限边界和两版说明保持不变。

## 本次文件及静态证据

修改：

- `examples/book/run_checks.py`
- `examples/book/README.md`
- `examples/book/split_walkthrough.py`
- `docs/book/14_split_and_contamination.md`
- `docs/book/en/14_split_and_contamination.md`
- 本复核记录，以及去重整改记录中的状态链接

由另一代理修改：`examples/book/data_checks.py`。

本次静态读取时摘要：

| 文件 | SHA-256 |
|---|---|
| run_checks.py | `E340B62765EFCE72F1F5A48851D48629E6F51994E1AEF32F0132966FF36EF984` |
| examples/book/README.md | `E3BA9CC50FE4ED411FBBA9B68744801F47D4463F8FE05E65FA44ED22C2A725EC` |
| split_walkthrough.py | `93114D34DC8F438EC56A599C115FE575C046212D2E6C11D56B4BF5E7B7DB23FE` |

`git diff --check` 对本次修改无空白错误。根代理已另行阅读运行器的JUnit解析、状态与终态摘要逻辑。它们都是静态证据，没有本地执行，也没有远端通过结果。

## 延后验证

用户已明确允许连接不上时暂不测试。后续在指定远端节点恢复后执行README中的标准库和完整命令，保留新目录、日志、JUnit XML和validation.json。还应检查运行器的失败路径：子命令失败、pytest零退出但包含skip、空/损坏JUnit、来源改变、人工中断均不得给出completed。实际网页渲染与矢量图验收另行进行。

此记录完成后作者冻结编辑，交根代理统一提交；不宣布出版通过。
