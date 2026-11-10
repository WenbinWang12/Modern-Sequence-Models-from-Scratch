# 数据章的可执行材料

这里的五个小实验分别对应获取、清洗、去重、切分和分词。它们解释各章的机制，不是大规模生产数据处理系统，也不提供模型质量或无污染保证。输入均为仓库内的原创教学文本。报告中的手算预期与实际运行产物应分别保存。

All five walkthroughs use the Python standard library and the original teaching inputs in this repository. They illustrate mechanisms, not production-scale preprocessing or model quality. Runtime logs are evidence only for the commands actually executed.

## 环境与执行

在实验用 Linux CPU 机器的仓库根目录运行；项目维护约定要求测试在指定远端 CPU 节点执行，不在开发电脑运行。

已有 Python 3.10 及以上时，标准库实验无需安装机器学习框架：

~~~bash
python3 examples/book/run_checks.py --output outputs/book-check-v1
~~~

若使用项目的完整锁定环境：

~~~bash
TTT_INSTALL_CUDA=0 bash scripts/bootstrap_linux.sh
.tools/uv sync --frozen --python 3.12.16 \
  --extra dev --extra checkpoint --extra serve --extra tokenizer
.venv/bin/python examples/book/run_checks.py \
  --output outputs/book-check-full-v1 --tokenizer --lint
~~~

The first command needs only Python 3.10 or newer. The full command additionally checks the production tokenizer and runs Ruff. The CPU bootstrap and frozen sync use the repository's lockfile. Execute commands on the experiment machine, from the repository root.

运行器拒绝覆盖已有目录，默认执行两个边界检查文件和五个章节实验，共七条子命令。每条命令的标准输出、标准错误和实际退出码写入独立文件；`validation.json` 保存执行计划、逐条状态、Python 版本及来源摘要。当前 `data_checks.py` 包含 10 个标准库测试方法，`dedup_checks.py` 包含 7 个；子测试还会检查多个输入变体，不能把“七条命令”当成“七个测试”。具体执行数量以本次日志为准。

任何命令失败、测试报告不符合要求或首尾来源摘要不一致，都会使 `all_requested_commands_passed` 保持 `false`。终态分别为 `completed`、`failed` 或 `interrupted`；只剩 `running` 的报告不是完成证据。摘要覆盖教学脚本、输入、tokenizer 实现与测试、`pyproject.toml` 和 `uv.lock`，并保存运行结束时的再次读取结果。它检查这些文件在首尾是否一致，不是对整个操作系统或 Python 安装做快照。应在没有并发修改的仓库副本中执行。

`--tokenizer` 先检查依赖，再运行测试文件并保存 `tokenizer-tests.junit.xml`。只有 JUnit 中测试数大于零，且失败、错误、跳过数都为零，才能通过这一项；缺失或损坏的 XML 同样失败。即使 pytest 实际退出码为零，也不会把跳过当成通过：报告保留真实退出码，另写 `validation_error`。此测试命令不加载模型的 `conftest.py`，并忽略外部 `PYTEST_ADDOPTS`，避免意外只收集或筛掉测试。当前文件包含 6 个 pytest 用例，其中 BPE 和 Unigram 分别占一个参数化用例。未请求此选项时，成功状态不包含这些 tokenizer 检查。

The default run contains seven subprocess commands: two boundary-check files and five walkthroughs. The current standard-library suites contain ten and seven test methods respectively; subtests exercise additional input variants. Command count is not test count. The runner preserves its plan, command states, actual return codes, stdout, stderr, and interpreter information.

Any command failure, invalid test report, or mismatch between initial and final source digests leaves `all_requested_commands_passed` false. Terminal states are `completed`, `failed`, and `interrupted`; a report still marked `running` is not completion evidence. Digests cover the teaching sources and inputs, tokenizer implementation and tests, `pyproject.toml`, and `uv.lock`. They compare file contents at the start and end, not an entire operating-system or Python-installation snapshot. Run in a checkout without concurrent modifications.

`--tokenizer` checks dependencies and saves `tokenizer-tests.junit.xml`. Its JUnit counts must show at least one test and zero failures, errors, and skips. Missing or malformed XML also fails the run. A pytest exit code of zero does not turn skipped tests into a pass: the actual code is retained and a separate `validation_error` explains rejection. The command avoids the model's `conftest.py` and ignores external `PYTEST_ADDOPTS` so collection-only or filtering options do not silently narrow it. The current file has six pytest cases, including separate BPE and Unigram parameterizations. Without this option, success does not include those checks.

完整 GPU、训练恢复、模型质量、语料无污染和视觉排版均不属于此入口的验收范围。成功运行也只支持实际执行的检查，不证明这些教学样例已经组成端到端生产流水线。

GPU behavior, training recovery, model quality, contamination-free corpora, and visual layout are outside this runner's scope. A successful run supports only the checks actually executed, not a completed end-to-end production pipeline.

## 分别运行

| 章节 / Chapter | 命令（使用同一个 Python 环境） / Command |
|---|---|
| 获取 / Acquisition | `python3 examples/book/acquisition_walkthrough.py --input examples/sample.jsonl --output outputs/acquisition-v1` |
| 清洗 / Cleaning | `python3 examples/book/cleaning_walkthrough.py --output outputs/cleaning-v1` |
| 去重 / Deduplication | `python3 examples/book/dedup_walkthrough.py --output outputs/dedup-v1` |
| 切分 / Splitting | `python3 examples/book/split_walkthrough.py --output outputs/split-v1 --seed 17` |
| 分词 / Tokenization | `python3 examples/book/bpe_walkthrough.py --output outputs/bpe-v1.json` |

分词的 `--output` 是 JSON **文件**；前四个实验的 `--output` 是新**目录**。`data_checks.py` 检查文件身份、Unicode、清洗归因和切分泄漏；`dedup_checks.py` 检查碰撞、候选、代表关系与小全集上的 MinHash 恒等式。`tests/test_tokenizer_training.py` 另行检查实际 BPE / Unigram 训练器的未见字节覆盖与规范化往返。

For BPE, `--output` names a JSON **file**. For the other walkthroughs it names a new **directory**. Boundary checks exercise actual files and deliberate failures, rather than accepting process success alone.

这些实验使用不同的微型语料，是为了让每个机制都能手算，不应直接把五份报告拼成一条已经完成的预训练语料流水线。正式端到端准备需要让前一步的实际输出成为下一步输入，并统一文档身份、分组规则、tokenizer 和版本；书稿后续的语料验收与训练章节负责连接这些部分。

The walkthroughs intentionally use different tiny corpora so each mechanism remains inspectable. They are not evidence of one completed end-to-end pretraining corpus. Such a pipeline must connect real stage outputs and maintain consistent identities, grouping, tokenizer, and versions.
