## 7 适用范围与发布说明

- **模型性质。** 四个主实验模型都是只做过预训练的小型英文基座语言模型，没有经过指令微调、对话微调或任何安全对齐训练。它们适合研究序列建模方法，不适合直接面向用户，也不应把它们的输出当作事实来源。
- **数据许可。** 训练只使用两种公开语料：FineWeb-Edu（ODC-By 1.0）和 PG-19（公有领域书籍文本，数据集卡片为 Apache-2.0）。仓库不再分发语料本身，只提供下载脚本、固定的 revision 和文件摘要。评测集按各自的许可只下载评测划分，不进入训练。
- **内容风险。** FineWeb-Edu 来自网页，PG-19 是 1919 年之前出版的书籍，其中有与当代规范不符的表述，模型会复现训练语料中的偏见和过时内容。本报告没有做毒性、偏见或越狱评测，这些是面向用户发布之前必须补上的工作，而不是这一版声称已经完成的事。
- **测试时更新特有的风险。** 快权重只存在于单个会话里，会话结束即丢弃，发布的慢参数不会被用户输入改写；实现上不同会话的快权重和缓存相互隔离。但在会话内部，模型的行为会随读到的内容漂移，这既是方法的目的，也意味着长会话里的行为比固定权重的模型更难事先界定。第 4.5 节给出了快权重在 32K token 内的漂移幅度。
- **本报告不声称的事。** 不声称达到同规模模型的最好水平（第 4.7 节的公开模型在多数任务上更强，它们的训练量大得多）；不声称结论在更大的模型、更长的训练或更长的窗口下必然成立；不声称与任何闭源模型可比。

## 8 复现

全部实验可以用仓库里的脚本重跑。每一步都把输入文件的 SHA256、配置和环境写进输出目录。

```bash
# 环境（Linux，CUDA 12，Python 3.12）
uv sync --frozen --no-install-project --python 3.12 --extra cuda --extra checkpoint
uv pip install --python .venv/bin/python tiktoken matplotlib pyahocorasick

# 1. 语料：下载、分词、去重、评测集
python scripts/download_public_corpora.py --dataset pg19 --output data/raw/pg19
python scripts/download_public_corpora.py --dataset fineweb-edu --output data/raw/fineweb-edu --shards 0 1 2
python scripts/download_public_corpora.py --dataset fineweb-edu --output data/raw/fineweb-edu-heldout --shards 13
for split in train validation test; do
  python scripts/tokenize_corpus.py --source text-dir --input data/raw/pg19/$split --output data/tokens/pg19-$split
done
python scripts/tokenize_corpus.py --source parquet --input data/raw/fineweb-edu/00[012]_00000.parquet \
  --output data/tokens/fineweb-edu-train
python scripts/audit_report_data.py duplicates --fineweb-train data/raw/fineweb-edu/00[012]_00000.parquet \
  --fineweb-heldout data/raw/fineweb-edu-heldout/013_00000.parquet --pg19 data/raw/pg19 \
  --output outputs/audits/duplicates.json
python scripts/tokenize_corpus.py --source parquet --input data/raw/fineweb-edu-heldout/013_00000.parquet \
  --max-documents 20000 --exclude-digests outputs/audits/duplicates.json \
  --output data/tokens/fineweb-edu-heldout-dedup
python scripts/download_benchmarks.py --output data/benchmarks

# 2. 训练与评测：两张卡各跑一条流水线，互不通信；中断后重新执行即可续跑
python scripts/make_report_configs.py
scripts/report_pipeline.sh 0 baseline   # 滑窗 → 全注意力 → 推理吞吐 → 消融队列
scripts/report_pipeline.sh 1 ttt        # TTT-E2E → 无更新对照 → 朴素更新基线 → 消融队列
scripts/run_reference_benchmarks.sh 0   # 公开参考模型 + lm-evaluation-harness 交叉核对
scripts/run_extended_benchmarks.sh 7 main-full no-adapt 100
scripts/run_extended_benchmarks.sh 7 main-ttt adapt 100
scripts/run_extended_benchmarks.sh 7 main-ttt no-adapt 100

这里的 `0` 和 `1` 只是作业分配后的 GPU 编号示例，不是固定的机器编号。本报告在 H200 机器上使用 GPU 7 运行 baseline、GPU 5 运行 TTT，两条流水线彼此独立且不做跨卡通信。

# 3. 审计、图表和报告
python scripts/audit_report_data.py sampler --recipe configs/data/fineweb_pg19_50_50.json \
  --config configs/report/main_ttt.json --output outputs/audits/sampler_main.json
python scripts/audit_report_data.py benchmarks --fineweb-train data/raw/fineweb-edu/00[012]_00000.parquet \
  --pg19-train data/raw/pg19/train --output outputs/audits/benchmark_overlap.json
python scripts/make_report_figures.py && python scripts/make_report_diagram.py
python scripts/render_report.py
```

| 项目 | 内容 |
|:--|:--|
| 代码 | 本仓库。滑窗注意力和 TTT-E2E 两个主模型训练时的 `ttt/` 源码是提交 `d54407d`（源码摘要 `e3553bf5…`），其余训练使用提交 `854484e`（摘要 `5610909c…`）。两者之间 `ttt/` 的差别只有一个默认关闭的一阶近似开关、续训时比较配置的方式和 import 顺序，数值计算相同。每个运行的源码摘要记录在 `results.json` 的 `provenance` 字段，算法见 `ttt/provenance.py`。 |
| 配置 | `configs/report/*.json`（由 `scripts/make_report_configs.py` 生成），数据配比 `configs/data/fineweb_pg19_50_50.json` |
| 原始结果 | `docs/assets/report/results.json`（本页全部数字的来源）；`docs/assets/report/raw/` 下是各项评测脚本的原始输出 |
| 软件 | Python 3.12.16，JAX 0.4.38，Optax 0.2.4，Orbax 0.10.3，NumPy 2.5.4；参考模型用 PyTorch 2.15 和 Transformers 5.23 加载；lm-evaluation-harness 0.4.13 |
| 硬件 | NVIDIA H200 NVL（143,771 MiB，驱动 580.196.07），AMD EPYC 9555，Ubuntu 22.04 |
| 随机性 | 初始化、数据顺序由配置中的 `seed` 决定；评测窗口的选取是确定的，不含随机数 |

模型权重体积较大，没有放进 Git 仓库；检查点保存在训练服务器上，按上面的命令可以从头重新得到。

## 9 版本记录

| 版本 | 日期 | 内容 |
|:--|:--|:--|
| 0.1–0.6 | 2026-11-16 至 11-17 | 流程验证阶段：1M 参数的 CPU 试跑和 15M 参数、字节级 tokenizer、WikiText-2 上的 H200 配对实验。这些实验确认了训练、恢复和评测链路可用，但模型在公开 benchmark 上处于随机水平，不能支持任何能力结论。记录保留在 [CPU Pilot](reports/cpu_pilot_1m.md)、[H200 15M Scaling](reports/h200_scaling.md) 和 [Benchmark 测评协议](reports/benchmark_protocol.md)。 |
| 1.0 | 2026-11-18 | 本报告：GPT-2 BPE、30 亿级 token 的两种公开语料、4K 上下文下四个约 1.3 亿参数模型的匹配实验，长上下文、逐字回忆、公开 benchmark、消融、随机种子、规模和成本评测，以及数据审计。 |
