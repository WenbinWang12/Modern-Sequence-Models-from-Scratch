# 多领域预训练语料

仓库中的 `configs/data/` 保存语料来源配置、固定 revision 和构建后的统计报告。大规模原始文件与 token 分片不进入 Git，构建脚本把它们写到 `TTT_CORPUS_SCRATCH` 指定的临时目录，并把可审计的清单和报告保存到 `data/corpus_meta/`。

## 当前 Dolma3 版本

`configs/data/corpus_dolma3.json` 固定了 `allenai/dolma3_mix-150B-1025` 的 revision `afa92bfb22366821c5e6cd427cdd036b34b713ef`，使用 GPT-2 tokenizer，保留公开 mix 中有意重复的高质量文档。当前已完成的构建报告在 `configs/data/corpus_dolma3.report.json`。

按 GPT-2 token 统计，训练池包含约 **174.48B tokens** 和 **101.56M documents**。

| 来源 | tokens | 占比 |
|---|---:|---:|
| Common Crawl 网页 | 127.87B | 73.29% |
| olmOCR science PDFs | 21.38B | 12.26% |
| Stack-Edu 代码 | 19.34B | 11.08% |
| FineMath 3+ | 4.35B | 2.49% |
| arXiv / Proof-Pile | 1.48B | 0.85% |
| Wikipedia / Wikibooks | 0.063B | 0.036% |

这是一份多领域 base-pretraining 语料池。Dolma3 官方公开配方的结构与当前版本一致，网页、科学 PDF、代码、数学、arXiv 和百科都保留。当前代码比例高于官方样本表，差异来自实际文件统计和 tokenizer 口径。后续若要强化数学、代码或问答能力，应在 midtraining 阶段调整采样，而不是修改这份来源池的历史报告。

## 构建和验证

在已经下载评测数据的 Linux 机器上，从仓库根目录运行：

```bash
export TTT_CORPUS_SCRATCH=/dev/shm/$USER-ttt
export TTT_CORPUS_PROCESSES=64
bash scripts/build_corpus.sh dolma3
```

如果需要让 Dolma3 和另一份语料在训练集、held-out 和评测污染方面互相排除，可以把另一份已构建的名称作为第二个参数：

```bash
bash scripts/build_corpus.sh dolma3 v2
```

构建流程会固定 Hugging Face revision，记录原始文件 SHA256，按文档摘要分配 held-out，执行跨来源去重和 benchmark 去污染，再生成 token 分片。`verify.json` 记录复查结果，`report.json` 记录来源规模与过滤统计。大规模分片默认位于临时目录，机器重启后需要重新构建。

已有构建结果时，脚本会拒绝覆盖。确认需要重建后再设置：

```bash
TTT_CORPUS_FORCE=1 bash scripts/build_corpus.sh dolma3
```

## 仓库中应提交的内容

- 来源配置，例如 `configs/data/corpus_dolma3.json`
- 固定输入的统计报告，例如 `configs/data/corpus_dolma3.report.json`
- 构建、验证和配比实验脚本
- `ttt/corpus.py` 与 `ttt/dataset.py` 的数据读取和校验逻辑

原始 `.jsonl.zst`、token shard、临时下载目录和运行输出不提交到 Git。这样可以在不把百 GB 级文件放入仓库的情况下复现来源、校验结果和训练入口。
