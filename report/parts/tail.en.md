## 7 Scope and release notes

- **What the models are.** The four main models are small English base language models that have only been pre-trained. They have had no instruction tuning, chat tuning or safety alignment. They are suitable for research on sequence modelling, not for direct use by end users, and their output should not be treated as a source of facts.
- **Data licences.** Training uses two public corpora: FineWeb-Edu (ODC-By 1.0) and PG-19 (public-domain book text; the dataset card is Apache-2.0). The repository does not redistribute the corpora; it ships download scripts, pinned revisions and file digests. Evaluation sets are downloaded as evaluation splits only, under their own licences, and never enter training.
- **Content risks.** FineWeb-Edu comes from web pages and PG-19 from books published before 1919, some of which use language at odds with present-day norms. The models reproduce biases and outdated content in their training data. This report contains no toxicity, bias or jailbreak evaluation. Those are required before any user-facing release and are not claimed here.
- **Risks specific to test-time updates.** Fast weights exist only inside one session and are discarded when it ends; user input cannot rewrite the released slow weights, and the implementation isolates the fast weights and caches of different sessions. Within a session, however, the model's behaviour drifts with what it reads. That is the purpose of the method, and it also means that behaviour in a long session is harder to bound in advance than for a fixed-weight model. Section 4.5 reports how far the fast weights drift over 32K tokens.
- **What this report does not claim.** It does not claim the best results among models of this size (the public models in Section 4.7 are stronger on most tasks, with far more training). It does not claim the conclusions must hold for larger models, longer training or longer windows. It claims no comparability with any closed model.

## 8 Reproduction

Every experiment can be rerun with the scripts in the repository. Each step writes the SHA256 of its input files, its configuration and its environment into the output directory.

```bash
# Environment (Linux, CUDA 12, Python 3.12)
uv sync --frozen --no-install-project --python 3.12 --extra cuda --extra checkpoint
uv pip install --python .venv/bin/python tiktoken matplotlib pyahocorasick

# 1. Corpora: download, tokenize, de-duplicate, evaluation sets
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

# 2. Training and evaluation: one pipeline per GPU, no communication; rerun to resume after an interruption
python scripts/make_report_configs.py
scripts/report_pipeline.sh 0 baseline   # sliding window → full attention → throughput → ablation queue
scripts/report_pipeline.sh 1 ttt        # TTT-E2E → no-update control → naive-update baseline → ablation queue
scripts/run_reference_benchmarks.sh 0   # public reference models + lm-evaluation-harness cross-check
scripts/run_extended_benchmarks.sh 7 main-full no-adapt 100
scripts/run_extended_benchmarks.sh 7 main-ttt adapt 100
scripts/run_extended_benchmarks.sh 7 main-ttt no-adapt 100

The `0` and `1` values are examples of device IDs assigned to a job, not fixed machine-wide IDs. On the H200 run for this report, GPU 7 ran the baseline pipeline and GPU 5 ran the TTT pipeline. The two pipelines were independent and exchanged no cross-device data.

# 3. Audits, figures and the report
python scripts/audit_report_data.py sampler --recipe configs/data/fineweb_pg19_50_50.json \
  --config configs/report/main_ttt.json --output outputs/audits/sampler_main.json
python scripts/audit_report_data.py benchmarks --fineweb-train data/raw/fineweb-edu/00[012]_00000.parquet \
  --pg19-train data/raw/pg19/train --output outputs/audits/benchmark_overlap.json
python scripts/make_report_figures.py && python scripts/make_report_diagram.py
python scripts/render_report.py
```

| Item | Content |
|:--|:--|
| Code | this repository. The `ttt/` source used to train the sliding-window and TTT-E2E main models is commit `d54407d` (source digest `e3553bf5…`); all other training used commit `854484e` (digest `5610909c…`). Between the two, `ttt/` differs only by a first-order switch that is off by default, the way configurations are compared on resume, and import order; the numerics are identical. The source digest of every run is recorded under `provenance` in `results.json`; the algorithm is in `ttt/provenance.py`. |
| Configurations | `configs/report/*.json` (generated by `scripts/make_report_configs.py`); data mixture `configs/data/fineweb_pg19_50_50.json` |
| Raw results | `docs/assets/report/results.json` (the source of every number on this page); raw outputs of the evaluation scripts under `docs/assets/report/raw/` |
| Software | Python 3.12.16, JAX 0.4.38, Optax 0.2.4, Orbax 0.10.3, NumPy 2.5.4; reference models loaded with PyTorch 2.15 and Transformers 5.23; lm-evaluation-harness 0.4.13 |
| Hardware | NVIDIA H200 NVL (143,771 MiB, driver 580.196.07), AMD EPYC 9555, Ubuntu 22.04 |
| Randomness | initialisation and data order are set by `seed` in the configuration; evaluation windows are chosen deterministically, with no random numbers |

Model weights are large and are not stored in the Git repository. Checkpoints remain on the training server; the commands above regenerate them from scratch.

## 9 Version history

| Version | Date | Content |
|:--|:--|:--|
| 0.1–0.6 | 2026-11-16 to 11-17 | Pipeline validation: a 1M-parameter CPU run and 15M-parameter, byte-level paired runs on WikiText-2 on H200. These confirmed that training, restore and evaluation work end to end, but the models were at chance on public benchmarks and support no claim about capability. The records are kept in [CPU Pilot](reports/cpu_pilot_1m.md), [H200 15M Scaling](reports/h200_scaling.md) and [Benchmark protocol](reports/benchmark_protocol_en.md). |
| 1.0 | 2026-11-18 | This report: GPT-2 BPE, two public corpora with billions of tokens, a matched experiment with four models of about 130M parameters at 4K context, evaluations of long context, verbatim recall, public benchmarks, ablations, seeds, model size and cost, and data audits. |
