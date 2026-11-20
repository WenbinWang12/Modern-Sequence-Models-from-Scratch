# First eight-GPU screening run

This entry point runs one configuration at a time with all eight visible H100s. It uses single-node JAX data-parallel execution. `training.batch_size=64` is the global batch, so each card receives eight sequences when eight GPUs are visible.

The generator writes one JSON file per cell and a `manifest.json`. The filename, output directory, and `metadata.run_name` use the same explicit factor naming scheme. A name such as

```text
f8a__data-dolma3__mixer-mha__adapt-no-update__w-064__c-064__p-125m__tpp-024__tok-3b__ctx-4k__b-64__opt-adamw__seed-000
```

can be filtered in W&B without opening the config. The metadata also stores the factor dictionary and the W&B project/entity placeholders.

## Server commands

```bash
git clone https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch.git
cd Modern-Sequence-Models-from-Scratch
bash scripts/bootstrap_linux.sh
source .venv/bin/activate
```

Build or point to the verified Dolma3 recipe. The default is `configs/data/sweep/dolma3.json`.

```bash
TTT_CORPUS_SCRATCH=/dev/shm/$USER-ttt bash scripts/build_corpus.sh dolma3
```

Run the 100-step distributed smoke test first.

```bash
TTT_DATA=$PWD/configs/data/sweep/dolma3.json \
  bash scripts/run_first8_h100.sh smoke
```

Run the nine Wave A cells sequentially, using all eight cards for every cell. Wave A contains the 4×2 window/update grid plus one plain sliding-window anchor with the fast branch removed.

```bash
TTT_DATA=$PWD/configs/data/sweep/dolma3.json \
  bash scripts/run_first8_h100.sh wave_a
```

Wave B is selected with `wave_b`. `all` runs smoke, Wave A, and Wave B in that order. Existing checkpoints are resumed with `--auto-resume`. Override `TTT_TOKENS`, `TTT_BATCH_SIZE`, `TTT_SEQUENCE_LENGTH`, `TTT_OUTPUT_ROOT`, or `TTT_PYTHON` through the environment.

The first full cells use 3B tokens, a 4K sequence, global batch 64, and AdamW. This is about 11,445 optimizer steps and a TPP around 20–24 for the 125M-class models. `WANDB_API_KEY` is blank by default. Local `metrics.jsonl`, `run_metadata.json`, `environment.json`, and checkpoints are written before any W&B callback is enabled.
