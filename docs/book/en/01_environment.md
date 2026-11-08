# 1. Environment and the first training run

## The engineering question

The first failure in a from-scratch project is often not a wrong equation. It is an environment, seed, data path, output directory, or resume rule that was never made part of the experiment.

This chapter builds the shortest trusted path: install the environment on a new machine, create traceable sample tokens, run a few steps, save a checkpoint, and continue from the same directory. Every later chapter reuses this path.

## The machine contract

The repository records three layers:

| Layer | Fixed information | Artifact |
|---|---|---|
| System | GPU, driver, CUDA, filesystem | environment.json |
| Python | Python, JAX, Optax, dependencies | pyproject.toml and lock file |
| Experiment | commit, config, data digest, seed | experiment.json |

Together they answer what the machine is, what the code requires, and what this run did. A terminal history is not enough; a training directory should explain itself.

## Code entry points

- scripts/bootstrap_linux.sh installs uv, Python 3.12, and locked dependencies.
- scripts/check_environment.py checks Python, the JAX backend, devices, and key packages.
- scripts/prepare_sample.sh creates token data from examples/sample.txt.
- scripts/run_train.sh is the generic CPU or single-device entry; scripts/run_gpu_train.sh adds GPU settings.
- ttt/train.py owns configuration parsing, steps, logging, and checkpoints.

The shell layer makes the project easy to start. Python owns experiment semantics so the same run can be placed behind PBS, Slurm, or a container.

## First run

Run this on the target Linux machine or inside a GPU allocation. For a published result, replace main with an exact commit.

~~~bash
git clone https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch.git
cd Modern-Sequence-Models-from-Scratch
git checkout main

TTT_INSTALL_CUDA=1 bash scripts/bootstrap_linux.sh
python3 scripts/check_environment.py --require-gpu --min-gpus 1
bash scripts/prepare_sample.sh outputs/sample-tokens.npy

bash scripts/run_gpu_train.sh \
  --config configs/runs/tiny_gpu.json \
  --data outputs/sample-tokens.npy \
  --output outputs/tiny-gpu
~~~

A run directory should contain experiment.json, environment.json, metrics.jsonl, status.json, latest.json, and checkpoints. The first step may include XLA compilation; measure steady throughput after compilation. CPU runs validate behavior, not GPU throughput.

## Evidence and transition

Completion means that the parsed vocabulary and sequence length are recorded, loss is finite, a checkpoint can be read, the output contains source and data digests, and auto-resume does not restart at zero. The remote CPU path is recorded in [the generic Linux run report](../../reports/remote-generic-run.md).

The next chapter turns “the code runs” into “the experiment has an identity” by fixing source, environment, data, state, and resume semantics.
