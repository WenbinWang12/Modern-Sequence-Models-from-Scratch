# 2. Environment: the first trusted training run

## Why the environment belongs to the experiment

The first failure in a from-scratch project is often not a wrong equation. It is a Python version, a missing CUDA wheel, a different random seed, a data path, or a resume rule that was never written down. If the run cannot say which machine, code revision, and input data it used, its loss curve cannot be reproduced.

Think of the environment as the laboratory bench. The model is the instrument, the dataset is the material, and the training command is the protocol. Replacing the bench can change the result even when the model code is unchanged. This chapter builds the shortest trusted path: install the dependencies, inspect the device, create a tiny token file, run a few steps, save a checkpoint, and record evidence.

## The three layers of a machine contract

We record three layers because each one answers a different debugging question:

| Layer | Beginner question | Typical evidence |
|---|---|---|
| System | Can this machine see the requested device? | GPU model, driver, CUDA, filesystem, host |
| Python | Can the same code import the same libraries? | Python, JAX, Optax, checkpoint package versions |
| Experiment | What exactly did this run train? | Git commit, expanded config, tokenizer, data digest, seed |

The layers are related but cannot replace one another. A correct Python installation does not guarantee enough GPU memory. A visible GPU does not guarantee that a model configuration is valid. A valid configuration does not tell us which bytes were in the dataset. A training directory should therefore explain itself rather than relying on terminal history.

## What the repository provides

The shell scripts are thin entry points; Python owns the experiment semantics so the same run can be submitted through PBS, Slurm, or a container.

- `scripts/bootstrap_linux.sh` installs the pinned Python and dependencies. Setting `TTT_INSTALL_CUDA=1` selects the CUDA extra.
- `scripts/check_environment.py` reports Python, JAX backend, visible devices, and key package imports. `--require-gpu` turns device absence into a failure.
- `scripts/prepare_sample.sh` converts `examples/sample.txt` into a small one-dimensional token file for path and shape checks.
- `scripts/run_train.sh` is the generic single-device entry point. `scripts/run_gpu_train.sh` adds GPU-oriented defaults.
- `ttt/train.py` parses the configuration, runs steps, writes metrics, and manages checkpoint output.
- `scripts/record_environment.sh` records runtime information separately when a scheduler job needs an early environment snapshot.

The sample file is a wiring test, not a language benchmark. It can reveal an out-of-range token, an impossible sequence length, a broken mask, or a checkpoint path. It cannot show that the cleaning policy is good or that the model has learned useful language.

## A minimal first run

Run the following on the target Linux machine or inside a GPU allocation. For a published result, replace `main` with an exact commit hash.

~~~bash
git clone https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch.git
cd Modern-Sequence-Models-from-Scratch
git checkout main

TTT_INSTALL_CUDA=1 bash scripts/bootstrap_linux.sh
python3 scripts/check_environment.py --require-gpu --min-gpus 1 \
  --output outputs/environment-check.json
bash scripts/prepare_sample.sh outputs/sample-tokens.npy

bash scripts/run_gpu_train.sh \
  --config configs/runs/tiny_gpu.json \
  --data outputs/sample-tokens.npy \
  --output outputs/tiny-gpu
~~~

On a CPU-only validation node, use `TTT_INSTALL_CUDA=0` and omit `--require-gpu`. CPU runs validate data paths and function semantics; their throughput is not a GPU performance result.

## What happens during the first step

“The command started” does not mean that steady-state training has begun. A first step usually contains four stages:

1. parse the JSON and fill default values;
2. initialize parameters and derive random keys;
3. trace the JAX function and compile it with XLA;
4. execute the compiled program and write metrics.

If the first step takes 40 seconds and later steps take 2 seconds, averaging those five steps would report a misleading throughput. A report should separate `compile_seconds` from steady-state `tokens_per_second`. The same distinction matters when comparing a new kernel or a new sharding plan.

## The expected evidence directory

A successful tiny run should leave evidence similar to:

~~~text
experiment.json   # expanded configuration, commit, data identity, seed
environment.json  # Python, JAX backend, devices, source summary
metrics.jsonl     # per-step loss, gradients, speed, state statistics
status.json       # success, failure, or interruption
latest.json       # pointer to the newest checkpoint
step_*.npz        # tiny-run checkpoint files (format may evolve)
~~~

The exact checkpoint filename can change, but the meaning of the records must not. Completion means that the loss is finite, the checkpoint can be read, source and data digests are present, and automatic resume continues from the saved step instead of silently restarting at zero.

## Reading a failure without guessing

If `check_environment.py` cannot find a GPU, inspect the backend and device list before changing the model. If the device is visible but initialization runs out of memory, compare dtype, batch size, sequence length, preallocation settings, and model dimensions. If initialization succeeds but XLA compilation fails, preserve the config and compiler log and inspect shape and sharding assumptions.

If a resumed run diverges, compare the data fingerprint, sampler position, optimizer state, and global step before blaming floating-point noise. A machine contract narrows the search from “the model is random” to a concrete layer and artifact.

The next chapter turns this directory into a formal experiment identity: the environment answers **where** the run happened; the contract answers **what** happened.
