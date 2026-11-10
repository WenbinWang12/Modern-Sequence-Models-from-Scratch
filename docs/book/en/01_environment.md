# 2. Environment: the first trusted training run

## Overview: prove that the machine can run

This chapter does not try to train a large model in one command. We first make a smaller claim for <code>tiny-contract-001</code>: the machine can describe itself, produce a short token file, start a tiny run, and leave an interpretable output directory. When a later loss curve is wrong, we can then separate an environment problem from a data or model problem.

After this chapter, you should be able to read the JAX device list, inspect the length and vocabulary range of sample tokens, and find <code>experiment.json</code>, <code>environment.json</code>, <code>metrics.jsonl</code>, and a checkpoint in a tiny run directory. CPU is for functional checks; GPU is required for throughput and scale claims.

## Prerequisites and running input

You only need a terminal, basic shell commands, and the idea that JSON files in an output directory are machine-written records. We use <code>examples/sample.txt</code>. It is deliberately not a pretraining corpus; it is a minimal path check.

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
- `scripts/check_environment.py` records Python, package versions, visible devices, and GPU count. `--require-gpu` turns device absence into a failure.
- `scripts/prepare_sample.sh` converts `examples/sample.txt` into a small one-dimensional token file for path and shape checks.
- `scripts/run_train.sh` is the generic single-device entry point. `scripts/run_gpu_train.sh` adds GPU-oriented defaults.
- `ttt/train.py` parses the configuration, runs steps, writes metrics, and manages checkpoint output.
- `scripts/record_environment.sh` records runtime information separately when a scheduler job needs an early environment snapshot.

The sample file is a wiring test, not a language benchmark. It can reveal an out-of-range token, an impossible sequence length, a broken mask, or a checkpoint path. It cannot show that the cleaning policy is good or that the model has learned useful language.

For the tiny configuration, the first sample-data check should answer three questions: are there at least <code>sequence_length + 1 = 17</code> tokens, are all IDs in <code>[0, 259)</code>, and does the shifted window contain valid loss positions? If one answer is no, repair the data entry point before changing the model.

## A minimal first run

Run the following on the target Linux machine or inside a GPU allocation. For a published result, replace `main` with an exact commit hash.

~~~bash
git clone https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch.git
cd Modern-Sequence-Models-from-Scratch
git checkout main

TTT_INSTALL_CUDA=1 bash scripts/bootstrap_linux.sh
.venv/bin/python scripts/check_environment.py --require-gpu --min-gpus 1 \
  --output outputs/environment-check.json
~~~

The JSON should contain <code>python</code>, <code>packages.jax</code>, <code>device_count</code>, <code>devices</code>, <code>gpu_count</code>, and <code>passed</code>. There is no separate <code>backend</code> or <code>dtype</code> field. Asking a CPU node to satisfy <code>--require-gpu</code> is an unmet environment condition. For a CPU functional check, install with <code>TTT_INSTALL_CUDA=0</code> and omit <code>--require-gpu</code>.

Use <code>.venv/bin/python</code> explicitly: bootstrap creates the environment in a child shell and does not activate the calling shell. A bare <code>python3</code> could inspect the system interpreter while the training scripts later use a different one.

## What happens during the first step

“The command started” does not mean that steady-state training has begun. A first step usually contains four stages:

1. parse the JSON and fill default values;
2. initialize parameters and derive random keys;
3. trace the JAX function and compile it with XLA;
4. execute the compiled program and write metrics.

If the first step takes 40 seconds and later steps take 2 seconds, averaging them gives a misleading steady-state throughput. These are illustrative times, not measurements from this project. The trainer records per-step <code>seconds</code>, including first-step overhead; it does not emit a separate compile-time field. The <code>compilation_seconds</code> field from <code>ttt.validate</code> times a validation function and cannot replace the compilation measurement of a particular training job.

## The expected evidence directory

A successful tiny run should leave evidence similar to:

~~~text
experiment.json   # resolved model, inner-loop, training, tokenizer configuration
environment.json  # Python, JAX backend, devices, source summary
metrics.jsonl     # per-step loss, gradients, speed, state statistics
status.json       # running, failed, complete, or stopped
latest.json       # pointer to the newest checkpoint
step_*.npz        # tiny-run checkpoint files (format may evolve)
~~~

Git commit and data-file SHA256 must be recorded separately; <code>experiment.json</code> does not add them. The <code>source_sha256</code> in <code>environment.json</code> is a package-source digest, not a Git commit. The minimal run below records the additional identities explicitly. Recovery is checked in the next chapter.

## Reading a failure without guessing

If `check_environment.py` cannot find a GPU, inspect the backend and device list before changing the model. If the device is visible but initialization runs out of memory, compare dtype, batch size, sequence length, preallocation settings, and model dimensions. If initialization succeeds but XLA compilation fails, preserve the config and compiler log and inspect shape and sharding assumptions.

If a resumed run diverges, compare the data fingerprint, sampler position, optimizer state, and global step before blaming floating-point noise. A machine contract narrows the search from “the model is random” to a concrete layer and artifact.

## CPU, GPU, JAX, and a device mesh

A CPU starts the program, prepares data, and schedules work. A GPU has many parallel units suited to tensor operations. JAX turns numerical Python functions into differentiable, compiled computations. A mesh gives devices named coordinates, so an array axis such as the batch can be distributed according to an explicit rule.

For example, a hidden array of shape \([B,L,d]=[2,4,8]\) can reside on one GPU. A two-device data mesh can place one batch row, of shape \([1,4,8]\), on each device while preserving the global shape \([2,4,8]\). The processes must still agree on parameters, data assignment, and gradient communication. The tiny single-device run below does not require a mesh.

## Minimal checkpoint and expected evidence

On a node with the dependencies installed, the shortest CPU path is:

~~~bash
bash scripts/prepare_sample.sh outputs/sample-tokens.npy
JAX_PLATFORMS=cpu bash scripts/run_train.sh \
  --config configs/book/tiny_control.json \
  --data outputs/sample-tokens.npy \
  --output outputs/tiny-contract-001-C
git rev-parse HEAD > outputs/tiny-contract-001-C/source-commit.txt
sha256sum outputs/sample-tokens.npy > outputs/tiny-contract-001-C/data-sha256.txt
~~~

Use an output directory that has not already trained a model. Repeat a full experiment in a new directory, or use resume as described in the next chapter. On a GPU node, complete the device check and replace the entire <code>JAX_PLATFORMS=cpu bash scripts/run_train.sh</code> prefix with <code>bash scripts/run_gpu_train.sh</code>. Expected artifacts include the files above plus <code>step_00000002.npz</code> and <code>step_00000004.npz</code>. The final status should be <code>complete</code> at step 4; verify these expectations against your own artifacts.

The logged <code>tokens_seen</code> and <code>tokens_per_second</code> count batch size × sequence length, including masked positions. Ordinary NPY sample data uses target masks of one; later sharded supervision requires a separate effective-target count. Do not report first-step time, which can include compilation, as steady-state throughput.

If a command fails, inspect the sample path, <code>configs/book/tiny_control.json</code>, and <code>devices</code> in the environment record. Initialization failures can occur before <code>status.json</code> exists, so preserve terminal or scheduler logs; the trainer does not emit <code>error.json</code>. This checkpoint establishes a runnable tiny path, not language ability. The next chapter uses the step-2 checkpoint to compare resumed inputs and parameters with uninterrupted training.

## Summary

- System, Python, and experiment layers need separate records.
- Sample tokens first validate vocabulary range, length, shift, and masks.
- Tracing, compilation, and steady execution are different phases.
- Expected output and failure diagnosis belong next to the command.
- The next chapter checks whether these records form one identity and support recovery.
