# 3. Repositories and reproducible runs

## Overview: interrupt a tiny run and prove that it continues

“The checkpoint opens” is not a recovery experiment. We need to show that an uninterrupted run and an interrupted-then-resumed run read the same next input, use the same step and optimizer state, and produce the same next result within a declared tolerance. We use <code>tiny-contract-001-C</code> from the previous chapter.

After this chapter, you should be able to distinguish <code>--resume</code> from <code>--initialize-from</code>, explain why the random stream is part of experiment state, and decide from <code>status.json</code>, <code>metrics.jsonl</code>, and a checkpoint whether a failure is safely resumable.

## Prerequisites and comparison table

Suppose a tiny run saves at step 2. One path continues to step 4; the other resumes from step 2 and also continues to step 4:

| Check | Uninterrupted | Resumed | Passing condition |
|---|---:|---:|---|
| Starting point | step 0 | step 2 | the same checkpoint is loaded |
| Next batch | third draw from the initial RNG | first draw after restoring RNG state | tokens and masks agree elementwise |
| Optimizer step | 2→3 | 2→3 | it does not return to 0 |
| Valid-target count | 16 for this ordinary NPY sample | sum of the returned mask | agrees with the uninterrupted path; not inferred from nominal throughput |
| Loss | \(L_3\) | \(L'_3\) | \(\lvert L_3-L'_3\rvert\leq\epsilon\) |

Recovery is therefore a state comparison, not a file-existence check.

## A run is an object

A command proves that a program started. It does not explain why the result looks the way it does. A reproducible run has an identity made from source, configuration, data, environment, and state.

We write that identity as

$$
R=(S,C,I_{\mathrm{data}},E,K,Q_{\mathrm{budget}},O),
$$

where $S$ is the source commit, $C$ the resolved configuration, $I_{\mathrm{data}}$ the fingerprints of data, tokenizer, and manifests, $E$ the environment, $K$ random and optimizer state, $Q_{\mathrm{budget}}$ the token, sequence, batch, step, and cost budget, and $O$ the outputs and evidence. Reproduction means keeping these defining objects aligned while explaining unavoidable numerical differences.

~~~text
experiment id
├── source: git commit
├── config: model / optimizer / data
├── input: manifest and SHA256
├── environment: Python / JAX / device
├── state: parameters / optimizer / sampler
└── evidence: metrics / reports / status
~~~

Without one of these fields, reproduction becomes an approximation.

## Resume semantics

resume restores parameters, optimizer state, consumed-data position, and random state. It is for continuing an interrupted run and requires compatible configuration and data.

initialize-from loads parameters but creates a new optimizer, step, and data order in a new output directory. The current entry point requires the same parameter tree, leaf shapes, and dtypes. Structurally identical C and F models can reuse weights. A dense model without MLP-prime cannot initialize a model with an added MLP-prime through this flag alone: that requires explicit weight migration, which this entry point does not implement. Confusing initialization and resume changes the meaning of step, learning rate, and consumed tokens.

## A small failure case

Suppose a checkpoint was written at global step 100. Its parameters have seen 100 batches, and AdamW has accumulated first- and second-moment statistics for those updates. If a new directory copies only the parameters and writes step=100, the next command may run while silently starting a different experiment:

1. the optimizer moments are zero, so the update direction changes abruptly;
2. the schedule may restart at step 0, or use a step-100 learning rate with a fresh optimizer;
3. the BatchStream random state is missing, so the next batch is different;
4. a new manifest or sequence length may change the consumed-token count while the step number still looks continuous.

“The checkpoint opens” is therefore not a recovery test. Use the same short run in two paths: one uninterrupted path and one path that saves and resumes at the middle. Compare the next input order, step, valid-token count, and loss. Low-precision reductions may differ in the last digits, but the state fields and input order must not silently change.

Initializing F from a structurally identical C creates a new identity:

~~~text
C:      source=S1, data=D1, step=1000, optimizer=O_C
F init: source=S1, data=D1, step=0, optimizer=O_F
~~~

The two runs share an initial parameter file but are not one continuous curve. A report must not append the second loss curve to the first.

## Parameters and temporary fast state

A training checkpoint and a running session store different state. Fast-weight initialization belongs to learned model parameters. The weights adapted within one training sequence are temporary; the outer loop learns how to initialize and use them. During serving, adapted weights instead belong to that session, along with its cache and position, and may need snapshot or rollback. A session snapshot does not replace the optimizer and sampler state in a training checkpoint.

## Code and artifacts

Separate current automatic output from additional records the experimenter must preserve. After successful initialization, the trainer writes:

| File or field | Purpose |
|---|---|
| status.json | running, failed, complete, or stopped; a failed run includes error_type |
| experiment.json | resolved model, inner-loop, training, and tokenizer configuration |
| environment.json | Python, JAX, devices, driver, and scheduler information |
| metrics.jsonl | step, loss, grad_norm, seconds, nominal tokens_seen/tokens_per_second, and other statistics |
| step_*.npz, latest.json | tiny checkpoint files and the latest saved path |

The trainer does not create <code>error.json</code> or a separate <code>checkpoint/</code> directory. Save Git commit, data-file SHA256, full traceback, and interpretation separately; the environment chapter records the first two. Initialization, data loading, and restore checks can fail before <code>status.json</code> is written. Its absence does not prove that a job never started: retain terminal or scheduler logs. A production launcher should cover these early failures, but the current trainer does not cover them all.

- ttt/provenance.py records Python, JAX, device, driver, PBS, and source information.
- scripts/record_environment.sh writes the environment snapshot.
- ttt/checkpoint.py saves parameters, optimizer, and sampler state and checks identity on restore.
- docs/reproducibility.md contains the complete installation and recovery path.

Machine-readable records are the source of truth. The checkpoint's <code>data_state</code> holds the data fingerprint and RNG state; the metric token counts refer to nominal input positions. Reports should point to these specific records instead of claiming that <code>experiment.json</code> contains every identity field.

Failures should leave their output directory and error status. A failed compile or incompatible checkpoint is evidence about the boundary that still needs work.

## Which differences change the experiment?

For model quality, fix the effective-token budget and data identity. For system efficiency, also fix devices, precision, batch, sequence length, and compilation protocol. For streaming behavior, compare the same initial cache, position, and session state.

Vocabulary, architecture, or training-token changes are differences in the experiment's definition. GPU, driver, or compiler changes are environment differences; performance results need separate reporting. Small discrepancies from floating-point precision or reduction order are numerical differences and need a declared tolerance. Do not call all three “randomness.”

## Validate recovery from a specific checkpoint

A four-step tiny run can finish before a person manages to interrupt it. Instead, complete the uninterrupted path, then resume its saved step-2 checkpoint in another directory. The checkpoint contains the state at that point, so this checks numerical recovery semantics; it does not test power loss, damaged files, or scheduler-signal handling. Use two output directories that do not yet exist, on the target Linux CPU node:

~~~bash
set -e
test ! -e outputs/recovery-continuous
test ! -e outputs/recovery-resumed
bash scripts/prepare_sample.sh outputs/sample-tokens.npy
JAX_PLATFORMS=cpu bash scripts/run_train.sh \
  --config configs/book/tiny_control.json \
  --data outputs/sample-tokens.npy \
  --output outputs/recovery-continuous
JAX_PLATFORMS=cpu bash scripts/run_train.sh \
  --config configs/book/tiny_control.json \
  --data outputs/sample-tokens.npy \
  --output outputs/recovery-resumed \
  --resume outputs/recovery-continuous/step_00000002.npz
~~~

The uninterrupted log should contain steps 1–4; the resumed log should contain only steps 3–4. Do not change the configured total step count to manufacture an interruption: resume currently requires an identical resolved experiment configuration. Use <code>--auto-resume</code> to continue the latest state in one directory. Calling it on an already completed step-4 run does not train an additional step.

A pseudo-random generator is a deterministic machine: a seed selects its initial state, and each draw advances the state. Reapplying the seed starts again at the first batch; restoring the internal state continues with the third batch. The following compares two draws from the seed with the saved state, then compares step-3 loss and the final parameters and optimizer state:

~~~bash
JAX_PLATFORMS=cpu .venv/bin/python - <<'PY'
import json
from pathlib import Path
import jax
import numpy as np
from ttt.checkpoint import load_checkpoint
from ttt.data import BatchStream

continuous = Path("outputs/recovery-continuous")
resumed = Path("outputs/recovery-resumed")
middle = load_checkpoint(continuous / "step_00000002.npz", load_optimizer=True)
config = middle.experiment

def new_stream():
    return BatchStream(
        "outputs/sample-tokens.npy",
        sequence_length=config.training.sequence_length,
        batch_size=config.training.batch_size,
        vocab_size=config.model.vocab_size,
        seed=config.training.seed,
    )

reference, restored = new_stream(), new_stream()
for _ in range(int(middle.state.step)):
    reference.next_batch_with_masks()
restored.load_state_dict(middle.data_state)
expected = reference.next_batch_with_masks()
observed = restored.next_batch_with_masks()
for left, right in zip(expected, observed):
    np.testing.assert_array_equal(left, right)

def metrics(path):
    rows = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    return {row["step"]: row for row in rows}

full_log, resumed_log = metrics(continuous), metrics(resumed)
assert sorted(full_log) == [1, 2, 3, 4]
assert sorted(resumed_log) == [3, 4]
np.testing.assert_allclose(full_log[3]["loss"], resumed_log[3]["loss"], rtol=1e-5, atol=1e-6)
left = load_checkpoint(continuous / "step_00000004.npz", load_optimizer=True)
right = load_checkpoint(resumed / "step_00000004.npz", load_optimizer=True)
assert jax.tree.structure(left.state) == jax.tree.structure(right.state)
maximum = 0.0
for a, b in zip(jax.tree.leaves(left.state), jax.tree.leaves(right.state)):
    a, b = np.asarray(a), np.asarray(b)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    maximum = max(maximum, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
assert left.data_state == right.data_state
report = {"passed": True, "step": int(right.state.step),
          "next_batch_equal": True, "valid_targets": float(observed[1].sum()),
          "loss_step3": resumed_log[3]["loss"], "max_state_abs_diff": maximum,
          "rtol": 1e-5, "atol": 1e-6}
Path("outputs/recovery-check.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
~~~

Only after all assertions succeed does this write <code>outputs/recovery-check.json</code>. Its <code>valid_targets</code> comes from the actual returned mask, not the nominal token counter. These are verification steps to execute in the target environment; no unmeasured loss values are presented as results. If data differs, inspect the fingerprint and RNG first. If data agrees but state differs, inspect optimizer, step, dtype, and numerical tolerance. Timing and throughput need not match because compilation and system load can differ.

## Summary

Resume continues one optimization process; initialize-from creates a new one with a compatible parameter tree. A useful recovery check compares the next batch, optimizer, step, and numerical result. The next chapter connects these recorded configuration values to concrete model shapes and vocabulary dimensions.
