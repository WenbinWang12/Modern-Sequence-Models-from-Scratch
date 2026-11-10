# 1. The Experiment Contract: Making a Run Interpretable

## Overview: turn a comparison into a question

Suppose we train the same tiny model twice: once with the TTT inner loop disabled and once with it enabled. A lower loss in the second run does not yet show that TTT helped. The runs may have consumed different tokens, used different learning-rate schedules, or started from different checkpoints. This chapter turns the question “what changed?” into a contract that a machine can inspect.

After this chapter, you should be able to fill in the research variable, fixed variables, budget, and evidence files for a control/treatment pair. You should also be able to classify a run as runnable, learnable, or interpretable. Model construction and data processing are defined in later chapters; this chapter defines experiment identity.

## Running example: tiny-contract-001

We use <code>tiny-contract-001</code> throughout. It reads <code>configs/tiny.json</code> and byte-token shards made from <code>examples/sample.txt</code>, then runs 4 steps with batch size 1 and sequence length 16. The control disables the inner update; the treatment changes only <code>inner.enabled</code> from <code>false</code> to <code>true</code>.

| Field | Control | Treatment |
|---|---|---|
| Run identity | <code>tiny-contract-001-C</code> | <code>tiny-contract-001-F</code> |
| Only change | <code>inner.enabled=false</code> | <code>inner.enabled=true</code> |
| Fixed inputs | commit, tokenizer, data, 16-token window, 4 steps, seed, dtype | same |
| Main metrics | <code>loss</code>, <code>valid_tokens</code>, <code>tokens_per_second</code> | same |
| Evidence directory | <code>outputs/tiny-contract-001-C/</code> | <code>outputs/tiny-contract-001-F/</code> |

Write this table before looking at a result. It prevents the comparison rule from changing after the numbers are known.

## Decide what the experiment is meant to prove

This book does not treat “the code ran” as a research conclusion. The question is whether, under the same data, training budget, and system conditions, a new sequence module changes how a model stores, updates, or uses context, and whether that change produces a reproducible quality or efficiency benefit.

Before starting a run, write one claim in a fixed form:

> Holding ______ fixed, change only ______, and measure whether ______ changes.

For example: hold the tokenizer, effective training tokens, sequence length, optimizer, and hardware fixed; enable the TTT inner loop; then measure held-out loss, fast-weight state, and training cost. Without this sentence, an experiment can change several factors at once and describe an outcome without explaining its cause.

## The identity of a run

A run is more than a command. Treat it as an object with an identity:

$$
R=(S,C,I_{\mathrm{data}},E,K,Q_{\mathrm{budget}},O),
$$

where:

- \(S\): source commit;
- \(C\): parsed model and training configuration;
- \(I_{\mathrm{data}}\): data, tokenizer, and manifest fingerprints;
- \(E\): operating system, Python, JAX, driver, and device information;
- \(K\): random seed, optimizer, and sampler state;
- \(Q_{\mathrm{budget}}\): training tokens, sequence length, batch, steps, and cost budget;
- \(O\): logs, checkpoints, evaluations, and failure reports.

Reproduction does not always mean bitwise identical floating-point values. It means that the objects defining the experiment are the same, and that remaining numerical differences are recorded.

## Variables that must be fixed or reported

| Category | What to record |
|---|---|
| Data | sources, cleaning version, deduplication, split manifests, effective tokens |
| Representation | tokenizer files, vocabulary, special tokens, packing, masks |
| Model | hidden size, layers, parameter count, layer placement, initialization |
| Algorithm | loss, optimizer, learning rate, warmup, clipping, inner-loop rule |
| Budget | sequence length, batch, steps, effective tokens, FLOPs or measured time |
| System | GPU, device count, dtype, compiler, parallelism, communication |
| State | seed, global step, optimizer state, sampler state, cache, fast weights |

A comparison may change one of these categories, but the changed item must be named as the research variable. If architecture, tokenizer, and training tokens all change, call it a new recipe rather than an architecture ablation.

A non-attributable example is a control with a byte tokenizer and sequence length 16, compared with a treatment that uses a subword tokenizer and sequence length 32 while also reporting lower loss. The result mixes vocabulary, token count, and context length. It may be a useful new recipe, but it cannot establish that a TTT component caused the change.

## Three levels of evidence

Report every run at three levels.

**Runnable** means that dependencies, devices, configuration, and data entry points are valid and the program produces output.

**Learnable** means that loss and gradients are finite, training state changes, checkpoints can be loaded, and the resumed next step agrees with continuous training within a declared tolerance.

**Interpretable** means that the control and treatment differ only in the declared variable, evaluation uses an independent and trustworthy manifest, and the result contains quality, cost, and state evidence sufficient for a bounded claim.

A four-step tiny run can establish that the interface and optimization path are connected. It does not establish language ability or architectural superiority.

## The run directory must explain itself

A formal run should preserve at least:

~~~text
experiment.json   # inputs, parsed configuration, and budget
source.json       # commit and worktree state
environment.json  # Python, JAX, devices, drivers, and dtype
data.json         # manifest, tokenizer, and data fingerprints
metrics.jsonl     # loss, gradients, speed, and state statistics
status.json       # successful, interrupted, or failed stage
checkpoint/       # model, optimizer, and resumable state
report.md         # interpretation and conclusion boundary
~~~

JSON, JSONL, and checkpoints are the factual source. The Markdown report is the reading interface; every number in it must point back to the machine-generated record.

For <code>tiny-contract-001</code>, the reader should be able to trace each claim to one artifact:

| Claim | Primary artifact | Passing condition |
|---|---|---|
| The intended configuration ran | <code>experiment.json</code> | resolved values match <code>configs/tiny.json</code> |
| The intended device ran | <code>environment.json</code> | backend, device count, and dtype are present |
| The model updated | <code>metrics.jsonl</code> | finite loss and nonzero update or parameter summary change |
| The run can resume | <code>latest.json</code>, checkpoint, <code>status.json</code> | step and next input order continue |
| The comparison is attributable | <code>report.md</code> and control/treatment table | only the declared variable differs |

## Resume is different from initialize-from

`resume` continues the same optimization process. It must restore model parameters, optimizer state, global step, sampler state, data fingerprints, and learning-rate progress.

`initialize-from` uses old parameters to start a new experiment. It receives a new run identity and may create a new optimizer, step counter, and data order.

Loading parameters alone is not the same as resuming training.

## Failures are evidence

Failures may occur during installation, device discovery, XLA compilation, data loading, gradient updates, checkpoint writing, or restoration. Preserve the command, job ID, last parsed configuration, environment summary, existing metrics, failed stage, and whether the run can be resumed safely.

The purpose of a failure record is to reduce the unknowns in the next run, not to make the project look uninterrupted.

## Bound every conclusion

Every report should answer:

1. What actually changed?
2. Which measurements changed?
3. Can those changes be attributed to the declared variable?
4. Do they reproduce across another seed, dataset, or scale?
5. Which stronger claims still lack evidence?

Use the following form when needed:

> In the tiny setting, holding \(A\), \(B\), and \(C\) fixed, changing \(D\) produced \(E\). This supports \(F\), but does not establish \(G\) because \(H\) has not been controlled or tested.

The environment chapter answers where a run executes; the data and representation chapters answer what the model sees; the baseline chapter answers whether the system learns; architecture chapters answer how state changes; and the experiment chapters answer whether the gain comes from that change.

## Summary

- A run identity contains source, resolved configuration, data fingerprints, environment, random and optimizer state, budget, and evidence.
- A control/treatment comparison must name its fixed inputs and its one declared change before training.
- Runnable, learnable, and interpretable are separate thresholds.
- A training checkpoint, a session snapshot, and a report serve different recovery and interpretation purposes.
- The next step for <code>tiny-contract-001</code> is to create sample tokens in the environment chapter and fill this contract with real output.
