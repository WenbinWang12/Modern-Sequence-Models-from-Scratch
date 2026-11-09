# 1. The Experiment Contract: Making a Run Interpretable

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
