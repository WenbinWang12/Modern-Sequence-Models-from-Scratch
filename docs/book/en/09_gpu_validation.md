# 21. GPUs, multiple devices, and checkpoints: when to scale a run

## Three kinds of success require three kinds of evidence

A tiny model producing finite GPU loss establishes that one small graph runs. It does not establish intended parameter partitioning, continuation after interruption, or sufficient memory for the target length and batch. Before scaling, identify which admission criteria have actually been met.

We retain the tiny model and two-device expert example, separating numerical correctness, distributed-state correctness, and target-configuration readiness. Prerequisites are tensor shapes, inner/outer loops, and checkpoint purpose. We begin with a concrete two-GPU shape table rather than assuming familiarity with device meshes.

| Level | Question | Current main entry point |
|---|---|---|
| Small-configuration functionality | Finite gradients, causality, batch/streaming agreement? | <code>scripts/validate_gpu.sh</code> |
| Distributed state | Sharded training and the next step after restore agree? | Tiny mode of <code>scripts/validate_mesh.sh</code> |
| Production-run admission | Target memory, sustained throughput, recovery capability? | Target-configuration trials and separate evidence |

The final row is an experimental admission policy, not an existing one-command certification. We distinguish current report fields from measurements still required on target hardware.

## Start with single-GPU functionality

After environment installation and GPU allocation on a remote node, run from the repository root:

~~~bash
bash scripts/validate_gpu.sh outputs/ch21/gpu-validation.json
~~~

The script explicitly requires the CUDA backend, writes <code>outputs/ch21/environment.json</code>, and runs <code>ttt.validate</code>. A successful report contains top-level <code>passed: true</code>, <code>environment</code>, and <code>cases</code>. Cases include dense, gdn, MoE adaptation variants, repeated-computation variants, and context lengths 64 and 512 by default. Here dense means the case does not use MoE; it does not mean fast MLPs or TTT updates are disabled.

Each case mainly checks three properties. First, sequence loss and every gradient leaf are finite. Second, streaming logits differ from the chunked training reference by at most \(10^{-3}\) in maximum absolute error. Third, changing second-half inputs preserves first-half logits under <code>atol=1e-5, rtol=1e-4</code>. The third check is an assertion, not a separate causality boolean in the JSON.

These thresholds belong to the current float32 tiny cases, not every precision and scale. Failure reports include <code>passed: false</code>, <code>error_type</code>, <code>error</code>, and environment information. For missing GPUs, inspect JAX/CUDA setup. For nonfinite values, inspect precision, learning rates, and clipping. For streaming or causal differences, inspect target shifts, update boundaries, and caches. Retain stderr: the JSON summary does not contain the complete traceback.

Success does not validate 32K or 128K context: default context cases are only 64 and 512. It does not run session snapshot restoration, real-data cleaning checks, or validation-set quality evaluation. Record the cases, lengths, and dtype with the report.

## Compilation and execution are different timing boundaries

JAX's first execution can require compiling a Python-described graph into a device program. Later calls with the same shapes and types usually reuse it. Shape changes can trigger new compilation. At minimum, distinguish compilation, sustained execution, and end-to-end time including input/output.

Single-GPU validation explicitly calls <code>lower(...).compile()</code> and records <code>compilation_seconds</code>. It then executes one compiled loss-and-gradient computation, waits for the result, and records <code>training_seconds</code>. That is **one** graph execution, not a multi-sample steady training mean. It also excludes the complete optimizer update and data input.

Consider hypothetical compilation of 35 seconds followed by ten one-second steps, each containing 2048 valid targets. Execution-only throughput is \(20{,}480/10=2048\) tokens/s; including compilation, the short job averages \(20{,}480/45\approx455.1\) tokens/s. Both are reportable with their boundaries. The latter does not isolate steady kernel efficiency; the former does not capture short-job startup cost.

A proper measurement warms up the target shapes and times several consecutive training steps, waiting for asynchronous device work at timing boundaries. Record valid-token counts, the step-time distribution, inclusion of data waits and checkpoints, device count, and peak memory. Without synchronization, a timer may measure host submission instead of device execution. Current validation scripts do not supply this complete steady-state distribution; renaming one field does not create the evidence.

## Two GPUs must fit the mesh dimensions

A mesh arranges devices along named axes such as data and expert. <code>create_mesh</code> reshapes the global device array to <code>[data, expert]</code>, so

$$
n_{\rm devices}=n_{\rm data}\,n_{\rm expert}.
$$

Two GPUs can form \((2,1)\) or \((1,2)\). A \((2,2)\) mesh requires four GPUs; two devices cannot simultaneously provide independent two-way data and two-way expert axes.

| Two-GPU topology | data | expert | Conceptual input distribution for global hidden \([8,16,16]\) | Actual parameter partition for four-expert weights \([4,16,16]\) |
|---|---:|---:|---|---|
| Distribute samples along data | 2 | 1 | \([4,16,16]\) per device | \([4,8,16]\) per device with default parameter sharding |
| Distribute experts along expert | 1 | 2 | Input \([8,16,16]\) on each device | \([2,16,16]\) per device |

The first row matters: default <code>shard_parameters=True</code> can partition expert matrices' second dimension along data. This is not pure data parallelism with all parameters replicated. Nonexpert matrices can also partition their first dimension along data. The hidden column illustrates sample distribution, not a guarantee for every intermediate activation. Inspect <code>tree_shardings</code> and actual shards to establish a parameter's placement.

Global batch must be divisible by \(n_{\rm data}\), and expert count must be divisible by \(n_{\rm expert}\). Checking these integer constraints first is usually more useful than changing batch sizes blindly after a communication failure.

## What the two-device restore check actually covers

Run the following in a remote job with exactly two visible GPUs. Use a new output directory for a repeat audit and retain earlier reports and checkpoints. The default mesh uses all visible devices; eight GPUs produce data=4, expert=2, incompatible with the built-in batch of 2. Request two GPUs or select two devices within the scheduler's allocated mapping; do not replace it with machine-global GPU indices.

~~~bash
bash scripts/validate_mesh.sh \
  outputs/ch21/mesh-validation.json \
  outputs/ch21/mesh-checkpoints
~~~

The default is <code>expert_parallel_size=2</code>. With exactly two devices, expect data=1 and expert=2, and <code>expert_shards</code> showing global expert dimension 4 and local dimension 2. The report also includes two <code>step_seconds</code>, corresponding <code>metrics</code>, <code>device_memory</code>, and <code>checkpoint_roundtrip: true</code> in tiny mode.

The script trains two steps, saves <code>mesh-step-2.orbax</code>, restores model, optimizer, and step, and compares local arrays. It then takes one further step from the original and restored states and compares parameters, optimizer state, and loss. Merely opening a file is much weaker: omitting Adam momentum can still produce a loadable checkpoint that follows a different trajectory.

The batch is two fixed short sequences. This restore check does not cover real data cursors, process termination mid-run, filesystem failures, or changing the device topology. Training recovery in <code>ttt/train.py</code> additionally restores sampling state; session recovery from the preceding chapter is another kind of state. These are not interchangeable meanings of “checkpoint passed.”

For divisibility failures, inspect mesh and shapes. For write failures, inspect existing checkpoint names, storage, and permissions. For next-step differences, inspect optimizer state, step, and precision; retain discrepancies instead of simply loosening equality checks. The first reported step includes compilation, and the second is one execution. They do not establish a sustained step-time distribution.

## A parameter-scale trial has limits too

<code>ttt.validate_mesh</code> also accepts <code>--scale 1b</code>. For this worked two-device example, keep exactly two visible GPUs. Its built-in batch of 2 also constrains other topologies: four devices give data=2 and divide the batch, whereas eight devices give data=4 and do not. Consider this larger parameter trial only after tiny validation and with enough target-device memory. It is not the minimum required checkpoint for this chapter:

~~~bash
JAX_PLATFORMS=cuda .venv/bin/python -m ttt.validate_mesh \
  --scale 1b \
  --expert-parallel-size 2 \
  --require-gpu \
  --output outputs/ch21/mesh-1b.json
~~~

Expected JSON has <code>scale: "1b"</code>, a measured <code>parameters</code> count, and finite metrics for two steps. To avoid writing a large extra checkpoint, this mode **skips restoration**, so <code>checkpoint_roundtrip</code> should be false. The scale name is a recipe tier, not a substitute for counting parameters.

Its position limit is only 16, and training sequence length is 4. It probes larger parameter and sharding execution, not 1B long-context pretraining reproduction. For OOM, inspect actual parameters, precision, sharding, and current memory occupancy; remember that this trial already uses a very short sequence. Success still leaves target length, target batch, data throughput, and large-checkpoint recovery unverified.

## Admission criteria for a substantial training run

Retain the following as a manual experiment review. These are not additional fields emitted by the current scripts. Missing evidence should be marked unverified, not inferred from tiny <code>passed</code>.

| Criterion | Evidence needed | Why scaling without it is premature |
|---|---|---|
| Consistent data and task | Manifest, tokenizer hashes, masks, valid-target statistics | The job may train a different task or contaminate evaluation |
| Causal and numerical correctness | Relevant finite-gradient, streaming, and future-perturbation checks | More compute amplifies implementation mistakes |
| Target configuration fits | Measured memory and headroom at target length, batch, precision | Tiny runs do not predict activation peaks |
| Sustained stability | Warmed-up step-time distribution, loss/gradient trends, valid tokens/s | One execution misses intermittent stalls or divergence |
| Training recovery | Model, optimizer, step, data cursor, and next-step comparison | Restore can silently change the data or optimization trajectory |
| Accounted cost | Compilation, training, data waits, saving, device-hours | Budget-matched comparisons become impossible |
| Traceable provenance | Commit, environment, command, configuration, logs, failures | Another machine cannot explain discrepancies |

Choose thresholds before the run, including memory headroom, acceptable numerical error, and measurement duration. There is no universal numeric threshold suitable for every device. On failure, keep the original error together with its triggering configuration and report. Script-provided error JSON is a starting point; stage attribution and repair evidence still need to be recorded.

“Ready to scale” is now a set of inspectable conditions rather than an absence of exceptions. Only then should the real corpus from the data chapters enter larger training, evaluated with the preceding chapter's quality–cost protocol.
