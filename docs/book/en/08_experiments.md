# 20. Ablations, cost, and scaling: attributing an improvement

## From completing a run to answering a question

Suppose two jobs finish with validation NLLs of 2.10 for A and 1.98 for B. Is B's architecture better? Not yet: B might have seen more data, used a different tokenizer, or taken twice as long. An experiment needs an attributable question before it needs a result.

We place the previous chapters' implementations inside a comparison protocol. The central question is how much quality changes when the same architecture enables test-time updates, and what that change costs. We define controls, calculate a quality–cost table, and then distinguish the evidence supplied by configuration generation, profiling, and scaling fits. The mathematics needed is means, variance, and power functions, explained in context.

<figure class="book-figure">
  <img src="../../../assets/figures/experiment-loop.svg" alt="An experimental cycle from hypothesis and controls to execution, evidence, and attribution">
  <figcaption><strong>Figure 6. Closing the experimental loop.</strong> Follow the question through configuration, execution, evidence, and judgment. Every conclusion should lead back to the configuration and measurement that produced it, not just an experiment name.</figcaption>
</figure>

## Decide which one thing changes

In the <code>C2/F2</code> (or <code>C4/F4</code>) comparison, both arms retain the same fast-MLP architecture and initial parameters; only inner-loop updating changes. The number identifies the full-attention layer count. This asks whether adaptation helps. Removing MLP-prime from the C arm changes both parameter count and the forward function. That asks about the combined effect of extra structure and adaptation, and cannot attribute the whole difference to updating.

Write the protocol before launching training. “The same” should identify concrete files and hashes, not merely say that both runs use a similarly named dataset.

| Item | Constraint for an update ablation | Why it matters |
|---|---|---|
| Tokenizer, vocabulary, special tokens | Same versions and files | Define the prediction task and token count |
| Split, cleaning, packing, mask | Same manifest and sampling rules | Define the distribution actually observed |
| Architecture, initialization, precision | Same configuration and paired seed | Control capacity and starting point |
| Batch, learning rate, optimizer, schedule | Same unless explicitly studied | Control optimization |
| Inner-update switch | The deliberately changed variable | Define the treatment |
| Evaluation text, context, state reset | Same and documented | Control advantages from evaluation state |

Matching initialization does not imply matching the entire training trajectory. Updates change outer-loop gradients, so later static parameters can diverge. We control the starting point and protocol, not force the final weights to match. Evaluation must also specify whether fast state crosses document boundaries: reset single-document performance and continuous-stream adaptation answer different questions.

Structural ablations can follow: removing the fast MLP, changing suffix depth, modifying attention layout, enabling MoE, or repeating blocks. Recount parameters and cost for every change; a “125M” recipe label is not an exact parameter measurement.

## Calculate quality and cost together

The following numbers are **hypothetical teaching values, not repository results**. Both arms have 128M parameters and process the same \(D_{\rm valid}=1{,}000{,}000\) valid target tokens. Times exclude compilation and data preparation.

| Arm | Inner updates | Validation NLL | Steady training valid tokens/s | Training time | Peak memory |
|---|---|---:|---:|---:|---:|
| C | Off | 2.10 | 20,000 | 50.0 s | 10 GiB |
| F | On | 1.98 | 12,000 | 83.3 s | 13 GiB |

Time is \(t=D_{\rm valid}/r\), so \(1{,}000{,}000/12{,}000\approx83.3\) seconds. With the same tokenizer and evaluation targets, NLL improves by \(0.12\) nat/token, corresponding to perplexity ratio \(e^{-0.12}\approx0.887\). F's time per token is \(20{,}000/12{,}000\approx1.67\) times C's. The table supports a quality gain with increased cost under a fixed-token budget.

Under a fixed 50-second budget, C processes one million valid tokens while F processes about 600,000. The NLL of 1.98 belongs to F after one million tokens; it cannot be assigned to the earlier 600,000-token checkpoint. A time-matched claim requires evaluating that checkpoint. If the objective is reaching a target NLL, compare when each learning curve first reaches the threshold instead of comparing final points alone.

Valid-token count is \(D_{\rm valid}=\sum_{k,b,t}m_{k,b,t}\), not simply steps times batch times length. Padding, document-boundary exclusions, and other masks can make those counts differ. Record both scheduled positions and supervised targets to explain why 100 steps need not contain equal training information. Per-token NLL across different tokenizers is also not directly comparable; hold the task fixed or add a metric normalized by a common raw-text unit.

For a training-FLOP budget, let valid-token costs be \(f_C,f_F\). Matching compute requires \(D_Cf_C\approx D_Ff_F\). Those costs must correspond to actual inner updates, higher derivatives, sequence lengths, and precision. A generic \(6ND\) dense-training estimate does not automatically cover TTT trajectories, the MoE derivative reference path, or long-context attention overhead.

## Why one seed is insufficient

Initialization, data order, and execution can introduce variation. Pair C and F by seed to reduce unrelated differences, then analyze each pair's difference. Suppose three paired seeds produce NLL differences \(-0.10,-0.12,-0.08\), again hypothetical:

$$
\bar\delta=-0.10,\qquad
s=\sqrt{\frac{(0)^2+(-0.02)^2+(0.02)^2}{3-1}}=0.02,\qquad
\operatorname{SE}(\bar\delta)=\frac{s}{\sqrt3}\approx0.0115.
$$

The mean describes observed improvement; sample standard deviation describes seed variation; standard error describes uncertainty in the estimated mean under assumptions such as approximately independent repetitions. Three seeds do not establish stability across all domains or scales. Do not draw an unexplained “95% confidence interval” from them, or treat token positions as independent training runs.

Keep per-seed results, failed jobs, and exclusion reasons. Selecting favorable seeds after seeing outcomes changes the question answered by any subsequent average.

## Generate configurations and inspect the result

<code>ttt/experiments.py</code> generates dense, hybrid, TTT, and optional MoE/Loop ablations. On a remote machine with the environment installed, run from the repository root:

~~~bash
.venv/bin/python -m ttt.experiments \
  --output outputs/ch20/configs \
  --sizes 125m \
  --tokens 1000000 \
  --include-moe
~~~

Expect <code>outputs/ch20/configs/manifest.json</code> and experiment JSON files under <code>125m/</code>. Standard output also prints the manifest. Inspect experiment paths, <code>scheduled_tokens</code>, and <code>pending</code> compute-matched arms awaiting cost measurements. Rounding steps upward can schedule slightly more than one million positions; this is also not the masked valid-target count.

This command generates configurations. It does not train, download a tokenizer, or prepare data. The recipe references a Llama tokenizer; training still requires checking access, actual tokenizer files, and vocabulary alignment. If output is missing, inspect argument errors and directory permissions first. If compute-matched arms remain pending, provide cost records instead of manually declaring completion. Configuration generation does not prove trainability or matched compute.

Cost input records <code>training_flops_per_token</code> by experiment name and labels <code>method</code> as <code>profiler</code> or <code>xla_estimate</code>. The generator adjusts baseline steps using the ratio of per-token costs. Measurements must match the corresponding configurations; tiny-model costs cannot be substituted for 125M-model costs.

## Distinguish inference profiling from training-cost estimates

A successful tiny mesh validation in the preceding chapter leaves the real checkpoint <code>outputs/ch19/checkpoints/mesh-step-2.orbax</code>. This command uses it with context plus decoding length \(8+4=12\), below its maximum position 16:

~~~bash
JAX_PLATFORMS=cuda .venv/bin/python -m ttt.profile \
  --checkpoint outputs/ch19/checkpoints/mesh-step-2.orbax \
  --context-length 8 \
  --decode-tokens 4 \
  --training-flops
~~~

Standard output is JSON containing <code>parameters</code>, <code>prefill_seconds</code>, <code>decode_seconds</code>, <code>decode_tokens_per_second</code>, <code>state_bytes</code>, and <code>inner_updates</code>. Additional training-estimate fields are <code>xla_estimated_training_flops</code> and <code>xla_estimated_flops_per_token</code>. The command does not write a default report file; redirect output into the experiment directory and retain the command and environment when recording a measurement.

<code>profile_session</code> performs one full warmup run and then one measured run, waiting for device results before reading timings. It measures single-session prefill and token-by-token decode, not training tokens/s or concurrent serving throughput. <code>device_peak_bytes</code> comes from device statistics, can be absent, and is a process-level peak rather than memory uniquely added by this measurement. <code>parameters.fast_mlp</code> counts the initial fast-state subtree, already included in total; do not add it twice.

Training FLOPs come from compiled graph cost analysis, not hardware counters. They do not include the entire input pipeline, optimizer, and checkpoint overhead; unavailable backend statistics can produce null. If the checkpoint is missing, complete the previous validation first. If lengths fail, inspect the saved configuration. If estimates are unavailable, retain that fact rather than inserting zero. When graphs differ substantially, report estimates together with measured training time.

## A scaling hypothesis must face held-out scales

A result at one size does not automatically extrapolate. An empirical model can express loss as a function of parameters \(N\) and training targets \(D\). <code>ttt/scaling.py</code> uses normalized variables:

$$
\widehat{\mathcal L}(N,D)
=L_\infty+
A\left(\frac{N}{10^6}\right)^{-\alpha}
+B_s\left(\frac{D}{10^9}\right)^{-\beta}.
$$

\(L_\infty\) is the fitted floor term; \(A,B_s\) set scales; \(\alpha,\beta\) describe rates of decrease. The subscript separates coefficient \(B_s\) from batch size \(B\); JSON retains the code's keys <code>E</code>, <code>A</code>, and <code>B</code>. Doubling \(N\) multiplies the parameter term by \(2^{-\alpha}\); it does not necessarily halve total loss. This is an empirical fit to a fixed protocol, not a theorem covering every architecture.

The fitter requires at least eight training results, three distinct parameter counts, and three distinct token counts. Each input row provides <code>parameters</code>, <code>tokens</code>, and <code>loss</code>; <code>held_out: true</code> excludes a scale from fitting. The guard checks counts and distinct values. Researchers must still design crossed coverage so that “larger models always train longer” does not confound the two effects.

Output includes training RMSE, held-out predictions, and bootstrap intervals. Small training error does not establish reliable extrapolation. Resampling existing observations does not automatically include contamination, implementation errors, or model misspecification. Reserve scales first, fit second, and inspect held-out error third. Repeatedly choosing formulas to match the held-out points requires a new final validation set.

A set of configuration files is not a research conclusion. Quality, cost, repeated runs, and held-out scales must support that conclusion together. The next chapter turns this principle into explicit admission criteria for larger training.
