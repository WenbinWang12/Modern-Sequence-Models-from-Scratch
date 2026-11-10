# 15. Dense causal-language-model baseline

The dense baseline is the control for TTT, Loop, and MoE. It fixes tokenizer, manifests, sequence length, effective token budget, initialization, optimizer, schedule, seed, precision, hardware, and checkpoint format before adding dynamic state or routing.

For logits \(z\) at position \(t\) and target \(y_t\):

$$
\ell_t=-\log\frac{\exp z_{t,y_t}}
{\sum_v \exp z_{t,v}}.
$$

The repository can compute cross-entropy in token and vocabulary blocks, preserving the same objective while reducing peak memory. Masked loss uses the sum of valid-token losses divided by the valid-token count.

The baseline disables experts, fast MLP, inner updates, and loop mode. It can still choose sliding or full attention, but attention type must be recorded as a separate variable.

Each training step samples a window, runs sequence_loss, computes outer gradients, applies Optax, records loss and state statistics, and periodically checkpoints. A finite decreasing loss is a health signal, not evidence of broad capability; tiny data can simply be memorized.

## What the baseline disables

~~~text
num_experts = 0
use_fast_mlp = false
inner.enabled = false
loop_mode = none
~~~

These switches remove routing, extra fast state, and depth recurrence. Attention may still be full or sliding, but attention type must be reported as its own variable. The `tiny` configuration is a pipeline check; the 125M and 760M directories contain recipe and ablation candidates, and their names do not guarantee an exact final parameter count after vocabulary or optional modules are changed.

## One training step and one recovery step

The training path creates the AdamW schedule from the resolved configuration, samples a window and mask, computes `sequence_loss`, differentiates the outer objective, clips and updates gradients, records state statistics, and writes a checkpoint at the configured interval. With `inner.enabled=false`, it still uses the same sequence-loss path and only disables the fast-state update; this makes the C/F comparison easier to interpret.

Recovery is part of the baseline. A resumed run must restore parameters, optimizer statistics, global step, sampler state, data fingerprint, and the next input order. Repeating a checkpoint and comparing the next loss to an uninterrupted run is a stronger check than verifying that a file can be opened.

## What a falling loss does and does not show

For a teaching run, check finite loss, nonzero and bounded gradients, no immediate NaNs, continuity after restore, and held-out loss from an independent manifest. A falling training loss alone is not evidence of capability: a tiny corpus can be memorized. If training loss falls while held-out loss rises, inspect split contamination, duplicate documents, and sampling before changing the architecture.

## Parameters, activation memory, and measured cost

Parameter count is the number of leaves in the model tree, but it is not activation memory. TTT may retain or recompute fast weights, gradients, and chunk state; MoE may have many total parameters while activating only a few experts per token. `ttt/profile.py` reports parameter counts, fast-MLP counts, prefill/decode time, state bytes, peak device memory, and XLA cost analysis. Cost analysis is an estimate; steady throughput requires a real-device run after excluding compilation.

For each later comparison, report data and tokens, total and active parameters, optimizer and inner-loop settings, hardware and precision, compile and steady cost, memory, communication, training and held-out loss, and restore behavior. The next part follows a token through the TTT forward path.
