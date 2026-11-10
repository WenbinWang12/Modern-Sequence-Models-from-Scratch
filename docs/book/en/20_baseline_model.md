# 15. Dense causal-language-model baseline

## From logits to loss

Logits are scores, not probabilities. If a vocabulary contains `cat`, `sat`, and `dog`, scores $(2,1,0)$ become probabilities only after softmax:

$$
p_v=\frac{\exp z_v}{\sum_u\exp z_u}.
$$

If the correct token is `sat`, this example gives probability about $0.245$ and negative log-likelihood about $1.407$. Raising the correct probability to $0.8$ lowers the contribution to $-\log(0.8)\approx0.223$. Cross-entropy therefore measures how much probability the model placed on the observed next token.

The dense baseline is the control for TTT, Loop, and MoE. It fixes tokenizer, manifests, sequence length, effective token budget, initialization, optimizer, schedule, seed, precision, hardware, and checkpoint format before adding dynamic state or routing.

For logits \(z\) at position \(t\) and target \(y_t\):

$$
\ell_t=-\log\frac{\exp z_{t,y_t}}
{\sum_v \exp z_{t,v}}.
$$

The repository can compute cross-entropy in token and vocabulary blocks, preserving the same objective while reducing peak memory. Masked loss uses the sum of valid-token losses divided by the valid-token count.

The baseline disables experts, fast MLP, inner updates, and loop mode. It can still choose sliding or full attention, but attention type must be recorded as a separate variable.

Each training step samples a window, runs sequence_loss, computes outer gradients, applies Optax, records loss and state statistics, and periodically checkpoints. A finite decreasing loss is a health signal, not evidence of broad capability; tiny data can simply be memorized.

The gradient with respect to one logit has a simple form:

$$
\frac{\partial\ell_t}{\partial z_{t,v}}
=p_{t,v}-\mathbf{1}[v=y_t].
$$

The correct token receives $p_{t,y_t}-1$, which pushes its score up; every other token receives its current probability, which pushes its score down. Backpropagation carries this local error through the embedding, attention, and MLP computation graph. AdamW then turns the gradients into a parameter update.

We can derive it by expanding the loss:

$$
\ell_t=-z_{t,y_t}+\log\left(\sum_u\exp z_{t,u}\right),
\qquad
\frac{\partial\ell_t}{\partial z_{t,v}}
=-\mathbf{1}[v=y_t]+\frac{\exp z_{t,v}}{\sum_u\exp z_{t,u}}.
$$

The second term is exactly the softmax probability, which gives the form $p_{t,v}-\mathbf{1}[v=y_t]$.

## One minimal gradient update

Use the same logits example to isolate the direction of the error signal. Let
\(\mathbf z=(2,1,0)\), with the second token (<code>sat</code>) as the target. Then
\(\mathbf p\approx(0.665,0.245,0.090)\), and
\(\partial\ell/\partial\mathbf z=\mathbf p-\mathrm{onehot}(y)\) gives

$$
\mathbf g\approx(0.665,-0.755,0.090).
$$

If logits were temporarily treated as directly updateable variables, a plain SGD
step with \(\eta=0.1\) would be

$$
\mathbf z'=\mathbf z-\eta\mathbf g
\approx(1.9335,1.0755,-0.009).
$$

The correct score rises from 1 to about 1.076, and its softmax probability rises
from about 0.245 to about 0.271. Real training does not optimize logits directly:
it propagates \(\mathbf g\) through the computation graph into the output
projection, hidden state, embedding, attention, and MLP, then AdamW updates the
parameters. This calculation demonstrates the local direction of one gradient;
it does not establish model capability.

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
