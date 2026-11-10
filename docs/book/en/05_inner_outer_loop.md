# 17. Inner and outer loops: two optimization timescales

## The basic picture

There are two different things that can change during a TTT run. Static parameters $\theta$ are learned across training steps. Fast weights $\phi$ are temporary parameters that adapt inside a sequence or serving session. Calling both of them “the model update” hides the algorithm we are trying to study.

For one chunk, the causal order is:

~~~text
static theta + fast phi_0
        |
        | predict with the observed prefix/suffix
        v
      loss_0  --gradient w.r.t. phi-->  phi_1
        |
        v
  outer loss --gradient w.r.t. theta--> AdamW update
~~~

The rule is simple: predict first, reveal the next label only when it is available, then update. If a label is used before its prediction, the update leaks the answer and the experiment no longer measures causal adaptation.

## A two-token example

Imagine that the current chunk contains tokens `[A, B, C]`. The model first uses the current fast weights to predict `B` from `A`, and then `C` from `A,B`. Their masked token losses form an inner objective, for example

$$
\ell_{\mathrm{inner}}(\phi)=\frac{1}{2}\bigl[-\log p_{\phi}(B\mid A)-\log p_{\phi}(C\mid A,B)\bigr].
$$

An SGD step is

$$
\phi_1=\phi_0-\eta_{\mathrm{inner}}\nabla_{\phi}\ell_{\mathrm{inner}}(\phi_0).
$$

The outer objective may include later positions or the whole batch. It differentiates through the static model path and, depending on the chosen algorithm, through the fast update itself. The inner loss says “adapt to this context”; the outer loss says “learn parameters that make adaptation useful.”

## A scalar update that we can calculate by hand

Compress the fast-weight tree to one scalar \( \phi \). Suppose the current chunk wants this scalar to approach \(3\), with

$$
\ell_{\mathrm{inner}}(\phi)=\frac{1}{2}(\phi-3)^2.
$$

If \( \phi_0=1 \), the derivative is

$$
\left.\frac{\partial\ell_{\mathrm{inner}}}{\partial\phi}\right|_{\phi_0}
=\phi_0-3=-2.
$$

With \( \eta_{\mathrm{inner}}=0.5 \), ordinary SGD gives

$$
\phi_1
=\phi_0-\eta_{\mathrm{inner}}(-2)
=1-0.5(-2)=2.
$$

The real model replaces the scalar with many matrix leaves and replaces the squared error with masked token cross-entropy, but the update has the same meaning: the observed chunk moves the fast state toward a state that predicts its context better.

The repository also clips the global fast-weight gradient. If the norm is \(2\) and the clip norm is \(0.5\), the scale is \(0.5/2=0.25\), so the effective update is

$$
\phi_1
=1-0.5(0.25)(-2)=1.25.
$$

The learning rate and clip norm must therefore be reported separately. The final value \(1.25\) alone cannot tell us whether the update was small because the learning rate was small or because clipping was active.

## What `inner_step` actually receives

`ttt/inner_loop.py` keeps the boundary explicit. `inner_step` receives the model, static parameters, current fast weights, suffix hidden states, target IDs, token mask, cache, position, and inner learning rate. It performs these operations in order:

1. run the suffix with the current fast weights;
2. project features to logits or evaluate a vocabulary-block loss;
3. multiply by the token mask and count valid labels;
4. differentiate with respect to fast weights;
5. compute the global fast-weight norm, clip it, and apply SGD;
6. optionally run a post-update forward pass.

The pre-update loss creates the update. A post-update loss asks whether the same block became easier after adaptation; it is not the outer loss and is normally disabled because it adds a forward pass. The inner clip norm and outer clip norm are separate fields because they stabilize different optimization problems.

## Scanning chunks without changing causality

`sequence_loss` first checks the one-dimensional token sequence and pads its length to a multiple of `ttt_chunk_size`. The prefix is computed once for the relevant positions. The suffix is scanned over chunks and, when configured, over steps inside each chunk. A checkpointed scan may recompute a forward result during backpropagation to save memory; it does not change the state transition.

The carry contains the current fast weights and cache. Metrics include token losses, valid-token counts, gradient norms, router loads, fast-weight norms, update norms, and optional post losses. The final partial chunk is scored with a mask. Padding positions are handled with `where` so they do not update the fast weights.

## Weighted outer loss

Different samples can have different numbers of valid labels. If sample \(b\) has \(n_b\) valid tokens and mean loss \(\mathcal L_b\), the batch objective is

$$
\mathcal L_{\mathrm{batch}}=
\frac{\sum_b n_b\mathcal L_b}{\max(\sum_b n_b,1)}.
$$

This is a token-weighted mean: a sample with 100 valid tokens contributes 50 times as many token observations as a sample with 2, rather than receiving the same weight merely because both are array rows. `training.py` differentiates this objective with respect to static parameters, applies a data-axis `pmean` when configured, and passes the gradients to the outer Optax optimizer. MoE auxiliary and router losses remain separate from language-model NLL in the metrics.

## First-order versus full meta-gradient

The fast update can depend on $\theta$ through the hidden features and the gradient. A full meta-gradient keeps the chain rule:

$$
\frac{d\mathcal L_{\mathrm{outer}}}{d\theta}
=\frac{\partial\mathcal L_{\mathrm{outer}}}{\partial\theta}
+\frac{\partial\mathcal L_{\mathrm{outer}}}{\partial\phi_1}
\frac{\partial\phi_1}{\partial\theta}.
$$

With one SGD update, $\phi_1=\phi_0-\eta\nabla_\phi\ell_{\mathrm{inner}}$, so the last factor contains a Hessian-like derivative of the inner loss. A first-order approximation applies `stop_gradient` to that path. The two versions are different algorithms, not just two compiler settings; report the choice, memory, compile time, and quality separately.

## C/F controls and evidence

The control \(C_k\) and adaptive run \(F_k\) must share architecture, initialization seed, tokenizer, manifest, sample order, sequence length, batch, token budget, outer optimizer and schedule, hardware, and precision. \(C_k\) disables the inner update; \(F_k\) enables it. If \(F_k\) adds fast-MLP parameters that \(C_k\) does not contain, the comparison changes parameter count as well as update behavior and needs a separate structural control.

At minimum, collect pre- and post-update loss, a causality check under changed suffix tokens, stable fast-weight update norms, finite outer gradients, resume continuity, and held-out loss at equal token budget. A changed fast-weight norm alone is evidence that an update occurred, not that it improved prediction.

## A remote validation entry point

The repository's functional GPU validator exercises the same inner-loop and streaming path used by the tiny architecture checks:

~~~bash
bash scripts/validate_gpu.sh outputs/gpu-validation.json
~~~

The resulting JSON records finite loss and gradients, full-versus-streaming logits error, compilation time, execution time, and state bytes. It is a semantic gate, not a quality claim. The next chapter follows the states that make this update causal across prefill, decode, chunks, and sessions.
