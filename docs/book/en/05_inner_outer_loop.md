# 17. Inner and outer loops

## Two optimization timescales

The inner loop updates fast weights inside a sequence chunk so the model can adapt to the current context. The outer loop updates static parameters so the model learns how to adapt. Their states, losses, and optimizers must remain separate.

~~~text
fast_0 ──forward(prefix)──> loss_0 ──SGD──> fast_1
fast_1 ──forward(suffix)──> loss_1
                              │
                              └── outer loss → static parameters
~~~

The safe order is predict, compute loss, then update. Exposing a label before the prediction creates leakage.

## Gradient semantics

Retaining higher-order gradients is closer to a full meta-learning objective but costs memory and compilation time. Truncating the inner update is useful for validating the system path. The choice must be recorded because the two procedures are different algorithms.

The outer optimizer, usually AdamW, updates static parameters, normalization, and routing parameters. Fast weights may use SGD or a local optimizer. Their learning rates, weight decay, and clipping rules should be recorded separately.

## Code and controls

ttt/inner_loop.py implements block prediction, token loss, SGD, and scan. ttt/loss.py handles masks. ttt/training.py owns AdamW and training state. ttt/checkpoint.py saves static state and sampler state; session fast weights belong to runtime.

Set inner.enabled=false for the static C path and true for the F path. Keep tokenizer, width, training tokens, and data order fixed. The evidence should include inner steps, fast-weight norms, finite loss, outer gradient norms, and the next loss after restore.

## What `inner_step` does

`inner_step` receives the model, static parameters, current fast weights, hidden states, targets, masks, cache, and position. Its objective follows one explicit order:

1. run the suffix with the current fast weights;
2. project hidden states to logits or a vocabulary-block loss;
3. apply the token mask;
4. differentiate with respect to fast weights;
5. clip the global fast-weight gradient and apply SGD;
6. optionally compute a post-update loss.

The pre-update loss creates the update. A post-update loss measures whether the same block became easier after adaptation; it is not the outer loss and is normally disabled because it adds another forward pass. The inner clip norm and outer clip norm are separate experiment fields.

## How `sequence_loss` scans a sequence

The sequence is padded to a multiple of `ttt_chunk_size`. The prefix runs once for all positions, while the suffix uses scans over chunks and, when configured, steps within a chunk. Checkpointing the scan trades recomputation for memory; it does not change the meaning that one chunk updates the fast state used by the next chunk.

Each carry contains the current fast weights and cache. The outputs include token losses, valid-token counts, gradient norms, router loads, fast-weight norms, update norms, and optional post losses. The final partial chunk is scored with a mask, while `where` keeps padding from changing the fast state.

## Weighted outer loss

Samples can contain different numbers of valid labels. The batch objective weights each sample by its valid-token count:

$$
\mathcal{L}_{\mathrm{batch}}=
\frac{\sum_b n_b\mathcal{L}_b}
{\max(\sum_b n_b,1)}.
$$

The training path then differentiates this objective with respect to static parameters and applies the outer optimizer. On a data axis it may use `pmean`. MoE auxiliary and router losses are reported separately from language-model NLL.

## C/F controls and evidence

C and F must share the same architecture, initialization, tokenizer, manifest, sample order, sequence length, batch, token budget, outer optimizer, schedule, hardware, and precision. C disables the inner update; F enables it. If F adds fast-MLP parameters that C does not contain, the comparison changes parameter count as well as update behavior and needs a separate structural control.

Use at least the following evidence: pre- and post-update loss, causality under changed suffix tokens, stable fast-weight update norms, finite outer gradients, continuity after restore, and held-out loss at the same budget. A single fast-weight norm change is not evidence of a quality gain.

## Transition

Fast weights create a changing state. Attention caches and Gated Delta determine how state moves across tokens, chunks, and requests.
