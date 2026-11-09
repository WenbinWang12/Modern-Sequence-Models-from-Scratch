# 16. Inner and outer loops

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

## Transition

Fast weights create a changing state. Attention caches and Gated Delta determine how state moves across tokens, chunks, and requests.
