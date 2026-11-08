# 4. Forward order and fast weights

## What TTT-E2E changes

TTT-E2E does not simply replace an MLP. Selected later blocks contain an additional MLP-prime that can be updated at test time. The original MLP remains a stable static transformation; MLP-prime absorbs local context through an inner loop.

~~~text
x
 ├─ causal attention ──┐
 ├─ static MLP ─────────┼─ residual output
 └─ fast MLP-prime ─ inner update ─┘
~~~

The model predicts with the current fast weights, computes the inner loss, and then updates the fast state. The boundary at which the new state is used is part of the configuration.

## Three boundaries

The implementation must make three boundaries visible:

1. compute boundary: which layers contain MLP-prime;
2. state boundary: which session owns fast weights, KV/Delta state, and incomplete chunks;
3. gradient boundary: whether higher-order gradients are retained and how static parameters receive outer gradients.

A shape check alone cannot detect suffix-label leakage. A causality check must verify that changing later tokens does not change earlier logits.

## Code and evidence

ttt/models/ttt_e2e/modeling_ttt_e2e.py organizes the full forward. The train and infer modules keep training and decoding paths separate. ttt/runtime.py owns per-session state.

~~~bash
bash scripts/validate_gpu.sh outputs/gpu-validation.json
~~~

The report compares full-sequence and streaming logits, causality, state shapes, and restore behavior under the same initial state and mask.

## Transition

The forward path says where fast weights are used. The next chapter defines exactly when they are updated and how that update interacts with the outer optimizer.
