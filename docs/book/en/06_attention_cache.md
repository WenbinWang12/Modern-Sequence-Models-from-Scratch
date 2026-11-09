# 18. Attention, caches, and streaming inference

## One model, two timelines

Training commonly processes a complete batch while inference receives one token at a time. Full forward can construct a causal mask directly; streaming inference must carry past information in KV cache, Delta state, and TTT fast weights.

Prefill processes \(x_{1:L}\) and produces hidden states, cache, and fast weights. Decode processes the next token \(x_{L+1}\), reads the cache, produces the next prediction, and appends the new state.

Prefill establishes context. Decode reads and appends state incrementally. Recomputing all history may be numerically correct but does not represent serving cost; an off-by-one cache position can silently change the output.

## State lifetimes

| State | Role | Lifetime |
|---|---|---|
| KV cache | past keys and values | request or session |
| Delta/GDN state | recurrent linear-attention state | streaming session |
| TTT fast weights | locally updated parameters | experiment-defined session |

These are not static parameters. Their batch, layer, head, position, and dtype must be explicit, and snapshots must support restore and rollback.

## Code and validation

ttt/models/ttt_e2e/attention.py implements causal attention, GQA, and KV cache. gated_delta.py implements recurrent state. ttt/runtime.py combines prefill, decode, chunk boundaries, and random state. ttt/session_store.py provides atomic snapshots and capacity limits.

~~~bash
bash scripts/validate_gpu.sh outputs/gpu-validation.json
~~~

The report compares full forward with prefill plus decode, chunk boundaries, random streams, snapshot restore, and capacity failures under the same initial state. The runtime tests cover causality, consistency, and session storage.

## Transition

Once a dense block has explicit state boundaries, MoE extends the problem to routing, shared experts, and cross-device shards.
