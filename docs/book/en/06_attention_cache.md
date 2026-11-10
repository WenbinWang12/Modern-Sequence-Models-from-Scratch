# 18. Attention, caches, and streaming inference

## One model, two timelines

During training, a batch can contain the whole sequence. During generation, a serving process usually receives one new token at a time. Full forward can construct a causal mask over all positions; streaming forward must remember the past without recomputing it at every step.

For a token at position (t), causal attention may read keys and values from positions (1\ldots t), but never from (t+1). A KV cache stores those past keys and values. If the query has shape `[batch, query_heads, 1, head_dim]` and the cache has shape `[batch, kv_heads, cache_length, head_dim]`, grouped-query attention reuses each KV head for the corresponding query-head group. The cache stores history; it is not another set of trainable parameters.

## Prefill and decode in a concrete example

Suppose a request starts with tokens (x_1,x_2,x_3). **Prefill** processes all three, writes their keys and values, and returns the logits for the last position. The next generated token (x_4) is handled by **decode**: read the three cached positions, compute one new query/key/value, predict the next token, and append the new key/value. Recomputing (x_1,x_2,x_3) each time may produce similar logits but measures a much more expensive serving algorithm.

TTT adds state beyond KV: a Delta/GDN recurrent state and possibly fast weights. A complete session therefore contains:

~~~text
session
├── position
├── KV cache or Delta state
├── TTT fast weights
├── incomplete chunk
└── random state
~~~

All fields must advance together. Restoring only the cache while leaving fast weights at a later state produces a different model; restoring the weights with the wrong position changes RoPE and masking.

## State lifetimes and memory

| State | Role | Lifetime |
|---|---|---|
| KV cache | past keys and values | request or session |
| Delta/GDN state | recurrent linear-attention summary | streaming session |
| TTT fast weights | locally adapted parameters | experiment-defined session |

For ordinary full attention, KV memory grows approximately as

$$
\text{bytes}_{\mathrm{KV}}
\approx 2\,B\,L\,H_{\mathrm{KV}}\,d_{\mathrm{head}}\,\text{bytes(dtype)},
$$

where the factor 2 counts keys and values. A sliding window caps (L) at the window size. Cache capacity is not identical to `max_position_embeddings`: the former is storage allocated for a session, while the latter is the maximum legal model position. `grow_cache` must copy valid entries and preserve positions; silently overwriting old entries corrupts causality.

## Code path

`ttt/models/ttt_e2e/attention.py` implements causal attention, GQA, and layer caches. `gated_delta.py` implements the recurrent state used by Delta layers. `ttt/runtime.py` combines prefill, decode, chunk boundaries, and random state. `ttt/session_store.py` provides atomic snapshots and capacity limits.

The runtime's `prefill` method calls the same observation path as a sequence of tokens, while `decode` supplies a one-token array. With `ttt_chunk_size>1`, a decode token may be placed into an incomplete chunk; the fast update is delayed until the next label closes the chunk. This is why `pending_features`, `pending_targets`, `pending_count`, and `chunk_start` are session fields rather than local variables.

## Why full and streaming outputs can differ

Compare per-position logits, not only the final generated string. Common causes of a mismatch are:

- position zero versus one-based indexing;
- an incorrect cache write index;
- a sliding-window boundary shifted by one token;
- recomputing prefix keys in a different dtype;
- applying a TTT update at different token boundaries;
- using a different random key for sampling;
- omitting one field from a session snapshot.

The tolerance belongs in the report because floating-point kernels and dtype change the expected error. A streaming check should also compare cache length, position, fast-weight norm, and pending-chunk count.

## Snapshot is not training checkpoint

`session_store.py` snapshots should be atomic and include model version, session ID, position, state digest, and protocol version. A training checkpoint additionally contains optimizer and sampler state. Loading one as the other can appear to succeed while resuming an unrelated lifecycle. Check the model parameter version and update policy before restoring, then verify the next token's logits.

## Remote validation

The repository provides a GPU validation entry point:

~~~bash
bash scripts/validate_gpu.sh outputs/gpu-validation.json
~~~

The report compares full forward with prefill-plus-decode, changed-suffix causality, chunk boundaries, state bytes, finite gradients, and compilation versus execution time. It is a functional gate; it does not claim a particular generation quality or latency on every GPU.

The next chapter adds multiple experts. Routing makes the state boundary dynamic and introduces communication between devices.
