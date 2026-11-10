# 18. Attention, caches, and streaming inference

## One model, two timelines

During training, a batch can contain the whole sequence. During generation, a serving process usually receives one new token at a time. Full forward can construct a causal mask over all positions; streaming forward must remember the past without recomputing it at every step.

For a token at position \(t\), causal attention may read keys and values from positions \(1,\ldots,t\), but never from \(t+1\). A KV cache stores those past keys and values. If the query has shape `[batch, query_heads, 1, head_dim]` and the cache has shape `[batch, kv_heads, cache_length, head_dim]`, grouped-query attention reuses each KV head for the corresponding query-head group. The cache stores history; it is not another set of trainable parameters.

## Prefill and decode in a concrete example

Suppose a request starts with tokens \(x_1,x_2,x_3\). **Prefill** processes all three, writes their keys and values, and returns the logits for the last position. The next generated token \(x_4\) is handled by **decode**: read the three cached positions, compute one new query/key/value, predict the next token, and append the new key/value. Recomputing \(x_1,x_2,x_3\) each time may produce similar logits but measures a much more expensive serving algorithm.

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

<figure class="book-figure">
  <img src="../../../assets/figures/attention-kv.svg" alt="Prefill builds the KV cache, while decode reuses history and appends one new key and value">
  <figcaption><strong>Figure 4. Prefill and decode.</strong> The KV cache stores past keys and values; it is not trainable, but it must be restored together with positions, recurrent state, and TTT fast weights.</figcaption>
</figure>

## Two tokens by hand: Q, K, V, and the causal mask

Let the query, key, and value width be \(2\). Choose

$$
q_1=(1,0),\quad q_2=(1,0),\quad
k_1=(1,0),\quad k_2=(0,1),
$$

$$
v_1=(10,0),\qquad v_2=(0,10).
$$

Scaled dot-product attention uses

$$
s_{t,j}=\frac{q_t k_j^{\mathsf T}}{\sqrt{d_h}}.
$$

At position 1, the causal mask hides position 2, so the only visible score is \(1/\sqrt{2}\). Softmax over one visible entry gives weight \(1\), and the output is \(v_1=(10,0)\).

At position 2, both keys are visible. The score vector is

$$
\left(\frac{1}{\sqrt{2}},0\right)\approx(0.707,0).
$$

Its softmax is approximately \((0.67,0.33)\), so the output is

$$
0.67v_1+0.33v_2\approx(6.7,3.3).
$$

The query expresses what the current position is looking for, the key expresses how well each history position matches, and the value carries the information that is retrieved. The causal mask decides which keys are allowed to enter the softmax; it is therefore part of the model's semantics, not just a memory optimization.

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

where the factor 2 counts keys and values. A sliding window caps \(L\) at the window size. Cache capacity is not identical to `max_position_embeddings`: the former is storage allocated for a session, while the latter is the maximum legal model position. `grow_cache` must copy valid entries and preserve positions; silently overwriting old entries corrupts causality.

## A concrete KV-memory calculation

For one layer, let \(B=1\), \(H_{\mathrm{KV}}=2\), \(L=4096\), \(d_{\mathrm{head}}=64\), and bfloat16 storage with \(s=2\) bytes. Then

$$
2\cdot1\cdot2\cdot4096\cdot64\cdot2
=2{,}097{,}152\ \text{bytes}\approx2\ \text{MiB}.
$$

Twelve layers would use about \(24\) MiB for KV alone. Activations, fast weights, allocator fragmentation, and other session fields are additional. This is why reducing \(H_{\mathrm{KV}}\), using a sliding window, or changing dtype changes serving memory even when the hidden width is unchanged.

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
