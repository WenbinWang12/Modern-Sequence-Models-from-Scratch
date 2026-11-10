# 16. Forward order: how one token travels through TTT-E2E

## Do not start with a kernel

TTT-E2E contains several new terms: prefix, suffix, fast weights, inner loss, MLP-prime, and KV cache. The safest first reading is to follow one token through the residual stream. Once the order is clear, the implementation details have a place to attach.

The model first looks up a token ID in an embedding table. Each block then sends the hidden state through attention or a gated-delta state update, followed by an MLP. Selected later blocks have one additional fast MLP-prime. The final hidden state is normalized and projected into vocabulary logits.

~~~text
token ids
   ↓
embedding
   ↓
attention / gated delta + residual
   ↓
fast MLP-prime + residual       (selected later blocks)
   ↓
static MLP or MoE + residual
   ↓
final norm → lm_head → logits
~~~

Every arrow is an interface for a shape, a mask, a cache, and a gradient. Moving MLP-prime before or after the ordinary MLP changes the algorithm. Updating it after every token, at the end of a chunk, or before the next chunk also changes the meaning of the state.

<figure class="book-figure">
  <img src="../../../assets/figures/ttt-timeline.svg" alt="The timing of token chunks, inner-loop fast-weight updates, and outer-loop static-parameter updates in TTT">
  <figcaption><strong>Figure 3. Two clocks in TTT.</strong> Each chunk predicts with the current fast weights and updates <span class="arithmatex">\( \phi \)</span> from observed labels; <span class="arithmatex">\( \theta \)</span> is updated across batches by the outer loop.</figcaption>
</figure>

## The order implemented in this repository

The forward path in <code>TTTE2EForCausalLM._block</code> first executes the attention branch. A layer whose type is <code>full_attention</code> or <code>sliding_attention</code> calls attention. A <code>gated_delta</code> layer instead updates a recurrent state. The branch output is added back to the hidden state, producing the attention residual.

If the configuration contains <code>fast_mlp</code>, the block then reads the current fast-weight version of MLP-prime. It applies normalization, SwiGLU, and an optional post-normalization, and adds the result to the residual stream. Only after that does the block execute the ordinary static MLP or the MoE branch.

The conceptual order is therefore:

~~~text
attention / gated delta
→ fast MLP-prime
→ static MLP or MoE
~~~

This is an additional adaptation path, not a replacement of the original feed-forward network. If fast weights are absent, the configuration and parameter tree do not silently create a fast path. A shape check can confirm that tensors fit, but it cannot confirm that this semantic order is correct.

## Prefix and suffix

The configuration divides the block stack into a prefix and a suffix. The prefix runs the static part once and produces hidden states for every position. The suffix consumes those hidden states while carrying the updateable fast weights.

This division avoids repeating all static computation inside the inner loop, but it creates an alignment requirement. The prefix cache, hidden states, positions, and masks must describe the same sequence positions that the suffix scans. A suffix is not an independent piece of text; it receives representations produced for those exact positions.

In the current training path, <code>sequence_loss</code> pads a token sequence to a multiple of <code>ttt_chunk_size</code> and reshapes it into chunks. It runs the prefix once, then scans the suffix chunk by chunk. The fast state starts at <code>initial_fast_weights</code>, and the next chunk receives the state returned by the previous chunk. If the final chunk contains padding, its masked labels must not change the fast state.

## Why fast weights are not ordinary parameters

Ordinary parameters are shared by many examples and updated by the outer optimizer. They survive from one training step to the next. Fast weights are temporary task or session state: an inner loss can change them while a sequence is being processed.

Use two time indices to keep the clocks separate:

$$
\theta_{k+1}
=
\operatorname{AdamW}\!\left(
\theta_k,\,
\nabla_\theta \mathcal{L}_{\mathrm{outer}}
\right),
$$

$$
\phi_{t+1}
=
\phi_t
-\eta_{\mathrm{inner}}
\nabla_\phi
\mathcal{L}_{\mathrm{inner}}
\left(x_t;\theta,\phi_t\right).
$$

Here \( \theta \) denotes static model parameters and \( \phi \) denotes fast weights. The outer loop learns how the fast state should be initialized and used; the inner loop adapts that state to the current context. Putting \( \phi \) directly into the AdamW optimizer state would merge two different time scales and would no longer describe TTT.

## Prefix prediction, label use, and causality

A safe causal update has this order:

~~~text
use fast_t to predict y_t
→ compute inner loss from the observed label
→ update to fast_{t+1}
→ let later chunks use fast_{t+1}
~~~

The label used for an inner update must be available at the point where the update happens. If a suffix label is used to update the state before the same position is predicted, the model receives future information and the measured loss is too optimistic.

The test is simple to state. Change a later suffix token while keeping the prefix fixed, then compare the logits at earlier positions. Earlier logits must remain unchanged up to the numerical tolerance. This test catches semantic leakage that tensor-shape tests cannot see.

The current inner step computes a value and gradient with respect to fast weights, then applies <code>sgd_update</code>. The outer loss differentiates through <code>sequence_loss</code>. Whether higher-order gradients are retained depends on the resulting computation graph and on any rematerialization or checkpointing policy. “The fast state changed” and “the outer loss differentiates through that change” are separate claims.

## A small chunk calculation

Let the sequence length be \(8\) and the chunk size be \(4\). Positions \(0\) through \(7\) form two chunks:

~~~text
chunk 0: positions 0, 1, 2, 3
chunk 1: positions 4, 5, 6, 7
~~~

Chunk 0 predicts with \( \phi_0 \), obtains \( \mathcal{L}_0 \), and updates to \( \phi_1 \). Chunk 1 predicts with \( \phi_1 \), obtains \( \mathcal{L}_1 \), and may then produce the next state. If the last chunk has only two real positions, the other two are padding. The mask sets those label weights to zero, so padding contributes neither loss nor an inner update.

This is why chunk size is an algorithmic variable, not merely a batching detail. A smaller chunk lets the fast state observe context more often but increases scan and update overhead. It also changes the boundary through which gradients and caches are carried.

## How to validate that the forward pass is semantically correct

A useful validation set contains at least four checks:

1. Compare the dense path with fast MLP disabled, including shapes and numerical values.
2. Change suffix tokens and verify that prefix logits do not change.
3. Compare full-sequence execution with prefill/decode streaming under the same initial state and masks.
4. When changing chunk size, record whether it is an intended algorithm change or only an implementation optimization.

The GPU validation script writes architecture and runtime reports. Every report must be tied to the configuration, initial parameter state, mask, and software revision. One passing sequence length does not prove that every <code>layer_type</code>, chunk size, or session boundary is correct.

The next chapter turns this forward order into an explicit inner loop and outer loop, and follows the loss through both time scales.
