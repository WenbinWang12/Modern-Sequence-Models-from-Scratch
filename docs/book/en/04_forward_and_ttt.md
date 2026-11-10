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

If the layer's parameter tree contains <code>fast_mlp</code>, the block reads the current fast-weight version of MLP-prime. The corresponding architecture switch is <code>use_fast_mlp</code>. The branch applies normalization, SwiGLU, and optional post-normalization, then adds the result to the residual stream. Only afterward does the block execute the ordinary static MLP or MoE branch.

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

The current <code>sequence_loss</code> pads inputs, runs <code>model.prefix</code>, then reshapes its hidden features into chunks of <code>ttt_chunk_size</code>. It scans the suffix chunk by chunk. The fast state starts at <code>initial_fast_weights(params)</code>, and each complete chunk commits the state used by the next. An incomplete final chunk still scores real targets, but commits no fast-weight update at all, including updates from those real targets. This matches the streaming path's complete-chunk rule.

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
\phi_{c+1}
=
\phi_c
-\eta_{\mathrm{inner}}
\nabla_\phi
\mathcal{L}_{\mathrm{inner}}
\left(\theta,\phi_c;X_c,Y_c,M_c\right).
$$

Here \(k\) counts outer training steps and \(c\) counts complete inner-update chunks; \(X_c,Y_c,M_c\) are a chunk's inputs, targets, and mask. This equation shows plain gradient descent; the implementation also clips gradients, as the next chapter explains. Static parameters \(\theta\) include the learned MLP-prime initialization, from which <code>initial_fast_weights(params)</code> obtains \(\phi_0\). That initialization participates in outer training. The adapted values \(\phi_1,\phi_2,\ldots\) belong to a sequence or session and are not written back as global AdamW parameters. Learning the initialization and storing the adapted session value in the optimizer are different operations.

## Prefix prediction, label use, and causality

A safe causal update has this order:

~~~text
use the current fast state to predict a chunk's targets
→ wait until all its labels are observed, then compute inner loss
→ commit the new fast state
→ use it for the next chunk
~~~

The label used for an inner update must be available at the point where the update happens. If a suffix label is used to update the state before the same position is predicted, the model receives future information and the measured loss is too optimistic.

The test is simple to state: change a later token and compare logits at every earlier position. Earlier logits must remain unchanged up to numerical tolerance. Earlier/later refers to time positions; prefix/suffix above refers to layers. These are different axes. This test catches semantic leakage that tensor-shape tests cannot see.

The inner step uses <code>jax.value_and_grad</code> with respect to fast weights, then applies <code>sgd_update</code>. The outer differentiation through <code>sequence_loss</code> can follow that update and introduce second derivatives. <code>jax.checkpoint</code> recomputes values to save memory; it does not automatically truncate these gradients. A first-order approximation requires an explicit choice of which derivative paths to stop. Rematerialization is not itself such an approximation.

## A small chunk calculation

Let the sequence length be \(8\) and the chunk size be \(4\). The numbers \(0\) through \(7\) below are code array indices; in the book's mathematical notation they correspond to positions \(1\) through \(8\):

~~~text
chunk 0: positions 0, 1, 2, 3
chunk 1: positions 4, 5, 6, 7
~~~

Chunk 0 predicts with \(\phi_0\), obtains \(\mathcal{L}_0\), and updates to \(\phi_1\). Chunk 1 predicts with \(\phi_1\) and obtains \(\mathcal{L}_1\). Now consider a separate sequence with six target positions: its final chunk has two real targets and two padding positions. Real targets still contribute to loss; padding weights are zero. Because the chunk is incomplete, the current implementation retains the whole previous fast state \(\phi_1\), without committing an update from the two real targets either.

This is why chunk size is an algorithmic variable, not merely a batching detail. A smaller chunk lets the fast state observe context more often but increases scan and update overhead. It also changes the boundary through which gradients and caches are carried.

## How to validate that the forward pass is semantically correct

A useful validation set contains at least four checks:

1. Check the dense model without fast MLP and C with the module present but updates disabled separately; they differ in architecture, so their logits need not agree.
2. Change a later token and verify that every earlier logit remains unchanged.
3. Compare full-sequence execution with prefill/decode streaming under the same initial state and masks.
4. When changing chunk size, record whether it is an intended algorithm change or only an implementation optimization.

These are validation goals, not a claim that one script implements them all. The next section lists exactly what the GPU script executes and records; other comparisons require separate evidence.

## Shape walk for one token

Use the tiny configuration with \(d=16\), \(V=259\), batch \(B=1\), and \(L=4\) positions:

| Stage | Teaching object | Batched shape | Check |
|---|---|---|---|
| Input | <code>input_ids</code> | \([1,4]\) | every integer is in \([0,259)\) |
| Embedding | hidden | \([1,4,16]\) | token IDs become residual vectors |
| Attention/delta | residual | \([1,4,16]\) | only permitted history is read |
| MLP-prime | fast branch | \([1,4,16]\) | intermediate width returns to 16 |
| Output | logits | \([1,4,259]\) | every position has 259 candidate scores |

This table describes the conceptual batched layout. The current model's <code>forward</code> takes single-sequence <code>input_ids</code> of shape \([L]\), has hidden states \([L,d]\), and returns logits \([L,V]\). Training adds the batch axis outside this function with <code>jax.vmap</code>; the table's \([B,L]\) array is not a direct input to the single-sequence interface.

## A numerical causality assertion

For a sequence with four input positions, run twice with identical initial parameters and mask, changing only input index 3 from 18 to 19. Indices 0, 1, and 2 all precede that change, so logits at all three must agree. For single-sequence logits of shape <code>[L,V]</code>, the conceptual assertion is:

~~~text
max_abs_diff(logits_a[0:3, :], logits_b[0:3, :]) <= tolerance
max_abs_diff(logits_a[3:, :], logits_b[3:, :]) may be > 0
~~~

Here <code>max_abs_diff</code> means the maximum elementwise absolute difference; it is explanatory notation, not a repository API. Failing the first line indicates that future information influenced an earlier prediction. Changes are allowed only from index 3 onward. Declare the tolerance with dtype and comparison method rather than widening it merely to pass.

## Minimal validation

On a remote GPU node, run <code>bash scripts/validate_gpu.sh outputs/gpu-validation.json</code>. This executes the built-in tiny cases in <code>ttt/validate.py</code>; it does not read the previous chapter's C/F configuration files. Each case checks finite loss and gradients, compares <code>sequence_loss</code> with <code>TTTSession.observe</code>, and asserts causality under future-token perturbation.

The successful report has top-level <code>environment</code>, <code>passed</code>, and <code>cases</code>. Each case records <code>name</code>, <code>tokens</code>, <code>loss</code>, <code>finite_gradients</code>, <code>logits_max_error</code>, <code>compilation_seconds</code>, <code>training_seconds</code>, <code>state_bytes</code>, and <code>inner_updates</code>. The logit error is the sequence/streaming difference, not a separate causality-error field. A failed report includes <code>passed=false</code>, exception type, message, and environment.

The script does not export full configuration, masks, initial parameters, state shapes, or restore comparisons. Case settings and seeds live in its source, so preserve the source revision. It requires an available GPU but does not check a particular model or multi-GPU count. Recovery is covered in Chapter 3. A tiny pass covers only the cases actually listed; the next chapter explains the connection between inner and outer gradients.

## Summary

- Forward order determines what each residual path and state can observe.
- The fast-weight initialization participates in outer training; adapted values belong to a sequence or session.
- Prefix/suffix, chunks, and masks must use the same position coordinate.
- Causality requires a numerical future-token perturbation test.
- The next chapter expands this state transition into inner and outer gradient paths.
