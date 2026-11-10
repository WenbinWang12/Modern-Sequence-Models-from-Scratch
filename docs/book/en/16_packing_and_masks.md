# 10. Packing, labels, and masks

## The model does not receive a document

The model receives integer arrays. We use teaching names <code>input_ids</code>, <code>labels</code>, and <code>loss_mask</code> for the mathematical objects \(X,Y,M\). The actual repository loader returns <code>(tokens, masks)</code>; <code>sequence_loss</code> shifts tokens into inputs and targets. It does not receive a dictionary with those three teaching names.

### A four-token example

Use letters before using large integer IDs. Suppose a tokenizer produced <code>[A, B, C, D]</code>. A next-token example is

$$
X=(A,B,C), \qquad Y=(B,C,D).
$$

The first row asks “after <code>A</code>, what comes next?”; the second asks “after <code>A,B</code>, what comes next?”; the third asks “after <code>A,B,C</code>, what comes next?”. With the book notation, for a token sequence $(s_1,s_2,s_3,s_4)$,

$$
X=(s_1,s_2,s_3),\qquad Y=(s_2,s_3,s_4).
$$

The target at position $i$ is therefore $y_i=s_{i+1}$. If we accidentally set $Y=X$, we train copying rather than next-token prediction. If we shift twice, the loss is still a number, but it describes a different prediction problem.

For a model sequence length of <code>sequence_length</code>, <code>BatchStream</code> requests <code>sequence_length + 1</code> raw tokens. The argument <code>length</code> in <code>TokenDataset.rows(starts, length)</code> already means that raw count: it returns <code>length</code> tokens and <code>length - 1</code> target weights. Thus <code>rows(starts, 5)</code> supplies four prediction positions, not five. <code>BatchStream.next_batch_with_masks()</code> passes those arrays to the training code. Inspect this shift before inspecting the model.

## What a mask means

A mask answers one narrow question: should this target contribute to the loss? It does not automatically answer whether the token is visible as context.

For example, a prompt can be visible to the model while its targets are masked out. In supervised training, the model sees the prompt, but we may only charge the loss for the response. Padding can also have mask value zero, because it is an alignment placeholder rather than a real target.

Consider the integer stream

$$
s=(10,11,12,2,20),
$$

where <code>2</code> is EOS and the other IDs are invented for illustration. A length-four window has \(X=(10,11,12,2)\) and \(Y=(11,12,2,20)\). If 20 begins another document, a policy masking the boundary target could use

$$
M=(1,1,1,0),
$$

so that the final cross-document target is ignored. A continuous-stream policy could use $M=(1,1,1,1)$. The two policies estimate different conditional distributions. Adding EOS alone does not tell us which policy is in use; the input, labels, loss mask, and attention boundary must be documented together.

### From position losses to a weighted mean

If the model assigns probability \(p_i\) to the correct target, the position's cross-entropy is \(\ell_i=-\log p_i\). Giving the correct answer more probability reduces the penalty. Probabilities \(1/2\) and \(1/4\), for example, produce losses of approximately \(0.693\) and \(1.386\). We use natural logarithms, so the unit is the nat. Loss is not accuracy: confidence matters even when two predictions have the same correct/incorrect outcome.

Generated masks usually use \(m_i\in\{0,1\}\). The loader also accepts fractional values in \([0,1]\), interpreted as relative target weights. Define total weight \(W=\sum_{i=1}^{L}m_i\). The weighted mean is

$$
\mathcal L=
\begin{cases}
\displaystyle\frac{\sum_{i=1}^{L}m_i\ell_i}{W},&W>0,\\[6pt]
0,&W=0.
\end{cases}
$$

For losses \(2,4\) and weights \(1/4,1/2\), the weighted sum is \(2.5\), total weight is \(0.75\), and the mean is \(2.5/0.75=10/3\). Doubling both weights to \(1/2,1\) leaves the mean unchanged. Scaling all weights together should not change a mean; relative weights determine the result. With fractional weights, their sum is **total supervision weight**, not a literal token count.

The implementation selects a safe denominator of 1 only when \(W=0\). Using \(\max(W,1)\) would incorrectly replace \(0.75\) by 1 and return 2.5 in the example. An all-zero mask has no supervision and returns zero language-model NLL by convention. Data checks should still detect that window; zero does not indicate perfect prediction. This statement concerns NLL and does not automatically constrain auxiliary terms, such as those used by MoE models. Momentum or weight decay can update parameters even when the NLL gradient is zero, so an all-zero mask does not imply that the entire training step leaves parameters unchanged.

Across samples, reconstruct each weighted sum before dividing by the batch's total weight. A sample with 100 valid targets and mean loss 2, combined with two valid targets at mean loss 10, gives \((100\times2+2\times10)/102\approx2.157\), not \((2+10)/2=6\). Across devices, aggregate numerators and denominators under the same principle; a device with almost no supervision should not receive the same weight as one with many targets. `tests/test_weighted_loss.py` adds checks for fractional weights, outer gradients, zero weights, and sparse supervision across two devices; they have not been executed in this review.

The implementation replaces masked target IDs with a safe in-range ID before indexing logits, then selects zero for their loss. The safe ID prevents an indexing error; it is not a training target. This explanation assumes finite model logits: a mask does not repair numerical failures inside the model.

## Independent documents and continuous streams

There are two legitimate packing contracts:

- **Independent-document packing** prevents attention and carried model state from reading earlier documents, and handles next-token targets at boundaries explicitly. A zero loss mask alone cannot provide this isolation: it removes a loss term but does not erase context. Such packing needs document-aware attention or separate calls with state reset.
- **Continuous-stream packing** concatenates documents into one stream. It uses storage efficiently and may deliberately learn statistics around EOS, but it must state whether cross-document attention and loss are allowed.

The current <code>prepare_data</code> function writes one NPY file; <code>prepare_sharded_data</code> writes shards and a manifest. Both encode retained documents and append EOS when the tokenizer provides one. <code>TokenDataset.rows</code> selects a window by global offset, which can cross EOS and a physical shard boundary. Neither path creates a block attention mask that resets at EOS. Both retain continuous context rather than independent-document isolation.

Their supervision policies differ. A single NPY has no stored mask, so the loader assigns weight 1 to every target. The sharded path assigns zero weight to the opening BOS of each document. For two documents A and B, the byte tokenizer produces `[1,68,2,1,69,2]`, where 1 is BOS and 2 is EOS. Reading the full stream yields target weights `[1,1,1,1,1]` from the single NPY and `[1,1,0,1,1]` from the shards. The third target is the second document's BOS predicted after the first document's EOS. Masking that target still allows predictions in B to read A. The sharded path also skips whitespace-only text, whereas the single-NPY path encodes it. Switching preparation entry points can therefore change supervision and the token stream; it must not be recorded merely as a storage-format change.

## Inspect a boundary with actual byte IDs

Run this example from the repository root on the target machine. It follows the ordinary sharded path's encoding rules for two one-character documents. Constructing each array explicitly makes the shift visible.

~~~bash
.venv/bin/python - <<'PY'
from ttt.data import ByteTokenizer

tokenizer = ByteTokenizer()
tokens, stored_mask = [], []
for text in ("A", "B"):
    document = tokenizer.encode(text) + [tokenizer.eos_token_id]
    tokens.extend(document)
    stored_mask.extend([0] + [1] * (len(document) - 1))
print("tokens:", tokens)
print("X:", tokens[:-1])
print("Y:", tokens[1:])
print("M:", stored_mask[1:])
PY
~~~

ByteTokenizer maps bytes 65 and 66 to IDs 68 and 69, with BOS 1 and EOS 2. The expected stream is therefore <code>[1,68,2,1,69,2]</code>. After shifting:

| Code index | Input \(X\) | Target \(Y\) | \(M\) | Meaning |
|---|---:|---:|---:|---|
| 0 | 1 | 68 | 1 | Predict A after BOS |
| 1 | 68 | 2 | 1 | Predict EOS after A |
| 2 | 2 | 1 | 0 | Do not charge loss for the next BOS |
| 3 | 1 | 69 | 1 | Predict B after the second BOS |
| 4 | 69 | 2 | 1 | Predict EOS after B |

The third row has no direct loss, but the fourth can still attend to the earlier A. Boundary supervision and access to earlier context are two separate properties; the loss mask controls only the first.

## Code path and a remote smoke command

The relevant path is:

1. <code>ttt.dataset.prepare_sharded_data</code> writes token and mask shards plus <code>manifest.json</code>.
2. <code>TokenDataset</code> verifies the manifest and maps global offsets to shards.
3. <code>BatchStream.next_batch_with_masks</code> samples starts and returns <code>(tokens, masks)</code>.
4. <code>ttt.inner_loop.sequence_loss</code> shifts targets, applies the mask, and aggregates weighted NLL. The outer step combines samples using their supervision-weight sums, which equal valid-target counts only for binary masks.
5. <code>training.py</code> uses the resulting loss for the outer update.

On a machine with the repository and Python environment already prepared, a small data conversion is:

~~~
.venv/bin/python -m ttt.data \
  --input examples/sample.jsonl \
  --output outputs/packing-sample \
  --tokenizer byte \
  --sharded \
  --shard-tokens 256
~~~

The command is a data-preparation example, not a claim that a local workstation has the required dependencies. The resulting manifest is the input to the next chapter. Record its digest before a long run.

## Checks and limits

Check that inputs, targets, and masks each have \(L\) positions, as in the short example. A stricter masked-target check fixes every input, mask, and initial state, changing only the final masked target ID; that ID should not affect the result through the loss term. Do not generalize this to arbitrary masked tokens, which may also be inputs for later predictions. A shape check alone cannot establish document isolation.

The next chapter turns token files, masks, hashes, and sampler state into a recoverable dataset object.

## Summary and next question

Packing defines the window, label shift defines the prediction target, and the loss mask defines which targets receive supervision. EOS marks a boundary but does not by itself decide whether the following position is supervised. The next chapter stores these arrays with hashes and offsets in a recoverable dataset manifest.
