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

The current loader requests a window of <code>length + 1</code> tokens. <code>TokenDataset.rows(starts, length)</code> returns the window and a mask with its first column removed, so that the mask has the same positions as the shifted labels. <code>BatchStream.next_batch_with_masks()</code> passes those arrays to the training code. When auditing a new data path, inspect this shift before inspecting the model.

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

The masked token loss is commonly written as

$$
\mathcal{L} = \frac{\sum_{i=1}^{L}m_i\,\ell_i}{\max(1,\sum_{i=1}^{L}m_i)},
$$

where $m_i\in[0,1]$ and $\ell_i$ is the cross-entropy for target $y_i$. The <code>max</code> avoids division by zero in defensive code; a sample with no valid targets should still be rejected by preflight. Across samples or devices, weight by the number of valid tokens. Averaging one loss per sequence would give a short response and a long response the same weight.

The implementation replaces masked target IDs with a safe in-range ID before indexing logits, then multiplies their losses by zero. The safe ID prevents an indexing error; it is not a training target.

## Independent documents and continuous streams

There are two legitimate packing contracts:

- **Independent-document packing** prevents attention and carried model state from reading earlier documents, and handles next-token targets at boundaries explicitly. A zero loss mask alone cannot provide this isolation: it removes a loss term but does not erase context. Such packing needs document-aware attention or separate calls with state reset.
- **Continuous-stream packing** concatenates documents into one stream. It uses storage efficiently and may deliberately learn statistics around EOS, but it must state whether cross-document attention and loss are allowed.

The current <code>prepare_data</code> and <code>prepare_sharded_data</code> functions encode each document, append EOS when the tokenizer provides one, and write tokens in order. <code>TokenDataset.rows</code> selects a window by global offset. A window can therefore cross EOS and a physical shard boundary. The current model path does not create a block attention mask that resets at EOS. Describe it as continuous-stream behavior unless a future implementation adds that boundary state.

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
4. <code>ttt.inner_loop.sequence_loss</code> shifts targets, applies the mask, and aggregates valid-token loss.
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
