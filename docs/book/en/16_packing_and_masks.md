# 10. Packing, labels, and masks

## The model does not receive a document

People say that a language model “reads a document”, but the training loop receives arrays. The important contract is usually three arrays: <code>input_ids</code>, <code>labels</code>, and <code>loss_mask</code>. Packing is the step that turns a stream of token IDs into those arrays. If this step is wrong, the model can train, produce finite numbers, and still learn the wrong task.

### A four-token example

Use letters before using large integer IDs. Suppose a tokenizer produced <code>[A, B, C, D]</code>. A next-token example is

$$
X=(A,B,C), \qquad Y=(B,C,D).
$$

The first row asks “after <code>A</code>, what comes next?”; the second asks “after <code>A,B</code>, what comes next?”; the third asks “after <code>A,B,C</code>, what comes next?”. With the book notation, for a token sequence $(x_1,x_2,x_3,x_4)$,

$$
X=(x_1,x_2,x_3),\qquad Y=(x_2,x_3,x_4).
$$

The target at position $i$ is therefore $y_i=x_{i+1}$. If we accidentally set $Y=X$, we train copying rather than next-token prediction. If we shift twice, the loss is still a number, but it describes a different prediction problem.

The current loader requests a window of <code>length + 1</code> tokens. <code>TokenDataset.rows(starts, length)</code> returns the window and a mask with its first column removed, so that the mask has the same positions as the shifted labels. <code>BatchStream.next_batch_with_masks()</code> passes those arrays to the training code. When auditing a new data path, inspect this shift before inspecting the model.

## What a mask means

A mask answers one narrow question: should this target contribute to the loss? It does not automatically answer whether the token is visible as context.

For example, a prompt can be visible to the model while its targets are masked out. In supervised training, the model sees the prompt, but we may only charge the loss for the response. Padding can also have mask value zero, because it is an alignment placeholder rather than a real target.

Consider the integer stream

$$
z=(10,11,12,2,20),
$$

where <code>2</code> is EOS. A length-four window has $X=(10,11,12,2)$ and $Y=(11,12,2,20)$. If <code>20</code> begins a different document, an isolated-document policy could use

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

- **Independent-document packing** appends EOS and prevents supervision or attention from crossing the boundary. This is easy to reason about, but it may leave some positions unused.
- **Continuous-stream packing** concatenates documents into one stream. It uses storage efficiently and may deliberately learn statistics around EOS, but it must state whether cross-document attention and loss are allowed.

The current <code>prepare_data</code> and <code>prepare_sharded_data</code> functions encode each document, append EOS when the tokenizer provides one, and write tokens in order. <code>TokenDataset.rows</code> selects a window by global offset. A window can therefore cross EOS and a physical shard boundary. The current model path does not create a block attention mask that resets at EOS. Describe it as continuous-stream behavior unless a future implementation adds that boundary state.

## Code path and a remote smoke command

The relevant path is:

1. <code>ttt.dataset.prepare_sharded_data</code> writes token and mask shards plus <code>manifest.json</code>.
2. <code>TokenDataset</code> verifies the manifest and maps global offsets to shards.
3. <code>BatchStream.next_batch_with_masks</code> samples starts and returns <code>(tokens, masks)</code>.
4. <code>ttt.inner_loop.sequence_loss</code> shifts targets, applies the mask, and aggregates valid-token loss.
5. <code>training.py</code> uses the resulting loss for the outer update.

On a machine with the repository and Python environment already prepared, a small data conversion is:

~~~
python -m ttt.data \
  --input data/example.jsonl \
  --output runs/example_shards \
  --tokenizer byte \
  --sharded \
  --shard-tokens 100000
~~~

The command is a data-preparation example, not a claim that a local workstation has the required dependencies. The resulting manifest is the input to the next chapter. Record its digest before a long run.

## Checks and limits

For a tiny hand-check, print one sampled window and verify four facts: the labels are one position ahead, the mask length equals the label length, masked positions have no contribution, and a window crossing a shard boundary preserves token order. A passing shape check does not prove that document boundaries are semantically correct. That question belongs to the packing policy and its evidence record.

The next chapter turns token files, masks, hashes, and sampler state into a recoverable dataset object.
