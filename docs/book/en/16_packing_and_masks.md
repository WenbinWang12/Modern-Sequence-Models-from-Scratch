# 10. Packing, labels, and masks

The training loop sees code fields `input_ids`, `labels`, and `loss_mask`, not documents. For tokens \(x_1,x_2,x_3,x_4\):

Use readable symbols first. For the sequence `[A, B, C, D]`, the input is `[A, B, C]` and the labels are the one-position-left shift `[B, C, D]`. After seeing `A`, the model predicts `B`; after seeing `A,B`, it predicts `C`; after seeing `A,B,C`, it predicts `D`. This is next-token prediction, not copying the current token.

$$
X=(x_1,x_2,x_3),\qquad
Y=(x_2,x_3,x_4),\qquad
M=(1,1,1).
$$

The current TokenDataset.rows returns a window and a mask shifted by one position. A wrong shift can still produce finite loss while training the wrong task.

If `D` is padding, its target mask should be 0. If `B` belongs to a prompt, its mask may also be 0 while the prompt remains visible as context. A mask controls whether a label contributes to loss; it does not necessarily remove the token from the input.

Independent-document packing and continuous-stream packing are different contracts. The current prepare_data and prepare_sharded_data append EOS and write one logical stream; TokenDataset.rows can sample across EOS and does not reset attention at document boundaries. We should therefore describe it as continuous-stream behavior, not claim document isolation.

A mask means whether a label contributes to loss. Prompt tokens may be visible but masked; padding may also be masked, but the reasons are different. Aggregation should weight samples by valid-token count rather than average per-sample losses equally.

The next chapter makes shards, masks, hashes, and sampler state a recoverable Dataset.
