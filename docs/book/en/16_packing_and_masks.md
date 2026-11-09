# 10. Packing, labels, and masks

The training loop sees code fields `input_ids`, `labels`, and `loss_mask`, not documents. For tokens \(x_0,x_1,x_2,x_3\):

$$
X=(x_0,x_1,x_2),\qquad
Y=(x_1,x_2,x_3),\qquad
M=(1,1,1).
$$

The current TokenDataset.rows returns a window and a mask shifted by one position. A wrong shift can still produce finite loss while training the wrong task.

Independent-document packing and continuous-stream packing are different contracts. The current prepare_data and prepare_sharded_data append EOS and write one logical stream; TokenDataset.rows can sample across EOS and does not reset attention at document boundaries. We should therefore describe it as continuous-stream behavior, not claim document isolation.

A mask means whether a label contributes to loss. Prompt tokens may be visible but masked; padding may also be masked, but the reasons are different. Aggregation should weight samples by valid-token count rather than average per-sample losses equally.

The next chapter makes shards, masks, hashes, and sampler state a recoverable Dataset.
