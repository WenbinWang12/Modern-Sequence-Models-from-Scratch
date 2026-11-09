# 9. Packing, labels, and masks

The training loop sees input_ids, labels, and loss_mask, not documents. For tokens [x0,x1,x2,x3]:

~~~text
input_ids = [x0,x1,x2]
labels    = [x1,x2,x3]
mask      = [1,1,1]
~~~

The current TokenDataset.rows returns a window and a mask shifted by one position. A wrong shift can still produce finite loss while training the wrong task.

Independent-document packing and continuous-stream packing are different contracts. The current prepare_data and prepare_sharded_data append EOS and write one logical stream; TokenDataset.rows can sample across EOS and does not reset attention at document boundaries. We should therefore describe it as continuous-stream behavior, not claim document isolation.

A mask means whether a label contributes to loss. Prompt tokens may be visible but masked; padding may also be masked, but the reasons are different. Aggregation should weight samples by valid-token count rather than average per-sample losses equally.

The next chapter makes shards, masks, hashes, and sampler state a recoverable Dataset.
