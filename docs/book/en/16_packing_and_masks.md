# 16. Packing, labels, and masks

## From documents to batches

Training needs fixed shapes while documents do not have fixed lengths. Packing decides truncation, concatenation, padding, EOS boundaries, labels, and valid loss positions.

~~~text
input_ids: [x0, x1, ..., xL-1]
labels:    [x1, x2, ..., xL]
loss_mask: [ 1,  1, ...,  1]
~~~

Independent-document packing can mask cross-document supervision. Continuous-stream packing improves utilization but must define EOS, source boundaries, and evaluation separation. Sequence length, truncation, padding, EOS, and mask rules belong in the manifest.

## Checks

Every batch has matching input and label shapes, in-range token IDs, a valid mask, nonzero effective tokens, and an explicit next-token convention. Print a sample with text, tokens, labels, masks, and boundaries before a long run.

The next chapter makes shard indexing and sampler state reproducible.
