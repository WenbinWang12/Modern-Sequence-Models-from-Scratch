# 17. Datasets, shards, and manifests

## The manifest is the data entry point

A directory full of shard files is not a data version. A manifest records shard paths, token counts, vocabulary, sequence length, packing rules, and hashes. Paths should be relative to a data root.

## Reading and recovery

The dataset maps logical indices to physical shards, handles short or corrupt files, and restores sampler seed, epoch, position, and shuffle algorithm. In distributed training, global batch, local batch, host count, and shard axes are part of the run identity.

A checkpoint that restores parameters but changes data order is not a strict resume. Verify hashes before training and stop on a mismatch.

## Transition

The next chapter combines several sources into a deterministic mixture instead of letting file size define the model distribution.
