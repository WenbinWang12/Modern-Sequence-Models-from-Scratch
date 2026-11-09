# 10. Datasets, shards, and manifests

[中文版](../17_dataset_and_manifest.md)

A directory of NPY files is not a data version. The manifest fixes shard order, counts, vocabulary, mask files, and hashes. The current loader checks format_version, vocab_size, file location, SHA256, one-dimensional integer tokens, token ranges, mask shape, and mask bounds.

Per-shard hashes detect replacement; the manifest hash detects reordering or changed counts; the source digest explains where the derived tokens came from. All three are useful.

TokenDataset maps global offsets to physical shards with memory mapping. A window can cross shards because shards are storage boundaries, not necessarily document boundaries. BatchStream separately records RNG state, data digest, sequence length, batch size, and seed. A same seed without the generator state is not a strict resume.

The current preparation command assigns a UUID version and refuses to overwrite a manifest. That protects old data, but byte-identical reruns would require content-addressed versioning rather than random IDs.

The next chapter defines source mixtures; the current sampler is a single token stream and does not yet implement source-level mixture scheduling.
