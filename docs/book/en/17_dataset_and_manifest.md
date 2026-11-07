# 17. Dataset, Shards, and Manifests

The loader needs shard paths, counts, dtype, vocabulary range, hashes, and a reproducible sampling state.

`ttt/dataset.py` provides mmap-backed shards, SHA256 checks, and supervised masks. `ttt/data.py` provides tokenization and batch sampling. `ttt/checkpoint.py` stores data fingerprints and sampling state for recovery.

Exercise: modify one shard byte and confirm that the loader rejects the manifest before training begins.
