# 12. Data preflight

Preflight rejects an incoherent input before a long compile or multi-day run. It joins the data, tokenizer, model configuration, sampler, and output directory into one check.

Verify vocabulary and special-token alignment; shard existence, hashes, counts, shapes, and ranges; sequence length and mask validity; split and contamination reports; mixture token capacity; seed, rank, and resume semantics; and output-directory safety.

Return pass, warning, or fail for each item with inputs, observations, and a reason. A changed data digest must make checkpoint resume refuse to continue. The current repository exposes these checks through the data loader and tiny run; a dedicated ttt.preflight command remains a useful next implementation.

A passed preflight proves input consistency, not model quality. The next chapter establishes the dense baseline.
