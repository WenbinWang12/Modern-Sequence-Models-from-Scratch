# 19. Data preflight

## Why preflight exists

A vocabulary mismatch, stale manifest, or empty validation split discovered after compilation is expensive. Preflight accepts or rejects the complete data input without training the model.

## Required checks

Verify that manifest, tokenizer, and model share vocab_size; all shards exist and match their hashes; token IDs are in range; sequence length, EOS, packing, and masks agree; train/validation and contamination reports exist; the mixture can satisfy its token budget; seed and resume semantics are defined; and the output directory is safe to use.

Return pass, warning, or fail for every item, with counts, digests, and reasons. Save the JSON report beside experiment.json. A changed input digest must make checkpoint resume refuse to continue.

The next chapter establishes the dense causal-language-model baseline.
