# 13. Exact and Approximate Deduplication

Deduplication saves training tokens and reduces evaluation leakage. Run it before splitting the corpus.

Exact deduplication hashes normalized documents or paragraphs. Approximate deduplication can use n-gram signatures, MinHash, or SimHash to catch lightly modified copies and web templates.

Record the algorithm version, thresholds, removed counts, token savings, and per-source removal rates. “Deduplicated” without those records is not reproducible.
