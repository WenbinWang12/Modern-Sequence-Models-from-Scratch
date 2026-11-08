# 13. Exact and near deduplication

## Why repetition matters

Repeated documents increase the weight of one source and can leak information between training and evaluation. Exact copies and near copies are different problems and should be reported separately.

Exact deduplication hashes normalized documents and retains one representative. Near deduplication uses shingles and an approximate fingerprint such as MinHash or SimHash, followed by a thresholded candidate review.

Record normalization, shingle size, threshold, kept documents, removed documents, and cluster counts. The rules and code version are part of the data version.

## Evaluation boundaries

Compare training documents with public evaluation data and contamination lists. Report exact and near matches even when sources differ. Also inspect source, language, domain, and length retention so a threshold does not silently remove a small but important group.

The next chapter creates train, validation, and test boundaries without splitting near-duplicate documents across them.
