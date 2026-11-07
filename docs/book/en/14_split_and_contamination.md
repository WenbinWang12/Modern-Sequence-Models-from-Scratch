# 14. Splits and Contamination

Random splits are not always safe. The same document, author, time period, or benchmark can cross train, validation, and test boundaries.

Choose document-level, source-level, temporal, or domain-level splits. Deduplicate before splitting, then run overlap checks between all partitions.

The contamination report should record matching rules, thresholds, matched examples, and removed samples. Evaluation contamination requires a separate audit; validation loss cannot discover it by itself.
