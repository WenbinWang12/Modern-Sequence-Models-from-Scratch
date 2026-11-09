# 7. Deduplication: control implicit sampling weights

Repeated text receives repeated optimization weight. It can also cross the train/validation boundary and make evaluation measure memorization rather than generalization. Exact and near duplicates should be reported separately.

Exact deduplication hashes a defined comparison view. That view might be normalized text, not the original HTML. The normalization must preserve meaningful code structure and be versioned.

Near deduplication builds shingles and approximate fingerprints such as MinHash or SimHash, then reviews candidates using a threshold. Similarity is not transitive, so cluster formation and representative selection need explicit rules.

A useful report includes normalization, shingle size, threshold, kept and removed counts, cluster sizes, source retention, and mappings from removed records to representatives.

Deduplication should happen before splitting related documents into train and validation. Training-to-benchmark contamination is a separate comparison and needs its own report.

The current repository does not implement a production MinHash pipeline. It provides the manifest and token interfaces that a versioned upstream deduplication stage should produce.
