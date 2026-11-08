# 14. Splits and contamination

## What a validation set means

A validation set should represent data that did not participate in optimization. Randomly taking lines from a corpus is insufficient when versions, authors, web mirrors, or similar documents are related.

Choose a unit that matches the corpus: document, document cluster, time window, or token shard. Document or cluster splitting usually happens before tokenization and packing so one source cannot cross boundaries through chunking.

## The report

Record document and token counts, source and language distributions, exact and near matches between splits, matches to public benchmarks, split seed and rule, input manifest, and code commit. Use pass, review, and blocked states; a review result is not a pass.

Freeze the benchmark version and date. If evaluation data was used for tokenizer training or tuning, declare it. The next chapter fixes how text becomes token IDs.
