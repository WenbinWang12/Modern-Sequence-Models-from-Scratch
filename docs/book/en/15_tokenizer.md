# 15. Tokenizer and vocabulary

## The tokenizer is a model interface

A tokenizer maps variable-length text into a finite vocabulary of IDs. It controls sequence length, character coverage, unknown handling, special-token semantics, and the dimensions of embedding and lm_head.

Vocabulary size is a cost and coverage trade-off. A larger vocabulary can shorten common text but increases embedding and output parameters; a smaller vocabulary saves parameters but may lengthen sequences.

## Versioned choices

A tokenizer version records training data and manifest, algorithm, vocabulary size, normalization, special IDs, byte fallback or unknown policy, code commit, and output digest. Special IDs cannot be silently reordered after they enter checkpoints or token shards.

~~~text
token id ∈ [0, vocab_size)
embedding: [vocab_size, hidden_size]
hidden:    [batch, length, hidden_size]
logits:    [batch, length, vocab_size]
~~~

Track tokens per byte by source, length quantiles, special-token frequency, fallback rate, and truncation. The next chapter builds fixed-shape sequences, labels, and loss masks.
