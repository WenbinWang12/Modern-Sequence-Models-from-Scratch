# 8. Tokenizer and vocabulary

A tokenizer determines sequence length, character coverage, special-token semantics, embedding size, output-head size, and evaluation fairness. The same character corpus can produce different numbers of training positions under different tokenizers.

With vocabulary V and hidden size d:

$$
E\in\mathbb R^{V\times d},\qquad
\mathrm{logits}\in\mathbb R^{\ldots\times V}.
$$

Tied embeddings use E-transpose for lm_head; untied embeddings add another d by V matrix. The MLP can change its intermediate width, but it must return to d before the residual and vocabulary projection.

The repository ByteTokenizer maps UTF-8 bytes 0–255 to IDs 3–258, reserving 0, 1, and 2 for pad, BOS, and EOS. It is useful for interface and recovery checks, but inefficient for Chinese and mixed technical text.

A tokenizer version records training manifest, normalization, algorithm, vocabulary size, special IDs, unknown or byte fallback behavior, code version, and output digest. Measure tokens per byte, length quantiles, fallback rate, special-token frequency, truncation, and per-language behavior.

Changing the tokenizer requires new token shards and a compatible model configuration; changing only vocab_size in a checkpoint is not a conversion.
