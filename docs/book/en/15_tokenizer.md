# 15. Tokenizer and Vocabulary

The tokenizer maps text to token IDs. `vocab_size` fixes the vocabulary dimension of the input embedding and the language-model head:

```text
embed_tokens: [vocab_size, hidden_size]
lm_head:      [hidden_size, vocab_size]
```

Fix the tokenizer files and version, special-token IDs, document-boundary policy, Unicode policy, training corpus, and random seed. Changing the tokenizer changes the embedding and output-head shapes, so old checkpoints cannot be reused without an explicit conversion.

Exercise: compare token-length distributions for the byte tokenizer and the target tokenizer on the same cleaned corpus.
