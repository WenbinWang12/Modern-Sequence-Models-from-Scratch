# 18. Data mixtures and deterministic sampling

## Mixtures shape the model

File size should not decide the training distribution. A large web source can drown code, books, science, or minority languages. Define source weights in effective token budget and record temperature, seed, exhaustion policy, and target tokens.

Record target and observed proportions. Filtering and source exhaustion can make them differ. A changed mixture creates a new data version even when the tokenizer and model are unchanged.

Before training, sample indices without updating the model and check source proportions and repeats. During training, log cumulative source counts; after restore, continue from the same sampler state.

The final data chapter performs a preflight before a long job starts.
