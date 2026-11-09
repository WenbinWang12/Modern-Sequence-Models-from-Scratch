# 3. Pretraining data: distribution, quality, and reproducibility

## What does the model actually learn?

A language model does not optimize an abstract store of knowledge. It optimizes prediction loss over a data distribution. If tokenized sequences are sampled from `D`, then:

$$
\mathcal{L}(\theta;D)
=
\mathbb{E}_{x\sim D}
\left[
-\sum_t \log p_\theta(x_t\mid x_{<t})
\right].
$$

Changing `D` changes the objective even when the architecture, optimizer, and number of steps stay fixed. The model sees different words, code styles, document lengths, repetitions, and boundaries.

That is why data is part of the model definition:

~~~text
model = architecture + tokenizer + training algorithm + data distribution
~~~

Publishing the first three without a data version does not tell readers what the model was trained to learn.

## Why data changes capability, cost, and conclusions

**Coverage.** Web text, books, code, papers, and conversations provide different vocabularies and structures. A code-heavy mixture may improve code completion while changing ordinary-language statistics. The right question is whether the distribution matches the intended use.

**Quality.** Templates, navigation pages, corrupted text, and parser failures consume tokens without necessarily providing useful signal. Over-filtering can remove code, mathematics, minority languages, or long documents. Quality is a measurable trade-off, not a universal binary label.

**Repetition and contamination.** Repeated documents receive more optimization weight. Near duplicates across training and validation make held-out loss optimistic. Benchmark contamination can turn a high score into memorization evidence rather than generalization evidence.

**Tokenizer and cost.** Different tokenizers produce different token counts for the same characters. This changes context coverage, embedding and output-head size, attention and cache memory, training budget, and throughput comparisons.

## A concrete mixture example

| Source | Documents | Average length | Total tokens |
|---|---:|---:|---:|
| Web | 900,000 | 1,000 | 900M |
| Code | 50,000 | 2,000 | 100M |
| Books | 10,000 | 10,000 | 100M |

Concatenating files makes web text about 82% of the token stream. Sampling documents uniformly produces another distribution. Both can be valid, but they are different training objectives.

If 30% of web tokens are boilerplate duplicates, cleaning changes the denominator and automatically changes the effective weights of code and books. Every stage must report its own counts.

The current BatchStream samples windows from one logical token stream. It does not implement source-level mixture scheduling. The later mixture chapter defines that design without claiming it already exists.

## Five requirements for trainable data

A data pipeline should answer:

1. Is the source and release identifiable, with terms and retrieval evidence?
2. Are quality rules versioned, measurable, and auditable?
3. Are train, validation, and test boundaries protected from related documents and contamination?
4. Do tokenizer, special IDs, packing, label shift, and masks agree with the model?
5. Can a manifest, shard hash, sampler state, and checkpoint reconstruct the next batch?

These questions define the order of the data part and the preflight gate before a long run.

## Four data objects

**Raw files** are immutable downloads or snapshots.

**Document records** contain text and metadata used for cleaning, deduplication, and splitting.

**Token shards** contain integer IDs and supervision masks plus tokenizer and SHA256 metadata.

**Training windows** are sampled fixed-length sequences that become input IDs, labels, masks, and positions.

~~~mermaid
flowchart LR
    A[Raw source] --> B[Document records]
    B --> C[Clean and deduplicate]
    C --> D[Train/validation split]
    D --> E[Tokenizer]
    E --> F[Token shards + manifest]
    F --> G[Sample windows]
    G --> H[Batch, labels, masks]
    H --> I[Model training]
~~~

These objects have separate versions. Cleaning can change the document version while raw files remain fixed; a tokenizer change creates a new token version; a sequence-length change creates a new window protocol.

## What the repository currently implements

ttt/data.py provides a byte tokenizer and JSONL conversion. It can write one-dimensional NPY data or sharded token and mask files with a manifest.

ttt/dataset.py validates manifest version, vocabulary size, paths, hashes, token ranges, mask shapes, and mask values, then reads shards with memory mapping. BatchStream samples random windows and stores RNG state, data digest, sequence length, batch size, and seed.

~~~bash
.venv/bin/python -m ttt.data \
  --input examples/book/clean_corpus.jsonl \
  --output outputs/book-data \
  --sharded \
  --tokenizer byte \
  --shard-tokens 256
~~~

Shard size is a storage choice, not model sequence length. The current window can cross EOS and document boundaries, and there is no general attention reset at EOS. The loader should therefore be described as continuous-stream packing.

The manifest also lacks a complete document-offset sidecar, so arbitrary token-to-document tracing remains a requirement for a larger data pipeline. Keeping implemented behavior separate from design requirements is part of the book's data standard.

## The data route ahead

Chapter 4 records source versions, licensing, and acquisition losses. Chapter 5 cleans documents. Chapters 6 and 7 cover deduplication, splits, and contamination. Chapters 8 and 9 fix tokenizer and packing semantics. Chapters 10 through 12 connect shards, mixtures, sampler state, and preflight. Chapter 14 then uses the dense causal baseline to test whether these inputs support learning.

From the next chapter onward, “data” is no longer one vague input. We trace what each artifact is, how it changes, and how it is checked.
