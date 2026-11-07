# 10. Training Data: Sources, Schema, and Versions

## Start with what the model should learn

The first step in training a base model is not creating a Transformer. It is deciding which data distribution the model will learn from.

Web-heavy data, books, code, and scientific text produce different token frequencies, styles, lengths, and knowledge distributions. The training corpus is therefore part of the model definition, not just an argument passed to the training command.

The chapter follows one chain of questions:

```text
Where did the data come from?
    ↓
Are we allowed to use it?
    ↓
What should one record look like?
    ↓
Which processed version produced these tokens?
    ↓
Can the same training samples be reproduced later?
```

Cleaning, deduplication, splitting, and tokenization all depend on this chain.

## Three layers of a training corpus

Downloaded files should not be treated as training data directly:

| Layer | Content | Used for training |
|---|---|---|
| Raw corpus | Original files from public, web, or internal sources | No |
| Clean documents | Documents after format, character, language, length, and duplicate checks | Usually no |
| Training data | Token shards, masks, and manifests | Yes |

Raw files should be immutable. Cleaning and tokenization are derived processes; a changed rule creates a new version instead of overwriting the old one.

## The minimum document record

```json
{
  "id": "doc-0001",
  "text": "A raw text document.",
  "source": "dataset-a",
  "license": "CC-BY-4.0",
  "language": "en"
}
```

`text` is what the model eventually sees. The other fields support tracing and filtering:

- `id` must be stable and unique for deduplication and debugging;
- `source` must identify a concrete source;
- `license` keeps redistribution boundaries explicit;
- language, time, and domain fields support recipes and statistics;
- raw files, cleaned files, and token shards need SHA256 hashes.

The loader reads token shards, but a problematic token sequence should still be traceable back to a cleaned document and its original source.

## Provenance and versions

The repository does not commit the corpus itself. It should commit source records such as:

```json
{
  "source": "dataset-a",
  "url": "https://example.org/data",
  "version": "2026-01",
  "license": "CC-BY-4.0",
  "downloaded_at": "2026-11-07T00:00:00Z",
  "raw_sha256": "..."
}
```

This tells future readers where the data came from and whether a later corpus update still supports a fair comparison.

Licensing, personal information, benchmark contamination, and third-party content require separate review. A tokenizer or training loop cannot solve those problems automatically.

## From document version to training version

```text
raw source version
→ cleaning configuration
→ deduplication configuration
→ split configuration
→ tokenizer version
→ token shard manifest
```

The final `manifest.json` should record the format version, vocabulary size, shard paths, counts, and hashes. A checkpoint also stores data fingerprints, sampling seed, sequence length, and batch size so that recovery can reject incompatible inputs.

## The first data path in this book

We start with a small but complete path:

```text
examples/sample.txt
→ byte tokenizer
→ outputs/sample-tokens.npy
→ tiny baseline
→ checkpoint and resume
```

This first path verifies token ranges, sampling, and recovery. Later chapters extend it to JSONL, cleaned documents, sharded manifests, and a production tokenizer.

The next chapter defines which documents are retained or filtered and how every filtering decision becomes part of the statistics report.
