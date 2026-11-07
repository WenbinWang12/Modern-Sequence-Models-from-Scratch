# 10. Data Contract

Training data is an input protocol, not merely a file path. Before cleaning, define the schema, provenance, license, stable IDs, and version identifiers.

```json
{"id":"doc-0001","text":"...","source":"dataset-a","license":"CC-BY"}
```

Every processed dataset should produce a cleaned file, a manifest, and a statistics report. The manifest records paths, counts, token counts, hashes, and the processing version. Raw corpora stay outside the Git repository.

Exercise: add stable IDs, source, and license fields to `examples/retrieval.jsonl`, then reject duplicate IDs and empty text.
