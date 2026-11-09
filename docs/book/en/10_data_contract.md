# 3. Training data: from raw documents to model inputs

[中文版](../10_data_contract.md)

## Start by reversing the training command

A training command may have one data path, but that path hides source versions, document boundaries, duplicates, tokenizer identity, masks, and sampler state. This chapter defines the data objects that later chapters refine.

We distinguish raw files, document records, token shards, and training windows. They have different identities and different evidence.

| Object | Main question | Evidence |
|---|---|---|
| Raw file | What did we obtain? | source release and digest |
| Document | Why was it kept? | ID, metadata, filtering reason |
| Token shard | Which tokenizer produced it? | tokenizer version and hash |
| Window | What will the model consume next? | seed, sampler state, data digest |

A changed cleaning rule creates a new document version. A changed tokenizer creates a new token version. A changed sequence length creates a new window protocol.

## The document record

~~~json
{
  "id": "lesson-001",
  "text": "A document used in the corpus.",
  "source": "book-original",
  "language": "en",
  "split_group": "lesson-001"
}
~~~

IDs locate records; source and language support statistics; split_group keeps related versions together. An ID is not the same as a content hash: one source can change content, and multiple URLs can contain the same text.

The current prepare_sharded_data path does not emit a complete document-offset sidecar. Therefore arbitrary token-to-document tracing is a design requirement for the large-corpus pipeline, not a capability we claim the current manifest already provides.

## Counts must stay separate

Document count, character count, encoded token count, effective supervised token count, and consumed training tokens answer different questions. A corpus with many short documents can dominate document count while contributing fewer tokens than a small collection of long documents.

The current loader samples random windows from a logical token stream. It does not implement an epoch over documents. That sampling choice must be recorded, especially when comparing source proportions.

## Repository path

ttt/data.py creates a single token array or sharded token and mask files. ttt/dataset.py validates hashes, ranges, shapes, and manifest vocabulary size. BatchStream samples windows and saves RNG state.

~~~bash
.venv/bin/python -m ttt.data \
  --input examples/book/clean_corpus.jsonl \
  --output outputs/book-data \
  --sharded \
  --tokenizer byte \
  --shard-tokens 256
~~~

The shard size controls storage, not model sequence length. A manifest records the fields the loader actually reads; an idealized schema must not be presented as a compatible implementation.

## Transition

The next chapter records how raw sources are obtained and governed. Only after provenance is clear can cleaning statistics be interpreted.
