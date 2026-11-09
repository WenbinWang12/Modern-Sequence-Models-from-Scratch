# 5. Data acquisition, licensing, and provenance

## A download URL is not an identity

The same URL can point to a changed release. A source record should separate canonical location, immutable release or snapshot, retrieval time, terms, and file digests.

~~~json
{
  "source_id": "source-a",
  "publisher": "<publisher>",
  "source_url": "<canonical-page>",
  "release": "<immutable-release>",
  "retrieved_at": "<UTC-time>",
  "files": [
    {"name": "part-000.jsonl.gz", "bytes": 12345, "sha256": "<computed>"}
  ]
}
~~~

This is a provenance record, not the input schema consumed by TokenDataset. The training manifest points to derived token files; the source record explains how those files came to exist.

## Integrity and extraction

A correct archive hash does not prove that HTML extraction produced the intended text. Extractor versions, retained headings, tables, and code blocks can change the model distribution. Keep raw objects immutable and write clean outputs to a new versioned directory.

Download failure, parse failure, and content filtering are different losses and should not share one “removed” counter.

## Licensing boundaries

Review the license, personal information, third-party content, benchmark contamination, and intended redistribution separately. Unknown terms remain unknown until reviewed. A license field is evidence of a declared term, not proof that the full release plan is compliant.

The repository's sample corpus is an educational input. ttt.data does not implement web crawling, legal review, or general HTML extraction. Those stages belong upstream of tokenization.

The next chapter defines conservative cleaning rules and the statistics that make their effects visible.
