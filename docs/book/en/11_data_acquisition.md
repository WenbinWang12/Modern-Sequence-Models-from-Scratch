# 5. Acquiring data: provenance, licensing, and reproducibility

## Why data acquisition is part of model design

When people say that a model was trained on `dataset-a`, they often mean only that a download command mentioned that name. That description is not reproducible. A URL is a location, not an immutable identity: the publisher can replace the file at that location tomorrow. If two runs use the same URL at different times, their losses can differ even when the model code and random seed are identical.

Data acquisition therefore answers a concrete question: **which bytes were allowed to enter the training pipeline, and where did they come from?** A useful record separates the canonical location, an immutable release or snapshot, retrieval time, the terms that were reviewed, and a digest of every file. These fields let us distinguish a changed upstream release from a changed cleaner or a changed model.

The book uses a small, original corpus for its first commands. That choice keeps the experiment runnable while the reader learns the interface. A production corpus still needs the same records; replacing the sample text does not remove the provenance obligation.

## A small example: why a file name is not enough

Suppose a source page advertises `part-000.jsonl.gz`. On Monday it contains 10,000 documents. On Friday the publisher adds 500 documents but keeps the file name and URL. Both downloads are named `part-000.jsonl.gz`, yet the token count and the model's sampling distribution differ.

If the first run saves only the file name, we cannot tell whether a later loss change came from the architecture or from those 500 documents. If it saves the SHA256 digest, we can compare the exact bytes. The digest does not tell us whether the text is useful or legally redistributable; it tells us whether the object changed.

## Source records and file records

Keep a source-level record separate from the file-level record. One publisher release can contain many files, and one archive can contain documents from several sources. A line of prose such as “downloaded dataset-a” loses that structure.

~~~json
{
  "source_id": "source-a",
  "publisher": "<publisher>",
  "source_url": "<canonical-page>",
  "release": "<immutable-release-or-commit>",
  "retrieved_at": "<UTC-time>",
  "terms_snapshot": "<saved-terms-path-or-digest>",
  "files": [
    {
      "name": "part-000.jsonl.gz",
      "bytes": 12345,
      "sha256": "<computed-digest>"
    }
  ]
}
~~~

Here `source_id` is a stable label chosen by the project, `release` identifies the upstream version, and `retrieved_at` records when this copy entered the project. `bytes` is a cheap sanity check; `sha256` is the content identity used for later comparison. A terms snapshot can be a saved license page or its own digest, depending on the release policy.

This JSON is a provenance record. It is **not** the input schema consumed by `TokenDataset`. The training loader needs a token file or the versioned token `manifest.json` described later in this part. The provenance record should point to the derived artifact, while the manifest records the input digest that produced it.

For moving resources, prefer a publisher commit, revision, or snapshot over `latest`. An HTTP ETag may help a cache, but ETag meanings vary by service and it should not silently replace SHA256 in a reproducibility report.

## Integrity is different from successful extraction

An archive can have the expected hash and still produce the wrong training text. For example, an HTML extractor may keep headings and code blocks in one version and discard them in another. The archive is unchanged, but the model distribution changes because the extraction program changed.

Keep raw objects immutable. Write extracted and cleaned text to a new versioned directory, and record the extractor version, input digest, output digest, and counts. This makes it possible to inspect a failed parser without downloading a source that may have disappeared or changed.

Also keep failure categories separate:

| event | what it means | how to report it |
|---|---|---|
| download failure | bytes were not obtained | retry/error count |
| parse failure | bytes were obtained but could not be decoded | parser error count |
| content rejection | text was parsed and a rule rejected it | rule-specific count |

Putting all three into a single `removed` counter hides where the distribution changed. A timeout is not evidence that a document was low quality.

## Licensing and release boundaries

Review four questions independently: what terms were declared, whether personal information needs handling, whether third-party material is embedded, and which artifacts may be redistributed. A populated `license` field is evidence of a declared term; it is not proof that the entire release plan is compliant.

The raw archive, extracted text, cleaned JSONL, token shards, preprocessing code, and model weights are different artifacts. A source may permit local training while restricting redistribution of the original text. In that case a public reproduction must publish the recipe and digests, and explain how an authorized user can rebuild the same input.

The repository does not perform web crawling, legal review, general HTML extraction, or an automatic license decision. The directly runnable teaching input is `examples/sample.txt`; a JSONL corpus should be written as the replaceable placeholder `<prepared-jsonl>`, never as a file that does not exist. `ttt.data` starts after the upstream text has been selected. These boundaries are intentional: the training loader should be small, while the data project keeps the evidence needed to audit it.

## Count the acquisition pipeline

Consider a purely numerical example. A source list claims 1,000 files. We download 990, parse 980, and obtain 100,000 records; 500 records have no usable body. The next stage receives 99,500 documents. The count chain is:

$$
1000\ \text{listed files}
\rightarrow 990\ \text{downloaded}
\rightarrow 980\ \text{parsed}
\rightarrow 100000\ \text{records}
\rightarrow 99500\ \text{usable records}.
$$

Reporting only “99,500 documents” hides ten missing files and ten parsing failures. If the missing files are all from a small language or domain, the overall loss may look harmless while that domain disappears. Reports should keep the chain by source, language, and format, and should state whether the denominator is listed files, downloaded files, or parsed records.

## Mapping to this repository

For a small text file, the repository command is:

~~~bash
.venv/bin/python -m ttt.data \
  --input examples/sample.txt \
  --output outputs/sample-tokens.npy \
  --tokenizer byte
~~~

For JSONL records with a `text` field and a versioned shard manifest:

~~~bash
.venv/bin/python -m ttt.data \
  --input corpus.jsonl \
  --output data/pretrain-v1 \
  --sharded \
  --tokenizer byte
~~~

These commands encode already selected text; they do not create the source record above. Before running them on a real corpus, save the upstream release, file digests, extraction configuration, and license review beside the output directory. The resulting manifest contains the input SHA256 and token-shard digests, which connect the training artifact to that upstream record.

## Minimal run, evidence, and failure diagnosis

Start with the teaching text that actually exists in the repository:

~~~bash
.venv/bin/python -m ttt.data \
  --input examples/sample.txt \
  --output outputs/acquisition-sample.npy \
  --tokenizer byte
~~~

The expected artifact is a one-dimensional token file. It proves only that the tokenizer can read the input; it does not prove provenance, licensing, extraction quality, or data quality. To inspect a sharded manifest, run:

~~~bash
.venv/bin/python -m ttt.data \
  --input examples/sample.jsonl \
  --output outputs/acquisition-shards \
  --sharded --tokenizer byte --shard-tokens 256
~~~

On success, <code>outputs/acquisition-shards/manifest.json</code> should contain <code>vocab_size</code>, an input digest, and token/mask shard records. If the command fails, check the path and Python environment first. If the manifest has unexpected counts, return to the listed→downloaded→parsed→usable chain before changing the model.

For a real JSONL corpus, use the placeholder <code>&lt;prepared-jsonl&gt;</code> in the documentation and replace it with an actual path before running. Save the source record, file bytes and SHA256, download/parse/rejection counts, extractor version, and licensing review link. The repository does not generate these governance records automatically.

## Summary and next question

This chapter separates a URL (location), a release (version), and a digest (byte identity). It separates download failure, parse failure, and content rejection, and it treats raw files, document records, token shards, and model weights as different artifacts. The next chapter fixes an input version and measures how each cleaning rule changes its counts.

The next chapter defines conservative cleaning rules. We will keep the original text and make each rejection reason measurable instead of treating “clean” as a single hidden label.
