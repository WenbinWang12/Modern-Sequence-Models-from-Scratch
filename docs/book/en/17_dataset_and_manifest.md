# 11. Datasets, shards, and manifests

## Why a directory is not a data version

Suppose a directory contains <code>part-0.npy</code>, <code>part-1.npy</code>, and a temporary file. A script that reads “all NPY files” has hidden assumptions about sorting, ignored files, vocabulary size, and masks. Copying the directory to another machine can silently change the token stream. A manifest makes those assumptions explicit and gives the loader something to verify.

In this book, a **shard** is a storage fragment, while a **dataset** is the logical ordered stream plus the rules that make it readable. A shard boundary is not automatically a document boundary.

## A minimal manifest

The current loader accepts a version-one manifest with the following essential shape:

~~~
{
  "format_version": 1,
  "vocab_size": 259,
  "shards": [
    {
      "tokens": "v-00000.npy",
      "mask": "v-00000.mask.npy",
      "count": 256,
      "sha256": "<token-file-sha256>",
      "mask_sha256": "<mask-file-sha256>"
    }
  ]
}
~~~

<code>tokens</code> and <code>mask</code> are filenames relative to the manifest directory. The loader rejects paths that escape that directory. This is a reproducibility and safety rule: an old manifest should not silently read a file from an unrelated absolute path on a new machine.

The vocabulary field matters even when every integer is in range. A file produced with one tokenizer can contain valid integers for another tokenizer. <code>vocab_size</code> catches only the range mismatch; the tokenizer name, special-token IDs, and tokenizer digest must also be recorded in the experiment record.

## What the loader verifies

<code>TokenDataset.__init__</code> performs several checks with different purposes:

1. It checks <code>format_version</code> and <code>vocab_size</code>, so a new schema or model vocabulary cannot be mistaken for the old one.
2. It recomputes SHA256 for every token and mask file. SHA256 is a digest, a short fingerprint of the bytes; a changed file should produce a different digest.
3. It memory-maps non-empty, one-dimensional integer arrays. Memory mapping avoids loading every shard into the Python heap, but it does not remove storage latency.
4. It checks that every token lies in $[0,V)$, where $V$ is the vocabulary size. This prevents an embedding lookup from going out of range.
5. It checks that each mask has the same shape as its token shard and values in $[0,1]$.
6. It records cumulative end offsets, allowing a global position to be mapped to a physical shard.

A hash proves that bytes match the manifest; it does not prove that the bytes represent the right corpus. That is why source digests, tokenizer metadata, and split records remain necessary.

### A worked cross-shard window

Use two tiny shards:

| Physical shard | Token contents | Count | Cumulative end |
|---|---|---:|---:|
| <code>v-00000.npy</code> | <code>[10, 11, 12, 13]</code> | 4 | 4 |
| <code>v-00001.npy</code> | <code>[20, 21, 22, 23, 24]</code> | 5 | 9 |

These ends use half-open code-index intervals: shard 0 covers global positions [0,4), and shard 1 covers [4,9). For <code>start=2</code> and <code>length=5</code>, the requested global positions are 2, 3, 4, 5, 6. <code>searchsorted([4,9], 2)</code> selects shard 0, so the loader reads local offsets 2 and 3 (<code>[12,13]</code>), then continues at local offsets 0, 1, and 2 in shard 1 (<code>[20,21,22]</code>):

~~~text
global positions: 2, 3 | 4, 5, 6
physical shard:   v-0   | v-1
local offsets:       2, 3 | 0, 1, 2
tokens:            12,13 | 20,21,22
~~~

For a next-token window of length 5, the training path reads one additional token before shifting labels. A shard boundary does not insert EOS and does not reset attention; it is only a storage boundary. Document isolation comes from tokenization and packing policy, not from the fact that two arrays have different filenames.

## Three identities to keep

There are three useful layers of identity:

- A **shard digest** detects replacement or corruption of one physical file.
- A **manifest digest** detects changed shard order, counts, vocabulary, or metadata.
- A **source digest** connects derived tokens back to the input records used to create them.

If someone reorders two shards, every shard digest still matches but the training sequence changes. The manifest digest catches that. If someone replaces a shard without editing the manifest, the shard digest catches it. Neither digest explains which original JSONL file was used, so the source digest remains part of provenance.

<code>prepare_sharded_data</code> currently assigns a random UUID-like version and refuses to overwrite an existing manifest. This protects old outputs. It does not make two byte-identical runs receive the same version; content-addressed versioning is a future design if that property is required.

## Global offsets and windows

Let shard lengths be $(n_1,n_2,\ldots,n_K)$ and define cumulative ends

$$
e_j=\sum_{k=1}^{j} n_k.
$$

A global position $p$ belongs to the first shard with $e_j>p$. <code>TokenDataset.rows</code> performs this search and subtracts the previous cumulative end to obtain a local offset. A requested window may therefore read the end of one shard and the beginning of the next without changing the logical token order.

The same distinction matters for document boundaries: physical storage is not semantic structure. If document offsets are needed for contamination checks or interpretability, the current manifest must be extended with a document-offset sidecar; it does not provide complete token-to-document reverse lookup.

## Dataset state and sampler state

The dataset answers “what tokens exist?” The sampler answers “which window comes next?”. <code>BatchStream</code> uses a NumPy generator to sample starts and stores generator state, data SHA256, sequence length, batch size, and seed in <code>state_dict()</code>.

Restoring only the seed is insufficient. A pseudo-random generator has an internal state that advances after every draw. Restoring the same seed restarts the sequence instead of continuing it. Restoring the generator state without checking the data fingerprint is also unsafe: the same random state can select different tokens from a changed dataset.

<code>sample_rows(step, indices)</code> derives independent windows from a <code>SeedSequence</code> built from the seed, step, and index. This is useful for fixed probes and evaluation, but it is a different sampling protocol from <code>next_batch_with_masks</code>. Reports should name which protocol produced a metric.

## Remote inspection example

After preparing a sharded dataset on the experiment machine, inspect the manifest and digest before requesting a long GPU job:

~~~
python -c "import json,hashlib,pathlib; p=pathlib.Path('runs/example_shards/manifest.json'); print(json.dumps(json.loads(p.read_text()), indent=2)); print(hashlib.sha256(p.read_bytes()).hexdigest())"
~~~

The command records the manifest itself; it does not replace the loader's per-shard checks. Save its output with the run metadata.

## Limits

Memory mapping reduces host memory pressure but random windows can still be storage-bound. Very small shards increase file and hash overhead; very large shards are harder to copy and recover. The teaching implementation prioritizes explicit validation and resumability. Distributed prefetch, cache policy, and source-aware sampling are later engineering layers.

The next chapter separates file layout from the statistical mixture of data sources.
