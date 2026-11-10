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

The vocabulary field matters even when every integer is in range. Suppose two vocabularies both contain 259 entries, but ID 68 means A in one and B in the other. Every stored ID can be legal and every file digest can match while the model receives different meanings. <code>vocab_size</code> checks vocabulary-size agreement, not the identity of equally sized mappings. The tokenizer version, special-token IDs, and tokenizer-file digest must also be recorded in the experiment record.

## What the loader verifies

<code>TokenDataset.__init__</code> performs several checks with different purposes:

1. It checks <code>format_version</code> and <code>vocab_size</code>, so a new schema or model vocabulary cannot be mistaken for the old one.
2. It recomputes SHA256 for every token and mask file. SHA256 is a digest, a short fingerprint of the bytes; a changed file should produce a different digest.
3. It memory-maps non-empty, one-dimensional integer arrays. Memory mapping avoids loading every shard into the Python heap, but it does not remove storage latency.
4. It checks that every token lies in $[0,V)$, where $V$ is the vocabulary size. This prevents an embedding lookup from going out of range.
5. It checks that each mask has the same shape as its token shard and values in $[0,1]$.
6. It records cumulative end offsets, allowing a global position to be mapped to a physical shard.

A matching digest is evidence that bytes match the recorded version, assuming a trusted reference manifest and ignoring practically infeasible SHA256 collisions. It does not establish corpus quality, permission to use it, or immunity to later edits. Replacing both a shard and its recorded digest also defeats comparison with that replacement manifest. Source digests, tokenizer metadata, split records, and a trusted record of the intended version remain necessary.

### A worked cross-shard window

Use two tiny shards:

| Physical shard | Token contents | Count | Cumulative end |
|---|---|---:|---:|
| <code>v-00000.npy</code> | <code>[10, 11, 12, 13]</code> | 4 | 4 |
| <code>v-00001.npy</code> | <code>[20, 21, 22, 23, 24]</code> | 5 | 9 |

These ends use half-open code-index intervals: shard 0 covers global positions [0,4), and shard 1 covers [4,9). For <code>start=2</code> and <code>length=5</code>, the requested global positions are 2, 3, 4, 5, 6. <code>searchsorted([4,9], 2, side='right')</code> selects shard 0, so the loader reads local offsets 2 and 3 (<code>[12,13]</code>), then continues at local offsets 0, 1, and 2 in shard 1 (<code>[20,21,22]</code>):

~~~text
global positions: 2, 3 | 4, 5, 6
physical shard:   v-0   | v-1
local offsets:       2, 3 | 0, 1, 2
tokens:            12,13 | 20,21,22
~~~

The endpoint shows why <code>side='right'</code> matters. Global position 4 belongs to the second shard at local offset \(4-4=0\). <code>searchsorted([4,9], 4, side='right')</code> returns shard index 1; the default left search would return 0 and assign the position to the already ended first shard.

In this example, <code>length=5</code> is the raw count and supplies four prediction positions. If the model's configured input length is 5 instead, the training path reads six raw tokens before shifting labels. A shard boundary does not insert EOS and does not reset attention; it is only a storage boundary. Document isolation comes from tokenization and packing policy, not from the fact that two arrays have different filenames.

## Three identities to keep

There are three useful layers of identity:

- A **shard digest** detects replacement or corruption of one physical file.
- A **manifest digest** detects changed shard order, counts, vocabulary, or metadata.
- A **source digest** connects derived tokens back to the input records used to create them.

If someone reorders two shards, every shard digest still matches but the training sequence changes. Comparison with a trusted manifest digest catches that. Replacing a shard while retaining its trusted manifest triggers a digest mismatch. The loader checks the bytes available during loading; it does not lock files against later modification. Published data directories should be treated as immutable versions. Neither digest explains which original JSONL file was used, so the source digest remains part of provenance.

<code>prepare_sharded_data</code> assigns a random UUID version and rejects an existing manifest at the start. Use a single writer per output directory: that existence check is not a multiprocess write lock. Two identical runs do not receive the same version. Content-addressed versioning would instead derive identity from input and complete processing configuration, including the format version.

### The source digest must describe consumed bytes

Hashing the source only after processing leaves a provenance gap. Suppose the preparer has read `{"text":"A"}` and written A's tokens. Another process then replaces the source with `{"text":"Z"}`. A final-only source hash attaches Z's identity to A's tokens. The shards are intact, so ordinary shard verification cannot detect this mistake.

The preparer now hashes the source before reading, hashes the raw bytes actually consumed while reading, and hashes the source again before publishing the manifest. All three must agree. Blank lines and CRLF line endings participate in the consumed digest because source identity describes complete bytes, not just parsed JSON values. A mismatch prevents manifest publication. Some shard files may already have been written when preparation fails; without a successful manifest referencing them, they are not a completed dataset version.

This binds the output to consumed bytes and detects differences between the compared versions. It does not prove that the source never changed during processing: a file could change and be restored, or change after the final check. Formal preparation should still use immutable, read-only input versions. Regressions in `tests/test_dataset.py` change A to Z during encoding and require failure without a manifest, and separately cover blank lines and CRLF in the source digest. They were written but not executed in this review.

## Global offsets and windows

Let shard lengths be $(n_1,n_2,\ldots,n_K)$ and define cumulative ends

$$
e_0=0,\qquad e_j=\sum_{k=1}^{j} n_k.
$$

A zero-based global position $p$ belongs to shard $j$ satisfying $e_{j-1}\le p<e_j$, at local offset $p-e_{j-1}$. <code>searchsorted</code> returns the zero-based shard index $j-1$. For example, $p=6$ in the second shard gives local offset $6-4=2$, containing token 22. A window can read the end of one shard and the beginning of the next without changing logical order. Changing physical shard size should preserve every valid window **provided both tokens and masks remain identical**. The preceding chapter explains why switching between the single-NPY and sharded preparation entry points does not automatically meet that condition: their mask and empty-text policies differ.

The same distinction matters for document boundaries: physical storage is not semantic structure. If document offsets are needed for contamination checks or interpretability, add a document-offset sidecar linked to source record IDs. The current manifest does not provide complete token-to-document reverse lookup; shard filenames and token offsets alone cannot recover that provenance.

## Dataset state and sampler state

The dataset answers “what tokens exist?” The sampler answers “which window comes next?”. <code>BatchStream</code> uses a NumPy generator to sample starts and stores generator state, data SHA256, sequence length, batch size, and seed in <code>state_dict()</code>.

Restoring only the seed is insufficient. A pseudo-random generator has an internal state that advances after every draw. Restoring the same seed restarts the sequence instead of continuing it. Restoring the generator state without checking the data fingerprint is also unsafe: the same random state can select different tokens from a changed dataset.

<code>sample_rows(step, indices)</code> derives independent windows from a <code>SeedSequence</code> built from the seed, step, and index. This is useful for fixed probes and evaluation, but it is a different sampling protocol from <code>next_batch_with_masks</code>. Reports should name which protocol produced a metric.

## Remote inspection example

Run this single snippet from the repository root on the prepared remote CPU node. It uses the repository JSONL fixture. If <code>outputs/dataset-sample/manifest.json</code> is absent, it generates shards with at most 256 tokens each. If it exists, the snippet checks the input digest and byte-encoding mode before reading it, without overwriting the manifest. Use a new output directory for another data version; do not edit the old manifest to bypass a mismatch. This command was not executed during the manuscript review.

We read two raw windows of five tokens. The first starts two positions before the first shard ends and crosses into the second; the second begins exactly at that boundary. Each produces four inputs, four targets, and four target-mask values. The <code>rows</code> method has already removed the first mask column, so the returned mask must not be shifted again.

~~~bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np
from ttt.data import ByteTokenizer
from ttt.dataset import TokenDataset, file_sha256, prepare_sharded_data

source = Path("examples/sample.jsonl")
output = Path("outputs/dataset-sample")
manifest_path = output / "manifest.json"
if not manifest_path.exists():
    prepare_sharded_data(source, output, ByteTokenizer(), shard_tokens=256)

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if (manifest["input_sha256"] != file_sha256(source)
        or manifest["tokenizer"] != "byte"
        or manifest["supervised"] is not False):
    raise ValueError("Existing data differs from this recipe; use a new output directory.")

dataset = TokenDataset(manifest_path, ByteTokenizer.vocab_size)
if len(dataset.shards) < 2 or dataset.shards[0].size < 2 or dataset.shards[1].size < 5:
    raise ValueError("This example needs two shards with at least 2 and 5 tokens.")
boundary = int(dataset.ends[0])
starts = [boundary - 2, boundary]
tokens, masks = dataset.rows(starts, 5)
expected = np.stack([
    np.concatenate([dataset.shards[0][-2:], dataset.shards[1][:3]]),
    dataset.shards[1][:5],
])
np.testing.assert_array_equal(tokens, expected)
if tokens.shape != (2, 5) or masks.shape != (2, 4):
    raise ValueError("Unexpected window or target-mask shape.")

report = {
    "scope": "two_windows_at_first_shard_boundary",
    "manifest_sha256": dataset.fingerprint,
    "cumulative_ends": dataset.ends.tolist(),
    "starts": starts,
    "tokens": tokens.tolist(),
    "X": tokens[:, :-1].tolist(),
    "Y": tokens[:, 1:].tolist(),
    "M": masks.tolist(),
    "matches_direct_shard_read": True,
}
path = output / "rows-check.json"
path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY
~~~

Initial generation produces the manifest, token/mask shards, and their digests. A successful inspection also writes <code>outputs/dataset-sample/rows-check.json</code>, containing actual cumulative ends, starts, raw tokens, \(X,Y,M\), and the manifest digest. The <code>tokens</code> shape should be <code>[2,5]</code>; the three derived arrays have shape <code>[2,4]</code>. Comparing against direct reads of the two shard arrays checks token order for these two windows. Printing their masks helps inspection; it does not validate every window or document-boundary semantics.

The manifest stores each shard's <code>count</code>, not <code>start</code> or <code>end</code>. The report's <code>cumulative_ends</code> is computed by the loader. On disagreement, inspect shard order, counts, array lengths, <code>side='right'</code>, and local offsets. A digest or vocabulary failure requires recovering the intended data version, not merely replacing recorded digests. A later successful inspection updates the report; an exception writes no new report, so an old file does not establish success for the current attempt.

## Limits

Memory mapping reduces host memory pressure but random windows can still be storage-bound. Very small shards increase file and hash overhead; very large shards are harder to copy and recover. The teaching implementation prioritizes explicit validation and resumability. Distributed prefetch, cache policy, and source-aware sampling are later engineering layers.

The next chapter separates file layout from the statistical mixture of data sources.

## Summary and next question

A dataset provides a verifiable logical token stream; a shard is a storage fragment; the manifest records shard order and identity. The sampler separately chooses the next window. Recovery needs the data digest, generator state, and window protocol in addition to a seed. The next chapter asks which sources produced those windows and how a sampler realizes a target recipe.
