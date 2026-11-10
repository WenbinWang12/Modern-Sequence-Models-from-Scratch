# 13. Data preflight

## Why check before a long run?

A four-step tiny run can expose an import error quickly. A multi-day job can spend hours downloading data and compiling kernels before discovering that the tokenizer and model vocabulary disagree. **Preflight** is a short, explicit acceptance step that checks whether the inputs form one coherent experiment before a long GPU allocation.

Preflight is not a quality score. It does not prove that the model will learn useful language. It proves that the selected data, tokenizer, model configuration, sampler, and output directory agree on the objects they refer to.

## Eight checks, with the reason for each

### 1. Vocabulary alignment

The manifest vocabulary size, tokenizer vocabulary size, and model configuration must agree:

$$
V_{\mathrm{manifest}}=V_{\mathrm{tokenizer}}=V_{\mathrm{model}}.
$$

Special-token IDs must lie in $[0,V)$, and the BOS/EOS/PAD insertion rules used to create the shards must match the tokenizer metadata. A legal integer from the wrong tokenizer is still a semantic error.

### 2. File completeness and integrity

Every token and mask shard must exist. Recomputed SHA256 digests must equal the manifest, <code>count</code> must equal array length, arrays must be one-dimensional integers, and mask shape must equal token shape. A missing or changed file is a hard failure.

### 3. Trainable windows

For sequence length $L$, the dataset must provide at least $L+1$ tokens. Tokens must be in range, the returned mask must have length $L$, and at least one target must have a nonzero mask. A batch whose mask is all zero has no learning signal and should be rejected even if the numerical code avoids division by zero.

### 4. Boundary semantics

The report must state whether packing is independent-document or continuous-stream, whether windows can cross EOS, and whether the attention mask resets at document boundaries. The current loader can cross EOS and shard boundaries; preflight must not describe it as document-isolated.

### 5. Split and contamination evidence

Training and validation document groups must be disjoint according to the split policy. Record duplicate or near-duplicate findings, source and language counts, length statistics, and any blocked contamination decisions. A validation loss is not an independent estimate if the same document appears in training under another formatting or shard.

### 6. Sampling and resume semantics

Record seed, recipe version, shuffle or sampling protocol, rank assignment, initial position, and checkpoint resume rule. For the current <code>BatchStream</code>, also record the data fingerprint, sequence length, batch size, and generator state. A matching seed alone does not identify the next batch.

### 7. Model and data interface

Check <code>sequence_length &lt;= max_position_embeddings</code>, valid <code>ttt_chunk_size</code>, compatible sliding-window settings, and matching C/F architecture. If padding is used, verify that its mask is passed through the same shift as labels. These checks prevent a data shape that is legal for NumPy but invalid for the model's causal computation.

### 8. Output safety

The output directory must be empty, or the run must explicitly declare resume or initialization from a checkpoint. A checkpoint may resume only when its data digest, tokenizer identity, resolved configuration, and sampler settings match the current inputs. Otherwise the run should stop rather than silently continue on new data.

## A report a human can audit

Each item should contain its inputs, observations, status, and reason:

~~~
{
  "name": "vocab_alignment",
  "status": "pass",
  "inputs": {
    "manifest_sha256": "<sha256>",
    "config_sha256": "<sha256>"
  },
  "observed": {
    "manifest_vocab_size": 259,
    "tokenizer_vocab_size": 259,
    "config_vocab_size": 259
  },
  "message": "all values agree"
}
~~~

Use <code>pass</code>, <code>warning</code>, and <code>fail</code> deliberately. A warning states the human decision and its scope; a fail blocks a long job. Save the report under the run directory and reference it from <code>experiment.json</code>, so a later reader can see what was checked before training.

## A complete pass, warning, and fail example

Here is a shortened report for a tiny run. File and window checks pass, continuous windows across EOS receive a warning, and a vocabulary mismatch fails the run:

~~~
{
  "checks": [
    {"name": "file_integrity", "status": "pass", "message": "2 shards and masks match their hashes"},
    {"name": "boundary_semantics", "status": "warning", "message": "windows may cross EOS; document isolation is not claimed"},
    {"name": "vocab_alignment", "status": "fail", "observed": {"manifest": 259, "tokenizer": 259, "model": 32000}, "message": "model vocab_size differs"}
  ],
  "ready_for_long_run": false
}
~~~

The fail should be repaired before submitting a long job. Either the model must use the 259 positions required by the byte tokenizer, or the experiment must train a new 32K tokenizer, regenerate token shards, update the manifest and experiment identity, and run preflight again. The warning also needs a recorded human decision. The current repository has no standalone preflight command, but a remote operator can force the loader checks before requesting a GPU:

~~~
python -c "from ttt.dataset import TokenDataset; TokenDataset('runs/example_shards/manifest.json', 259); print('manifest and shards: pass')"
~~~

This command checks implemented file and range invariants. It does not replace split, contamination, or distribution checks.

## What the repository provides today

There is not yet a standalone <code>ttt.preflight</code> command. The current minimum gate is the validation performed by <code>TokenDataset</code>, a sampled window check through <code>BatchStream</code>, and a tiny training or validation run. On the remote experiment machine, an operator can at least force the loader to read the manifest before requesting a GPU:

~~~
python -c "from ttt.dataset import TokenDataset; TokenDataset('runs/example_shards/manifest.json', 259); print('manifest and shards: pass')"
~~~

This command checks the loader's implemented invariants; it does not perform split contamination analysis, source-mixture accounting, or model-quality evaluation. Those remain explicit inputs to the preflight report. A future <code>ttt.preflight</code> module should expose the eight checks without requiring a GPU.

## The boundary of the claim

Passing preflight establishes input consistency and a recoverable experiment identity. It does not establish useful perplexity, broad generalization, or a fair architecture comparison. Those claims require the dense baseline, independent evaluation data, and recorded cost and state evidence. The next chapter fixes that baseline before we add TTT or other changing state.
