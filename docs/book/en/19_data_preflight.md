# 13. Preflight: inspect inputs before starting a long run

## An openable file is not enough

Suppose training immediately fails with an out-of-vocabulary target. The GPU, compiler, and model have already started, but the cause may be simple: the data uses a 32K tokenizer while the model has only 259 output positions. We do not need thousands of training steps to discover this. Checking data, tokenizer, model, and run configuration together is preflight.

Preflight looks for contradictions among inputs. It differs from evaluation: intact files and valid labels do not establish representative data, and a finite tiny-model loss does not establish language ability. First ask whether the declared experiment can execute, then whether its results are useful.

## Follow one target through the checks

A target ID must belong to the intended tokenizer vocabulary. Even an in-range integer can mean different text under another tokenizer. Checking \(0\le y<V\) is therefore insufficient; tokenizer files, normalization rules, and special-token semantics also need fixed identities.

The model must then be able to assign probability to that target. The current sharded loader requires the manifest vocabulary size to equal the model vocabulary size passed to it. This is a repository interface constraint, not a theorem about every language model: a larger output vocabulary can be mathematically valid, but does not bypass this version check. Editing the manifest number also cannot make data from another tokenizer compatible.

Third, the target must come from the right position. A length-\(L\) next-token prediction requires \(L+1\) consecutive tokens. Targets are shifted by one, and the loss mask must align with those targets. An all-zero mask may produce a numerically safe zero loss, but that means no supervision rather than perfect prediction.

Finally, the available context must match the declared protocol. Current continuous-stream windows can cross EOS; loss masking does not automatically reset attention or fast weights. Independent-document experiments need implemented and checked state isolation, not just a report asserting independence.

## Which checks already exist?

| Object | Existing code checks | Upstream or manual checks still needed |
|---|---|---|
| Shard files | Paths, SHA256, nonempty one-dimensional integers, count | Provenance, licensing, extraction quality |
| Tokens and masks | ID range, matching shapes, finite mask values in \([0,1]\) | Special-token semantics, correct target selection |
| Manifest | Format version, vocabulary size, shard list | Tokenizer file identity, document lookup, split contamination |
| Windows | Length and index bounds | Source proportions, supervision distribution across many windows |
| Restoration | Data fingerprint and sampling configuration | Actual next-step comparison against uninterrupted execution |

The first three rows primarily correspond to <code>TokenDataset</code> in <code>ttt/dataset.py</code>; sampling state belongs to <code>BatchStream</code> in <code>ttt/data.py</code>. There is no single <code>ttt.preflight</code> command implementing this entire table. Keeping the missing analyses explicit prevents loader success from being mistaken for complete acceptance.

## Generate a small report from an existing example

Use the repository's four JSONL teaching records on the target Linux node, from the repository root after environment setup. Generate shards once. The generator refuses to overwrite an existing manifest; for repeated inspection, run only the second block, and use a new directory when changing data.

~~~bash
.venv/bin/python -m ttt.data \
  --input examples/sample.jsonl \
  --output outputs/preflight-example \
  --sharded --tokenizer byte --shard-tokens 256
~~~

Now inspect the manifest and take the first 17-token window, yielding 16 prediction targets. A fixed start makes the example easy to check; it does not measure sampling coverage.

~~~bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from ttt.dataset import TokenDataset

dataset = TokenDataset("outputs/preflight-example/manifest.json", 259)
if dataset.size < 17:
    raise ValueError("The stream must contain at least 17 tokens.")
tokens, mask = dataset.rows([0], 17)
if tokens.shape != (1, 17) or mask.shape != (1, 16):
    raise ValueError("Unexpected token or target-mask shape.")
valid_weight = float(mask.sum())
if valid_weight <= 0:
    raise ValueError("The inspected window has no supervised target.")
report = {
    "scope": "loader_invariants_and_one_window",
    "manifest_sha256": dataset.fingerprint,
    "stream_tokens": dataset.size,
    "window_tokens": tokens.tolist(),
    "target_mask": mask.tolist(),
    "valid_weight": valid_weight,
    "passed": True,
}
path = Path("outputs/preflight-example/window-check.json")
path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY
~~~

The code implies that a successful report contains one row of 17 integers in <code>window_tokens</code>, one row of 16 weights in <code>target_mask</code>, and a positive <code>valid_weight</code>. No invented hash or token count is presented as a measurement; the generated JSON records the actual data version. An exception prevents a new success report from being written. On repeated runs, also inspect the old report's time and digest rather than mistaking an existing file for current success.

This example checks the loader's invariants and one window. It does not inspect every window, contamination, source mixtures, tokenizer file identity, or model training. The report's <code>scope</code> field makes that boundary explicit.

## Deliberately create one mismatch

After the successful inspection above, keep the manifest unchanged and pass a model vocabulary size of 32,000:

~~~bash
.venv/bin/python - <<'PY'
from ttt.dataset import TokenDataset

try:
    TokenDataset("outputs/preflight-example/manifest.json", 32000)
except ValueError as error:
    print(type(error).__name__, str(error))
else:
    raise RuntimeError("Expected the vocabulary mismatch to be rejected.")
PY
~~~

The current implementation should raise <code>ValueError</code> with the message “数据清单版本或词表大小不匹配。” (manifest version or vocabulary size mismatch). This expectation comes from that source branch; execution confirms the machine's behavior. Since the same manifest was just read successfully with 259, the changed vocabulary argument isolates the failed interface. The example modifies no source data and does not label the failed check as success.

## Connect each error to the right object

For a manifest-version or vocabulary mismatch, compare the manifest, model configuration, and tokenizer's actual vocabulary size first. Changing learning rate or layer count cannot fix it. For a shard checksum failure, check transfer completeness and whether the manifest belongs to another version. Recomputing the hash and replacing it in the manifest would hide the substitution rather than recover the intended file.

If tokens and masks are individually valid but training repeatedly reports zero loss, inspect valid mask weights and print a short document's inputs, targets, and masks together. If loss is finite but validation is suspiciously easy, return to split and near-duplicate checks. Different failures concern different objects; one range check cannot answer them all.

## Evidence still needed before a long run

Complete acceptance also needs source and cleaning records, grouped split and contamination analysis, tokenizer identity, declared packing semantics, and a restoration comparison. Record pass, warning, fail, or not_checked for each item, together with scope and evidence. Unperformed checks must not default to pass.

Use <code>warning</code> for known and accepted limitations, such as an experiment deliberately allowing continuous cross-document context, and explain the effect on interpretation. Use <code>fail</code> for contradictions preventing the declared experiment, such as incompatible vocabularies. <code>not_checked</code> means evidence is absent; it does not disappear because the program raised no exception.

Keep these records in separate reports alongside the run. The current trainer's <code>experiment.json</code> stores experiment configuration and does not automatically collect upstream reports, so their paths and digests must be recorded explicitly. The next chapter uses these inputs in an ordinary dense language model to inspect gradients, optimization, and restoration before adding more complex structures.
