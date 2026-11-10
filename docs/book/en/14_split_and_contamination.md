# 8. Splits and contamination: define what “unseen” means

## Why random records can produce an easy validation set

Imagine ten mirrors of one tutorial. If each record is assigned independently, some copies almost certainly enter training and others enter validation. Their file names differ, but their text does not. A low validation loss then tells us that the model can continue a familiar tutorial; it does not tell us how it handles an unseen source.

The split unit must follow the evaluation question. Use a document cluster to test new documents of known sources, a project or repository to test unseen code projects, and a time window to test future news. “Validation is 2%” is only a ratio; a reproducible protocol also states the unit, source, time range, grouping rule, and contamination check.

## Keep related records together

Let `split_group` be a stable identifier for a duplicate cluster, book, repository, or other unit that should remain together. A deterministic hash can assign the group without depending on file order:

~~~python
import hashlib


def choose_split(group_id):
    digest = hashlib.sha256(group_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    if bucket < 100:
        return "test"
    if bucket < 300:
        return "validation"
    return "train"
~~~

The expected group proportions are about 1%, 2%, and 97%, but token proportions are not guaranteed. One long book can dominate the validation token count. The group ID itself must be stable: using a file's enumeration index creates an apparently deterministic split that changes when parallel download order changes. Python's process-randomized `hash()` is also unsuitable for a cross-run protocol; SHA256 makes the mapping explicit.

## Split before tokenization and packing

If we concatenate all documents, tokenize the stream, and then sample validation windows, adjacent windows can share almost every token. For length $L=1024$, windows offset by one position overlap in $1023$ tokens. A model can appear to generalize while simply seeing its answer in a neighboring window.

The safer order is:

1. assign documents or groups to train, validation, and test;
2. encode each set separately;
3. pack each set into its own token files and manifest;
4. sample windows only inside the selected set.

Physical shard files may split a long token stream, but a training sampler must not create a logical window that crosses train and validation. The repository loader reads one manifest and does not infer these boundaries for us.

## Contamination is a separate comparison

Internal deduplication asks whether training records repeat one another. Contamination analysis compares training input with an evaluation object. Candidate methods include full exact match, normalized long-substring match, n-gram or fingerprint similarity, and task-specific matching of a question stem. Each method has a different false-positive and false-negative profile.

For example, changing names and numbers in a math problem can defeat a byte hash, while a common definition can match many sources without leaking the complete benchmark item. A candidate match is evidence to review, not an automatic declaration of contamination. Record the method, normalization, candidate count, reviewed count, accepted matches, and unresolved cases.

Evaluation access also matters. Repeatedly inspecting one “test” set while selecting models turns it into a development set. Keep a frozen test set whose labels and examples do not participate in routine choices; it is the final held-out evidence for a claim.

## Read the split report with numbers

Suppose validation contains 2% of documents but 18% of tokens. This may be a legitimate result of long books, or it may reveal that grouping or length accounting is wrong. Inspect document-length quantiles, source retention, and group sizes before changing the ratio.

Now suppose exact train/validation overlap is zero, but many records share a domain and page template. The report supports “no exact match under this normalization,” not “no related content exists.” State what was checked and what remains outside the method's coverage.

A split report should include document and token counts per group and split, language and source distributions, group intersections, contamination candidate and review counts, the rule version, and input digests. These are bounded pieces of evidence, not an absolute guarantee about every possible evaluation set.

## Mapping to this repository

The repository's sample data is for pipeline checks, not for a claim about generalization. After producing independent JSONL files, build separate manifests:

~~~bash
python -m ttt.data \
  --input data/train-v1.jsonl \
  --output data/tokens-train-v1 \
  --sharded \
  --tokenizer byte

python -m ttt.data \
  --input data/validation-v1.jsonl \
  --output data/tokens-validation-v1 \
  --sharded \
  --tokenizer byte
~~~

Then pass the training manifest to the training job and use the validation manifest only for evaluation. Keep the split report and contamination report beside both manifests. The next chapter explains how the tokenizer turns each split into integer IDs and why vocabulary choices affect the model's output head.

## Minimal checkpoint, evidence, and failure diagnosis

The two split JSONL files must produce separate manifests:

~~~bash
python -m ttt.data \
  --input <train-jsonl> \
  --output outputs/split-train \
  --sharded --tokenizer byte --shard-tokens 256
python -m ttt.data \
  --input <validation-jsonl> \
  --output outputs/split-validation \
  --sharded --tokenizer byte --shard-tokens 256
~~~

The expected artifacts are two independent <code>manifest.json</code> files and one split/contamination report containing group, document, token, source, candidate-match, and review counts. If train and validation digests or group IDs overlap, stop before training and repair the split; a low validation loss cannot cancel contamination.

## Summary and next question

The split unit must match the evaluation question: document cluster, project, time period, or task object. Zero exact overlap rules out only the narrowest duplicate class; it does not prove that near or semantic contamination is absent. The next chapter fixes the tokenizer and vocabulary for each split and follows token IDs into the output head.
