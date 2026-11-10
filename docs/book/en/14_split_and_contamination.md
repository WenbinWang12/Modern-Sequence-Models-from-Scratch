# 8. Splits and contamination: define what “unseen” means

## Better training performance does not necessarily mean better learning

A teacher assigns ten practice problems and uses the same ten problems in the next day's exam. A perfect score does not establish that the student understands the topic: memorizing the answers remains an explanation. Different questions testing the same knowledge provide better evidence for the ability we intended to measure.

Language model training has the same distinction. Training text participates in parameter updates, and reducing loss on repeatedly encountered data is the optimization objective. A **validation set** measures performance on data excluded from direct parameter updates and helps select configurations, training duration, and cleaning rules. A **test set** supplies the final evaluation after those choices are fixed; it should not repeatedly participate in them.

These names describe experimental roles, rather than three arbitrary folders. If we repeatedly inspect “test” scores to choose a model, that data has taken on the validation role. Selection has used its information even without backpropagation through it. “Unseen” therefore includes both exclusion from training updates and a clear limit on which development decisions may use the data.

Language models commonly compute loss by predicting the next token, and validation and testing can use the same objective. We first establish document membership; the next chapter explains tokenization in detail. The immediate question is which complete documents may enter training and which are reserved for evaluation.

## The evaluation question determines the split unit

“Randomly reserve one percent for validation” specifies a proportion without identifying what is sampled. Should chapters from the same book be sampled independently? Are ten websites mirroring an article ten independent objects? The answer depends on the claim we want to support.

| Intended evaluation | Objects to consider grouping | Confound to prevent |
|---|---|---|
| New documents from similar sources | An original and its duplicate or near-duplicate versions | Training already contains a mirror |
| Unseen books | Books | Adjacent chapters appear in different sets |
| New code projects | Repositories or projects | Files and branches from the same project cross sets |
| Text from a future period | Time boundaries and related document groups | Training has used future information |

This table is not an automatic policy. Temporal splits must still handle old articles republished later; project splits must consider copied code. Define the evaluation question, construct relevant groups, and only then choose proportions.

Deduplication from the previous chapter can supply useful relationships, but deciding which documents to retain and deciding which documents may cross sets are separate tasks. Complementary documents from one source can all be retained while sharing one split. Grouping can therefore be broader than the similarity conditions used to delete duplicates.

## Why independently splitting mirrors causes leakage

Suppose one article has \(m\) mirrors. Independently assign each record to validation with probability \(p\), and to training with probability \(1-p\). We consider only two sets here to make the probability argument explicit.

There are exactly two ways for the group to avoid crossing sets: all mirrors enter training, or all enter validation. Independence allows us to multiply probabilities, giving \((1-p)^m\) and \(p^m\). The events are mutually exclusive. Consequently,

$$
P(\text{duplicate group crosses train and validation})
=1-(1-p)^m-p^m.
$$

For \(m=10\), the probability is approximately \(9.56\%\) when \(p=0.01\), and \(65.13\%\) when \(p=0.1\). Risk depends on group size and split proportion; it is not almost certain for every group regardless of the ratio. With many groups, even a modest individual risk deserves attention.

Group-level assignment makes one decision for the entire group and lets its records inherit the result. Provided the grouping is correct, that group's probability of crossing sets is zero. This does not establish that all mirrors were discovered: missing relationships can still cause leakage.

## Giving each group a stable assignment

The teaching fixture adds a `split_group` field to each record. Records `d01` and `d02` contain the same short text from two source types and both belong to `guide-blue`. Records `d04` and `d05` contain different project notes and both belong to `project-cache`. They illustrate duplicate relationships and shared-project relationships respectively.

We want two properties: changing file traversal order should not change existing assignments, and adding an unrelated group should not reshuffle all old data. One direct implementation hashes a stable group ID and maps the digest to a bucket. A **bucket** here is simply an integer from 0 to 9,999.

The core implementation is:

~~~python
payload = json.dumps(
    ["book-group-split-v1", seed, group],
    ensure_ascii=False, separators=(",", ":"),
).encode("utf-8")
bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 10_000
~~~

The first step serializes the policy name, seed, and group ID. Field boundaries matter: plain concatenation would make `"ab" + "c"` indistinguishable from `"a" + "bc"`. The second step computes SHA-256, takes its first eight bytes, and interprets them as a nonnegative big-endian integer. Big-endian means earlier bytes occupy more significant positions. Taking the remainder maps that integer to one of 10,000 buckets.

The `seed` is one input to this deterministic rule. Identical inputs always produce identical buckets; changing the seed defines another assignment. Python's built-in `hash()` is unsuitable as a direct persistent splitting rule because string and byte hashes use process-varying randomization by default. See [Python's `__hash__` documentation](https://docs.python.org/3.12/reference/datamodel.html#object.__hash__).

Buckets below 6,000 enter training, buckets from 6,000 through 7,999 enter validation, and the remainder enter testing. **The 60% / 20% / 20% proportions are for demonstrating three sets with ten small groups**, not a recommendation for production pretraining. Modulo reduction is not perfectly uniform because \(2^{64}\) is not divisible by 10,000. If the digest prefix is modeled as a uniform 64-bit value, bucket probabilities differ only minutely. Deterministic inputs establish stability; approximate proportions additionally depend on the groups and digest distribution. These are separate properties.

Stability also has limits. Group IDs derived from unstable file enumeration change when enumeration changes. A newly discovered relationship may require merging two old groups and reconsidering their membership. Stability under appending data means adding an independent group while preserving the rule and existing IDs, not immunity to arbitrary revisions of the grouping.

## Group proportions, document proportions, and token proportions differ

Approximately uniform assignment of groups does not give each set an exact document proportion. Let \(I_g\) indicate whether group \(g\) enters validation: it is one if the group enters and zero otherwise. Let \(w_g\) count its documents. Then the validation document count is

$$
N_{\mathrm{val}}=\sum_g w_g I_g.
$$

In an idealized model, each group independently enters validation with probability \(p\). Since \(I_g\) equals one with probability \(p\), its mean is \(p\). Linearity of expectation gives

$$
\mathbb E[N_{\mathrm{val}}]=p\sum_g w_g.
$$

This describes a long-run average, **not an exact count in a particular split**. To describe how outcomes fluctuate around that average, we use variance. For a random quantity \(Z\), define \(\operatorname{Var}(Z)=\mathbb E[(Z-\mathbb E[Z])^2]\): measure each outcome's distance from the mean, square it, and average. Squaring prevents positive and negative deviations from canceling; greater variance indicates more spread. If \(Z\) counts documents, variance is measured in squared document counts.

Since \(I_g^2=I_g\),
\(\operatorname{Var}(I_g)=\mathbb E[I_g^2]-\mathbb E[I_g]^2=p(1-p)\).
Multiplying by the fixed weight \(w_g\) also multiplies every deviation by \(w_g\), so squared deviations acquire a factor of \(w_g^2\): \(\operatorname{Var}(w_g I_g)=w_g^2\operatorname{Var}(I_g)\). Under group independence, these weighted variances add:

$$
\operatorname{Var}(N_{\mathrm{val}})
=p(1-p)\sum_g w_g^2.
$$

The squared weights explain the influence of large groups. Suppose two groups contain 90 and 10 documents and \(p=0.2\). The expected validation count is 20, yet an individual split can contain only 0, 10, 90, or 100 documents. Exactly 20 is impossible. Replacing \(w_g\) with each group's token count under a fixed tokenizer gives the same derivation; long books can then dominate the variation. This is a model of independent random assignment, not a proof that a fixed SHA-256 output is random.

Report group counts, document counts, and length measures together. A production report should add token counts after the tokenizer is fixed. Our tokenizer has not yet been chosen, so this chapter's script reports UTF-8 bytes of document text. It does not label bytes as tokens or include JSON field names and serialization overhead in text length.

## Run a split with deliberate failure controls

The fixture `examples/book/fixtures/split_records.jsonl` contains 12 documents in ten groups. Two groups contain two documents each. One longer text reminds us that count and length are different measures. Run:

~~~bash
.venv/bin/python examples/book/split_walkthrough.py \
  --output outputs/book-split-v1 --seed 17
~~~

The script uses only the standard library and writes `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `split-report.json`. It neither trains a model nor selects a seed using model scores. The report's `groups` field lists group buckets and assignments. `totals` records group and document counts, text UTF-8 bytes, and output file digests for each set. Document counts should sum to 12 and group counts to ten; inspect your report for the actual proportions.

The script also reverses traversal order and appends a new group before recomputing assignments. Successful checks produce:

~~~json
{
  "order_independent": true,
  "existing_assignments_stable_after_append": true
}
~~~

Producing three files alone does not establish that a validator works. The script therefore constructs two deliberately incorrect alternatives in memory:

| Failure control | Deliberate change | Expected rejection |
|---|---|---|
| A group crosses sets | Move one member of a multi-document group to another set | `group_leak` |
| Different groups share a complete text | Keep group assignments but copy a text into another set | `normalized_content_leak` |

The first tests group boundaries; the second tests content boundaries. If the validator fails to reject either control, the script fails. These alternatives are never written into the normal split files. `failure_probes` records the intercepted reasons; an expected rejection message is not a failure of the valid split.

The content check applies NFC normalization and compares SHA-256 digests of complete texts. It detects the fixture's full-text duplication, but not a shared passage, changed names and numbers, or paraphrased questions. Digest collisions could also produce false alarms: a digest match is an interception signal, not a semantic judgment. The report's `scope` explicitly limits its claim to grouping and NFC full-text digest overlap.

Use a fresh output directory for another run to preserve prior evidence. A replacement `--input` must retain at least one multi-document group and populate at least two sets for the failure demonstrations to be meaningful. Otherwise the script rejects the demonstration instead of inventing a successful verification.

Additional metadata is retained with each document, so it too must remain valid JSON. This example rejects `NaN`, `Infinity`, and numbers outside the supported finite floating-point range, such as `1e999`. Such a value at any nesting level stops the whole split before the output directory is created. The literal `1e999` is valid JSON number syntax; its rejection is a numeric-range limit of this reader, not a syntax error.

## Establish boundaries before encoding and packing

Documents should enter separate encoding pipelines after splitting. If we first concatenate every document into one token stream and then assign windows to training and validation, the windows can share substantial content: two windows of length 1,024 offset by one position overlap by 1,023 tokens. Those training instances are not independent evidence.

A clearer order is to establish related groups, split them, encode each set, and generate separate manifests. The current `TokenDataset` reads a continuous token stream described by one manifest; it does not create training, validation, and test boundaries automatically. Our script has produced real JSONL inputs for subsequent encoding. Do not recombine the three sets and sample windows from their joint stream.

A tokenizer is itself learned from data. For a strict unseen-data evaluation, restrict tokenizer training to the training set and encode validation and test with that frozen tokenizer. The next chapter distinguishes learning a tokenizer from using it to encode text.

## How to state a negative contamination result

Deduplication usually compares documents within training data. Evaluation contamination detection compares training inputs against evaluation objects. Some matching methods can be shared, but the comparison scope differs. A training corpus with no internal duplicates can still contain benchmark questions or answers.

Checks can start with full-text matches and extend to long substrings, normalized questions, or approximate similarity when appropriate. Each method has a coverage boundary: common short definitions can create false alarms, while paraphrases can evade detection. Record candidates, matching spans, comparison units, review outcomes, and methods that were not implemented.

After this chapter's script passes, a defensible conclusion is: “No group crosses sets, no cross-set NFC full-text digest overlap was detected within this check, and both failure controls were rejected.” It does not establish that the data is contamination-free, much less guarantee independence from every future evaluation. The tiny corpus cannot measure generalization quality either; it checks whether the split mechanism follows its rules.

Document membership is now fixed. Next we examine another choice that affects training: how the same text becomes an integer sequence, and how vocabulary and tokenization shape model inputs, outputs, and computation.
