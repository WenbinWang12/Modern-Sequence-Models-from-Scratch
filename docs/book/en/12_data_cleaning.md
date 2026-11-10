# 6. Cleaning text: every rule changes the distribution

## Why “clean” is not a natural label

Consider these three records:

~~~text
A. Home | Login | Related links | Next page
B. def normalize(text):
       return text.strip()
C. The proof has three steps. First choose ε > 0 ...
~~~

A looks like navigation boilerplate. B is short and contains punctuation, but it is useful code. C contains symbols that a web-prose filter may mistake for noise. A rule based only on alphabetic-character ratio or sentence count can delete B and C together with A.

Cleaning is therefore a distribution decision. The goal is to reduce obvious noise while measuring false deletions. Every threshold should have a version, an order, a denominator, and a sample of decisions that a person can inspect. A higher retention rate is not automatically better: retaining boilerplate can waste tokens, while deleting all short definitions can remove the knowledge we wanted.

## Three kinds of processing

Keep these operations separate because they answer different questions:

1. **Format repair** changes representation without intending to change meaning, such as converting CRLF to LF or removing a byte-order mark.
2. **Rule rejection** removes records that fail a mechanical condition, such as an empty body or an impossible length.
3. **Content assessment** records a score or decision about language, quality, privacy, or safety. These decisions can be wrong and need an audit sample.

If all three write `bad_text`, a sudden retention drop cannot be diagnosed. A stable rejection taxonomy might include `invalid_schema`, `empty`, `too_short`, `too_long`, `bad_unicode`, `language_mismatch`, and `safety_review`. Keep the first rejection reason separate from the full set of diagnostics so the counts remain additive.

## A conservative normalization function

Unicode normalization illustrates why the training view and the comparison view should be distinguished. NFC can combine canonically equivalent sequences. NFKC additionally folds some compatibility characters; that may help search keys, but it can alter technical typography, full-width symbols, or code. Do not overwrite the original text merely to make hashing convenient.

The following teaching function returns both a decision and a reason:

~~~python
import unicodedata


def inspect_document(record, *, min_chars=1, max_chars=100_000):
    text = record.get("text")
    if not isinstance(text, str):
        return None, "invalid_text_type"

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return None, "empty"
    if len(text) < min_chars:
        return None, "too_short"
    if len(text) > max_chars:
        return None, "too_long"

    result = dict(record)
    result["text"] = text
    return result, "kept"
~~~

The order is part of the contract. Type errors are not empty strings; normalization happens before length measurement; and each record receives one primary state. If later rules add multiple diagnostics, preserve a separate list such as `diagnostics=["too_short", "low_alpha_ratio"]` without counting one record twice in the primary rejection table.

## A conservation equation for the report

Let $n_{\mathrm{input}}$ be the number of records entering one cleaning stage, $n_{\mathrm{kept}}$ the records that leave it, and $n_{\mathrm{rejected},r}$ the records whose primary reason is $r$. A useful report satisfies

$$
n_{\mathrm{input}} = n_{\mathrm{kept}} + \sum_r n_{\mathrm{rejected},r}.
$$

The equation is only meaningful when all terms use the same unit and stage. If one HTML page is split into three passages, page counts and passage counts cannot be placed in the same equation. Report the parent-child relation separately.

Also state the denominator of every percentage. “The length rule removed 10%” could mean 10% of records after schema validation or 10% of the raw source. Both are valid measurements, but they answer different questions.

## A numerical example and a failure mode

Imagine 10,000 input documents. Schema validation keeps 9,900; non-empty checks keep 9,600; length rules keep 9,100; quality review keeps 8,700. The table below is an accounting example, not a measurement from this repository:

| stage | documents | of raw input |
|---|---:|---:|
| input | 10,000 | 100% |
| schema valid | 9,900 | 99% |
| non-empty | 9,600 | 96% |
| length valid | 9,100 | 91% |
| quality rules | 8,700 | 87% |

The total retention rate does not tell us whether cleaning succeeded. If a code source retains only 20% while prose retains 90%, the aggregate 87% hides a domain shift. Inspect counts by source, language, length quantile, and token count. Compare random retained records, random rejected records, records near each threshold, and records from rare sources using a fixed seed.

Short and mixed-language records are especially difficult for language detection. A Chinese code comment being classified as “not English” only says that it fails an English filter; it does not prove that it is low quality. A rule must be tied to the intended training distribution.

## Mapping to the repository boundary

The current `ttt.data` path performs limited schema and whitespace checks while converting text into tokens. It does not implement a full quality, language, PII, or safety pipeline, and it does not claim to make a legal decision. A production project should run those upstream stages, write a new versioned JSONL directory, and then call the repository tokenizer.

For the repository's educational input, the interface is:

~~~bash
.venv/bin/python -m ttt.data \
  --input examples/sample.jsonl \
  --output data/clean-corpus-v1 \
  --sharded \
  --tokenizer byte
~~~

The command assumes the JSONL already contains a `text` field and records an input digest in `manifest.json`. It does not make the preceding cleaning decisions for us. Keep the cleaner version, thresholds, counts, and audit sample beside that manifest. The next chapter uses the same normalized comparison view to identify exact and near duplicates.

## Minimal checkpoint, evidence, and failure diagnosis

~~~bash
.venv/bin/python -m ttt.data \
  --input examples/sample.jsonl \
  --output outputs/cleaning-sample \
  --sharded --tokenizer byte --shard-tokens 256
~~~

This command checks only that the cleaned teaching text can reach the tokenizer. It does not replace language, PII, length, or quality reports upstream. On success, <code>outputs/cleaning-sample/manifest.json</code> should exist. If empty text, out-of-range tokens, or a broken count chain appears, inspect the cleaning output and schema before changing the learning rate.

## Summary and next question

Cleaning rules change the training distribution; “clean” is not a hidden binary label. Save the rule version, thresholds, keep/reject counts, and audit sample for every rule. The next chapter computes exact and near duplicates on this fixed cleaned view, so normalization differences are not mistaken for duplicate differences.
