# 5. Acquiring data and preserving provenance

## The experiment begins before model training

Two students use the same download URL, training code, and random seed, but obtain different validation losses. Eventually they discover that the publisher replaced the archive between their downloads. Both files are called `corpus.jsonl.gz`. Their documents differ. Nothing in the model implementation had to change for the experiment to change.

The previous chapter explained why the data distribution matters. This chapter asks how to turn “these are the data we used” into a statement someone else can check. A download command alone is insufficient. We need to retain the received objects, identify their source and version, and preserve the connection between those objects and the eventual training input.

Think of receiving a laboratory sample. A label helps locate it, a batch number identifies its provenance, and a measurement checks its contents. In a data pipeline, the URL is a location, the release describes an upstream publication, and a digest helps compare the received bytes. Each answers a different question.

We begin with four original records already included in the repository. You can inspect every line without downloading an external corpus. The experiment demonstrates file identity and provenance records; it does not represent a corpus that has already passed a licensing or quality review.

## File bytes and document records are different objects

A file is a sequence of bytes. Each byte contains eight binary digits and can represent an integer from 0 to 255. An encoding such as UTF-8 maps text to bytes. A format such as JSON describes how text represents fields and values. These two layers explain why identical records need not have identical file bytes.

Consider two JSON representations:

~~~json
{"id":"d1","text":"A cat."}
{"id": "d1", "text": "A cat."}
~~~

Saved separately, they contain different bytes because the spaces differ. A JSON parser produces the same `id` and `text` fields from both. **Parsing** means interpreting a format to construct program objects such as dictionaries, lists, or numbers. It does not judge whether the text is useful.

Parsing also needs an ambiguity policy. For `{"text":"first text","text":"second text"}`, Python keeps only the last value by default. Our acquisition reader instead rejects repeated field names within any object, including nested metadata. This differs from checking document IDs across the file: it prevents a parser's last-value rule from silently deciding which source text exists. See [Python's repeated-name behavior](https://docs.python.org/3.12/library/json.html#repeated-names-within-an-object).

JSONL places one JSON object on each nonempty physical line. A newline inside the text field is represented by the JSON escape `\n`, rather than arbitrarily splitting the object across lines. Our teaching parser permits blank lines and skips them. Consequently, appending a blank line changes the file without adding a document. We will use this small change to separate byte identity from record identity.

Compression adds another layer. A downloaded `.gz` archive and the decompressed JSONL are distinct objects. Recompression can change the archive while preserving its decompressed contents. Every digest must identify the layer it describes; comparing an archive digest with a decompressed-file digest has no useful interpretation.

## What a digest establishes

Let \(b\) denote a file's bytes. SHA-256 maps the input to a 256-bit result:

$$
h=\operatorname{SHA256}(b).
$$

We commonly display that result as 64 hexadecimal characters, each representing four bits. This short string is a **digest**, convenient to store in a manifest. Python provides [`hashlib.sha256`](https://docs.python.org/3.12/library/hashlib.html) in its standard library; no training framework is required.

Identical bytes always produce the same digest. Therefore different digests establish that the bytes differ. The converse is not a mathematical proof: distinct inputs can have the same digest, a situation called a collision. There are more possible files than finite digest values. SHA-256 provides a practical integrity check for ordinary experiment files, while a small experiment can also compare the bytes directly.

A digest does not identify the publisher either. Receiving a file and an alleged reference digest from the same incorrect source cannot establish the source's authenticity. A verifiable publication page, fixed release, and trustworthy acquisition channel address provenance; a digest addresses content comparison.

The smallest useful implementation is:

~~~python
import hashlib
from pathlib import Path

data = Path("examples/sample.jsonl").read_bytes()
print(len(data))
print(hashlib.sha256(data).hexdigest())
~~~

`read_bytes()` reads the actual bytes, `len(data)` counts them, and `hexdigest()` returns a printable digest. Reading the complete object into memory is appropriate for this tiny file. For large files, feed successive blocks to `update()` and ensure that the blocks cover the same complete object; hashing does not require loading hundreds of gigabytes into memory at once.

## A local acquisition experiment that preserves its input

From the repository root, using the prepared environment, run:

~~~bash
.venv/bin/python examples/book/acquisition_walkthrough.py \
  --input examples/sample.jsonl \
  --output outputs/book-acquisition-v1
~~~

The script uses only the Python standard library. It requires neither a GPU nor model weights. It validates the input `id` and `text` fields, requires unique document IDs, and copies the original bytes to `raw.jsonl`. It then creates a control copy with one additional newline byte and rereads both saved objects to compute digests, parse records, and verify their contents. It does not alter the source. The default example already ends with a newline, so appending another creates a blank line. For an input without a final newline, the added byte only terminates the last line. Either case adds one byte without adding a document.

| Output | Contents | Question it answers |
|---|---|---|
| `raw.jsonl` | A copy of the received bytes | What object did this run actually receive? |
| `with-extra-newline.jsonl` | A control with one extra newline byte | Must a byte change imply a record change? |
| `source-record.json` | Digests, sizes, record counts, and script identity | How can this observation be checked? |

With the repository's default input, the report should satisfy the following conditions. These are acceptance conditions derived from the input and program; obtain the complete digests from your actual output.

~~~json
{
  "records": 4,
  "byte_counts_differ_by": 1,
  "same_parsed_records": true,
  "different_byte_digest": true,
  "original_preserved": true
}
~~~

`same_parsed_records` compares the complete object lists parsed again from the two saved files, rather than merely counting texts. Changing a document while retaining a count of four cannot pass that comparison. `original_preserved` comes from comparing the source and saved copy against the received bytes; the script also checks that the control contains exactly the additional newline byte. A failed check raises an error instead of producing a report that claims success.

The `source_id` is a local content identifier derived from the input digest. It distinguishes this experiment's input without impersonating an external dataset release. `observed_at_utc` records the observation time and normally changes between runs. `repository_commit` records the current commit in a Git checkout and is `null` when running from an archive without Git metadata. The script's own digest is also recorded because an uncommitted script edit is not described by the commit ID.

Reproduction therefore does not require identical timestamp fields. Compare results determined by the fixed input, processing code, and rules. The script refuses to overwrite an existing output directory. Use `outputs/book-acquisition-v2` for a second run to preserve both records.

## Extending the local record to an external source inventory

The local experiment has established a relationship between an input object, an action, and output objects. External acquisition adds upstream facts. Keep source information and file information as two related layers.

The source layer records the publisher, canonical location, immutable revision or snapshot when available, acquisition time, and location of the terms-review evidence. The file layer records the actual objects received, their sizes and digests, and acquisition outcomes. One source may contain a thousand files; one file may combine multiple original sources. In the latter case, preserve document-level provenance instead of collapsing it into an overly broad file label.

Do not invent missing fields. If a publisher exposes only a changing latest version, record the acquisition time, received object, and verifiable publication information available then. A service's ETag can support caching under that service's documented semantics; it is not universally a SHA-256 content digest.

Our report explicitly sets `terms_review` to `not_performed_by_this_script`. Hashing does not review terms. A real project needs evidence appropriate to the particular source and intended release objects. Raw files, cleaned text, token shards, and model weights are different objects; an ambiguous “data usable” field cannot replace the relevant records. The example fields are not legal conclusions.

## Why retain the original object?

Let \(R\) be the raw object, \(e\) the extractor version, and \(c\) the cleaner version. The resulting documents can be written as

$$
D_{\mathrm{clean}}=C_c(E_e(R)),
$$

where \(E_e\) performs extraction and \(C_c\) performs cleaning. The subscripts emphasize that changing either program can change the documents even when \(R\) is fixed.

Suppose a web page contains prose, navigation, and code blocks. An older extractor retains the prose but drops the code; a newer version restores it. Different outputs do not imply that the downloaded page changed. If only final text remains, a reader cannot locate the loss of code in the original page, extraction stage, or later filtering.

Write each processing stage to a new location and record its input digest, code and configuration versions, output digest, and counts. These records describe a processing relationship, not merely a list of final filenames. They let us trace training tokens back through cleaned documents and extracted text to the received object. Small experiments can copy files. Large projects can use immutable object storage with manifest references. The principle is traceable identity, not unconditional duplication of every intermediate byte.

## Use the correct denominator for acquisition losses

Consider an explicitly **hypothetical** acquisition. Of 1,000 requested files, 10 fail to download. Of the 990 downloaded files, 10 cannot be parsed. The remaining 980 files yield 100,000 records, of which 500 lack the text field.

| Stage | Input unit and count | Failure or exclusion | Output |
|---|---:|---:|---:|
| Download | 1,000 files | 10 files | 990 files |
| Parsing | 990 files | 10 files | 980 parseable files |
| Field validation | 100,000 records | 500 records | 99,500 documents |

It would be wrong to report “520 items removed”: the first two rows count files, whereas the last counts records. Download success is \(990/1000=99\%\). Parseability among received files is \(980/990\approx98.99\%\). Text-field completeness among parsed records is \(99500/100000=99.5\%\). Each fraction answers a different question.

Even these totals can hide bias. If all ten failed downloads belong to a small language source, that source may lose a substantial fraction of its coverage. Break down acquisition outcomes by source, language, or format. Keep accidental operational losses separate from deliberate quality filtering so that the eventual training distribution can be explained.

We can now distinguish changes to bytes, parsing, and processing rules while preserving the acquired object. The next chapter examines the text itself: which transformations merely organize its representation, which can discard information, and how to make every keep or reject decision inspectable.
