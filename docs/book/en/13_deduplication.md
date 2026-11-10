# 7. Deduplication: control implicit sampling weights

## Why repeated text is a modeling problem

Suppose one useful article appears once while a navigation template appears 1,000 times. A sampler that treats every token equally gives the template much more optimization weight. Repetition is therefore more than wasted storage: it changes the empirical distribution that the model is trained to fit.

Duplicates can also cross an evaluation boundary. If a paragraph occurs in training and its mirror occurs in validation, a low validation loss may measure memorization rather than generalization. We must report internal duplicates and evaluation overlap as separate questions.

## Exact deduplication: define the bytes first

Exact deduplication computes a digest for a comparison view and keeps one representative. The comparison view might be the raw HTML, extracted text, NFC-normalized text, or a whitespace-normalized key. Those choices define what “exact” means.

For example, two records that differ only in line endings should usually share a key, while two code snippets whose indentation differs may not. Keep the original training text and create a separate comparison key so that a convenience normalization does not destroy meaningful content.

~~~python
import hashlib
import unicodedata


def exact_key(text):
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
~~~

The function maps a string to a 256-bit digest. It does not decide which source wins, build a disk index, or scale to a corpus larger than memory. A production job should stream keys into a durable index and write a mapping from every removed record to its representative.

## Near duplicates: a worked similarity example

Full-text hashes miss documents that differ by a date, a page header, or a tracking URL. Near deduplication first turns a document into shingles, such as consecutive word or character $n$-grams. For two shingle sets $A$ and $B$, Jaccard similarity is

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

If $A=\{\mathrm{ab},\mathrm{bc},\mathrm{cd}\}$ and $B=\{\mathrm{ab},\mathrm{bc},\mathrm{ce}\}$, the intersection has two elements and the union has four, so $J(A,B)=2/4=0.5$. This measures set overlap. It does not understand that a translation, a paraphrase, or a factual update may have the same meaning.

MinHash and locality-sensitive hashing (LSH) reduce the number of candidate pairs that need full comparison. They are retrieval tools, not a final truth oracle. Save the shingle rule, size, seed, fingerprint configuration, candidate threshold, and the final review rule. A threshold selected on a convenient sample may behave differently for code, mathematics, and short documents.

## Similarity is not transitive

Let A and B pass a similarity threshold, and B and C pass it, while A and C do not. If we form a connected component, all three enter one cluster. That is a valid clustering policy, but it is not a mathematical consequence that A and C are duplicates.

The project must choose a stable representative rule: longest text, earliest publication, most complete metadata, or a source priority. The result must not depend on filesystem traversal order. The mapping should include the removed ID, retained ID, relation type, similarity, and reason so that deletion is explainable.

## Local, global, and evaluation deduplication

Deduplicating inside each source does not remove overlap between sources. A tutorial, a book excerpt, and a web mirror can share paragraphs. Conversely, global deduplication can erase a source's coverage and assign all credit to whichever file is visited first.

At minimum, report three layers separately:

| comparison | question |
|---|---|
| within source | does one publisher repeat its own records? |
| across sources | do different sources contain the same material? |
| train versus validation/test | could evaluation reward training exposure? |

Long documents add another choice: document-level deduplication can miss a copied paragraph. Paragraph or substring deduplication is a new rule with a new version and should not be silently folded into the document report.

## A count and quality report

Suppose 1,000 records enter exact deduplication, 80 share keys with an earlier record, and 50 near-duplicate candidates are reviewed; 30 are merged and 20 are retained as distinct. The report should say `kept=890`, `exact_removed=80`, `near_removed=30`, and `near_reviewed=50`, rather than simply `removed=110`. It should also show source and language retention, cluster-size distribution, and token counts before and after deduplication.

“More removal” is not the same as “better data.” A low threshold can merge genuinely different versions, and a large template cluster can swallow rare technical content. Inspect near-threshold examples and rare-source clusters before adopting a new version.

## Mapping to this repository

The current `ttt/data.py` and `ttt/dataset.py` implement tokenization, manifests, SHA256 checks for token artifacts, and sampling. They do not implement a production MinHash, LSH, or semantic deduplication pipeline. The upstream deduplication job should output a versioned JSONL file and its report, then run:

~~~bash
python -m ttt.data \
  --input data/dedup-v1.jsonl \
  --output data/tokens-dedup-v1 \
  --sharded \
  --tokenizer byte
~~~

Run deduplication before assigning related documents to train and validation. Contamination checks against public benchmarks are a separate comparison and deserve a separate report. The next chapter turns stable duplicate groups into evaluation boundaries.

## Minimal checkpoint, expected report, and failure diagnosis

The production sequence creates a new JSONL and report before tokenization:

~~~bash
python -m ttt.data \
  --input <deduped-jsonl> \
  --output outputs/dedup-v1 \
  --sharded --tokenizer byte --shard-tokens 256
~~~

<code>&lt;deduped-jsonl&gt;</code> is an upstream artifact that the reader must replace; it is not a repository file. The report should contain additive fields such as <code>kept=890</code>, <code>exact_removed=80</code>, <code>near_removed=30</code>, and <code>near_reviewed=50</code>, plus the input digest and rule version. If the counts do not balance, inspect cluster membership and representative selection. A single <code>removed</code> total cannot diagnose which rule deleted content.

## Summary and next question

Exact hashing answers whether bytes match; MinHash/LSH narrows candidates; a review or explicit merge rule decides what to remove. Freeze duplicate groups before splitting and link them to the token manifest. The next chapter turns groups into train/validation boundaries and checks evaluation contamination separately.
