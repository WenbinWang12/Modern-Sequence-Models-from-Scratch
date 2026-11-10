# 9. Tokenizer and vocabulary

## A tokenizer is more than preprocessing

A tokenizer determines sequence length, character coverage, special-token semantics, embedding size, output-head size, and evaluation fairness. The same character corpus can produce different numbers of training positions under different tokenizers.

Before a model sees text, the tokenizer performs two linked operations: it segments the text according to fixed rules and maps each segment to an integer ID in a vocabulary. For text $u$ and tokenizer $\tau$, write

$$
\tau(u)=s=(s_1,\ldots,s_L),
\qquad s_t\in\{0,\ldots,V-1\}.
$$

A token need not be a complete word; it may be a byte, character, or subword. The ID is an index, so its numeric value does not express semantic similarity. The tokenizer therefore fixes the sequence length seen by the model, the set of representable strings, and the boundary semantics of special tokens.

## How the tokenizer relates to the vocabulary

The **vocabulary** is one component produced by a tokenizer. It records which tokens are available and which integer ID represents each token. For example:

~~~text
"low"   -> 1523
"er"    -> 4987
"<eos>" -> 2
~~~

The **tokenizer** is the larger system. It includes:

- a normalizer for Unicode, case, and whitespace;
- a pre-tokenizer for initial boundaries;
- the vocabulary and token-to-ID mapping;
- segmentation rules such as BPE merges or Unigram probabilities;
- a decoder for reconstructing text;
- special-token handling.

The vocabulary is therefore not trained separately and then handed to the tokenizer. BPE training usually learns the vocabulary and merge rules together; Unigram training produces a vocabulary and subword probabilities together. A fixed byte tokenizer can skip statistical training because its vocabulary is specified in advance.

Keep tokenizer training separate from language-model training:

~~~text
text corpus
→ tokenizer training
→ vocabulary + segmentation rules
→ freeze tokenizer
→ token IDs
→ Transformer training of embeddings and model parameters
~~~

Transformer pretraining learns embeddings, attention, MLP, and other neural parameters. It does not automatically change the vocabulary. Changing the vocabulary or segmentation rules changes the model input space, so token shards usually need to be regenerated and embeddings reinitialized or explicitly converted.

## Vocabulary size and parameter count

With vocabulary \(V\) and hidden size \(d\):

$$
E\in\mathbb R^{V\times d},\qquad
\mathrm{logits}\in\mathbb R^{\ldots\times V}.
$$

Tied embeddings use the transpose of \(E\) for `lm_head`; untied embeddings add another \(d\times V\) matrix. The MLP can change its intermediate width, but it must return to \(d\) before the residual and vocabulary projection.

**Tied embeddings** means that the input embedding and the output vocabulary projection share parameters. Input token IDs use

$$
E\in\mathbb{R}^{V\times d}
$$

to look up hidden vectors. At the output, a hidden state \(h\in\mathbb{R}^{d}\) uses the transpose of the same matrix:

$$
z=hE^{\mathsf{T}}\in\mathbb{R}^{V}.
$$

With untied embeddings, the model stores a separate \(W_{\mathrm{out}}\in\mathbb{R}^{d\times V}\) and computes \(z=hW_{\mathrm{out}}\). Tying saves roughly \(Vd\) parameters and places input and output tokens in one representation space, at the cost of not letting the two transformations learn completely independent weights.

For the tiny configuration with $V=259$ and $d=16$, the embedding has 4,144 parameters; an untied output head adds the same number. At $V=32{,}000$, one embedding has 512,000 parameters; at $V=128{,}000$, it has 2,048,000. With $d=768$, increasing the vocabulary from 32K to 128K increases one embedding from about 24.6M to 98.3M parameters. Choosing a tokenizer therefore chooses a substantial part of the parameter and bandwidth budget.

The MLP output interface remains $d$, because it returns to the residual stream. The vocabulary projection then maps $d$ to $V$, so the final logits have shape $[\ldots,V]$. Changing the tied-embedding choice or inserting an additional projection is a new architecture and a new parameter budget.

## How a subword tokenizer is trained

Training a tokenizer is not training a Transformer. It is usually a corpus-statistics procedure that produces a fixed vocabulary, segmentation rules, and token IDs. Once pretraining begins, the tokenizer normally remains fixed.

Before training, fix:

- the tokenizer-training manifest and version;
- Unicode normalization, whitespace handling, and pre-tokenization;
- target vocabulary size \(V\);
- the algorithm: BPE, WordPiece, or Unigram;
- BOS, EOS, PAD, UNK, and their IDs;
- byte fallback, unknown-token, and truncation behavior.

For BPE, the process is:

1. Choose base symbols, such as characters or UTF-8 bytes;
2. count adjacent symbol pairs in the training corpus;
3. merge the most frequent pair into a new token;
4. recount pairs using the new symbols;
5. repeat until the target vocabulary size is reached.

For example, if `l` and `o` often occur together, the procedure may learn `lo`, and later merge `lo` with `w` to form `low`. The trained tokenizer stores the vocabulary and merge order; encoding new text applies these fixed rules.

WordPiece also learns subwords, but uses a likelihood-based criterion and commonly applies greedy longest-match encoding. BERT uses this family of tokenizers.

Unigram starts with a larger candidate set and progressively removes candidates with low contribution. It can evaluate several segmentations of the same text and choose a high-probability one. SentencePiece is a toolkit that supports BPE and Unigram; it is not simply a third merge algorithm alongside them.

After training, save at least:

- the vocabulary and token-to-ID mapping;
- BPE merge rules or Unigram probabilities;
- normalizer and pre-tokenizer configuration;
- special-token strings and IDs;
- the training manifest, code version, and output digest.

Evaluation should measure more than whether a vocabulary file was produced. On independent representative data, measure tokens per byte, tokens per document, language-specific length, code segmentation, unknown or fallback rates, and EOS or special-token frequency. The tokenizer-training corpus should not contain validation or benchmark text; otherwise the segmentation rules themselves can carry contamination.

## How to train a tokenizer in practice

Tokenizer training does not require choosing a Transformer first. The training object is a set of BPE merge rules, Unigram subword probabilities, or WordPiece vocabulary criteria. The Transformer is trained only after the tokenizer is fixed.

### Prepare the corpus

Train on the `train` portion of the pretraining data, using the same Unicode normalization and text-extraction rules as the final pipeline. If the source is JSONL, extract the `text` field into a plain-text file with one document per line. Do not include validation, test, or benchmark text.

Record the corpus manifest and version, document and whitespace boundaries, normalizer, pre-tokenizer, algorithm, target vocabulary size, and special-token strings and IDs.

### Train BPE or Unigram with SentencePiece

SentencePiece can train BPE or Unigram directly from raw text. This command trains BPE; replace `--model_type=bpe` with `unigram` for a Unigram model:

~~~bash
spm_train \
  --input=data/tokenizer/train.txt \
  --model_prefix=outputs/tokenizer-32k \
  --model_type=bpe \
  --vocab_size=32000 \
  --character_coverage=1.0 \
  --pad_id=0 \
  --bos_id=1 \
  --eos_id=2 \
  --unk_id=3
~~~

Save the complete command, SentencePiece version, input-manifest hash, and output digest. For multilingual data, report character coverage, whitespace handling, and Unicode rules explicitly.

### Train byte-level BPE with the repository script

The repository now provides a real tokenizer-generation entry point: [`scripts/train_tokenizer.py`](https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch/blob/main/scripts/train_tokenizer.py). It reads plain text or JSONL, trains BPE or Unigram, and uses `PreTrainedTokenizerFast.save_pretrained` to write a directory that `AutoTokenizer.from_pretrained` can load.

Install the tokenizer extra:

~~~bash
uv sync --extra tokenizer
~~~

Train a 32K byte-level BPE model from the `text` field of a JSONL corpus:

~~~bash
uv run python scripts/train_tokenizer.py \
  --input data/tokenizer/train.jsonl \
  --output outputs/tokenizer-32k \
  --algorithm bpe \
  --vocab-size 32000 \
  --text-key text
~~~

Use `--algorithm unigram` for Unigram. Add `--no-byte-level` when the experiment should use whitespace pre-tokenization. The output directory contains `tokenizer.json`, special-token configuration, and `tokenizer_metadata.json` with input hashes, algorithm, vocabulary size, and special tokens.

The script reads JSONL into temporary plain-text training files, constructs `models.BPE` or `models.Unigram`, creates the corresponding trainer, calls `tokenizer.train(...)`, and saves the result through `PreTrainedTokenizerFast`. It produces tokenizer files, not a Transformer checkpoint.

### Use the Hugging Face Tokenizers API directly

The `tokenizers` library lets the experiment define the normalizer, pre-tokenizer, and trainer explicitly:

~~~python
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(
    vocab_size=32_000,
    special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
)
tokenizer.train(["data/tokenizer/train.txt"], trainer)
tokenizer.save("outputs/tokenizer-32k/tokenizer.json")
~~~

This trains byte-level BPE. Switching to Unigram requires a different model and trainer and a new recorded configuration; changing only the filename or `vocab_size` is not enough.

### Connect the trained tokenizer to the model

Before training the language model, run an independent check:

~~~text
tokenizer files
→ encode/decode round trip
→ token ID range check
→ special-token check
→ tokens/byte and length statistics
→ token shards and manifest
→ set model vocab_size
→ train the dense baseline
~~~

The current repository uses `ttt.data` to load a trained tokenizer and generate token shards; the training entry point is `scripts/train_tokenizer.py`. After training, pass the output directory to `ttt.data` and generate new token shards. Old token IDs cannot be reused with a new vocabulary.

### Where these steps live in the repository

- [`scripts/train_tokenizer.py`](https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch/blob/main/scripts/train_tokenizer.py): train BPE/Unigram from plain text or JSONL and save a Hugging Face tokenizer directory;
- [`ByteTokenizer` and `HuggingFaceTokenizer`](https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch/blob/main/ttt/data.py#L14-L53): byte-tokenizer implementation, loading of prepared Hugging Face tokenizers, and the shared `encode`/`decode` interface;
- [`load_tokenizer` and data conversion entry points](https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch/blob/main/ttt/data.py#L51-L147): select a tokenizer and convert JSONL or text into token files;
- [`prepare_sharded_data`](https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch/blob/main/ttt/dataset.py#L82-L168): write token shards and a manifest with hashes, vocabulary size, and mask metadata;
- [`ExperimentConfig.tokenizer`](https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch/blob/main/ttt/config.py#L17-L24): connect the tokenizer name and vocabulary constraint to training configuration;
- [`generate.py` vocabulary check](https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch/blob/main/ttt/generate.py#L25-L27): prevent a tokenizer vocabulary from exceeding the model output dimension.

The SentencePiece command and `tokenizers` Python example explain the algorithms; the repository's actual generation entry point is [`scripts/train_tokenizer.py`](https://github.com/WenbinWang12/Modern-Sequence-Models-from-Scratch/blob/main/scripts/train_tokenizer.py). `ttt.data` loads its output, validates IDs, and creates shards.

## Byte tokenizer in this repository

The repository `ByteTokenizer` maps UTF-8 byte values 0–255 to IDs 3–258 and reserves 0, 1, and 2 for PAD, BOS, and EOS:

~~~text
byte 0       -> token 3
byte 255     -> token 258
BOS          -> 1
EOS          -> 2
PAD          -> 0
vocab_size   = 259
~~~

It can encode any UTF-8 text without downloading a vocabulary and has no unknown-character case. Its cost is sequence efficiency: one Chinese character normally occupies three UTF-8 bytes, and common English words may also span several positions. It is useful for interface and recovery checks, but is not the final tokenizer choice for a serious Chinese-English pretraining run.

## Special tokens and data boundaries

Special tokens are not ordinary words. If the tokenizer inserts BOS during encoding and the data script inserts it again, the model sees a duplicate marker. If EOS is removed between documents, continuous packing loses an important boundary signal.

The special-token strings, IDs, insertion positions, and loss behavior must therefore be recorded in the tokenizer configuration and data manifest. BOS, EOS, PAD, UNK, and conversation-control tokens are a joint contract between the tokenizer, packing policy, and model. Changing a tokenizer requires checking these IDs and masks again, not only changing `vocab_size`.

## Measure efficiency by source

Let $B$ be the UTF-8 byte count of a text and $T$ its token count. The ratio $T/B$ measures encoding density. It is not a quality score, but it predicts how much character context fits into a fixed sequence length and how many token positions a data budget buys.

At minimum, report:

| Metric | Why it matters |
|---|---|
| tokens/byte | estimates context coverage and data cost |
| tokens/document | exposes short and long document behavior |
| fallback or unknown rate | finds coverage failures |
| special-token frequency | checks templates and boundaries |
| truncation rate | shows whether max length is too small |
| per-language or per-domain groups | prevents aggregate means from hiding bias |

A tokenizer can make English more compact while expanding Chinese or code. Overall averages cannot replace grouped statistics.

## Current code and replacement boundary

The repository exposes `ByteTokenizer` and a Hugging Face path through the tokenizer extra. `HuggingFaceTokenizer` calls `AutoTokenizer.from_pretrained`; the current command line accepts a model name but does not expose a revision argument, and remote tokenizer files are not committed to Git. A production run must record the name, revision, local-cache digest, and file hashes outside the current manifest.

Prepare data with:

~~~bash
python -m ttt.data --input corpus.jsonl --output data/pretrain --sharded --tokenizer byte
python -m ttt.data --input corpus.jsonl --output data/pretrain --sharded --tokenizer meta-llama/Meta-Llama-3-8B
~~~

The three implemented paths have these boundaries:

| Path | BOS | EOS | loss mask |
|---|---|---|---|
| `prepare_data` ordinary NPY | added by default in `encode` | appended per document | no mask file; target positions default to 1 |
| `prepare_sharded_data` ordinary LM | added by default in `encode` | appended per document | BOS is 0; other tokens are 1 |
| `prepare_sharded_data --supervised` | added to the prompt, not repeated for the response | appended after the response | prompt is 0; response and EOS are 1 |

The repository does not train BPE, WordPiece, or Unigram tokenizers. The training discussion above describes how an external tokenizer should be prepared and audited. `prepare_data` and `prepare_sharded_data` append EOS after ordinary text; `ByteTokenizer.encode` adds BOS by default; the unsupervised sharded path masks BOS out of the loss, while the supervised path computes loss only for the response. The Hugging Face path calls `encode(add_special_tokens=False)`, then prepends BOS when configured; it does not add EOS automatically. Replacing a tokenizer requires checking these behaviors against the new template and special-token contract.

The current manifest records the tokenizer name, vocabulary size, input digest, and shard digests. It does not automatically record special-token IDs, normalization, a remote revision, or a tokenizer-file hash; those fields must be added to the higher-level experiment record for production reproducibility.

Changing the tokenizer requires new token shards and a compatible model configuration; changing only vocab_size in a checkpoint is not a conversion.

The next chapter turns the token stream into fixed windows and handles label shifting, EOS boundaries, and loss masks.
