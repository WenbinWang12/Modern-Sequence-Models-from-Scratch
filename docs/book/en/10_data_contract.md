# 3. Pretraining Data: Distribution, Quality, and Reproducibility

This chapter follows the [notation and conventions](../notation.md) used throughout the book. Mathematical positions start at 1; code array indices start at 0.

## Define the two central objects

This chapter first defines two phrases that are often used without explanation: **training data distribution** and **prediction loss**. Cleaning, tokenization, packing, and manifests only fit together once these objects are explicit. We denote the tokenized sequence by $x$ and the model parameters by $\theta$ throughout the rest of the chapter.

Let one raw record be

$$
r_i=(u_i,m_i),
$$

where $u_i$ is text and $m_i$ contains metadata such as source, license, language, time, and document group. Let $P_{\mathrm{raw}}$ denote the distribution of raw records. For a fixed collection of raw files, it can be the empirical distribution

$$
P_{\mathrm{raw}}=\frac{1}{N}\sum_{i=1}^{N}\delta_{r_i},
$$

where $\delta$ is a point mass at one record. A collection of raw files is not yet the final training distribution. After source selection, cleaning, deduplication, and splitting, we obtain a training collection

$$
C_{\mathrm{train}}=\{r_1,\ldots,r_n\}.
$$

If cleaning, splitting, and source mixing form a transformation $C$, the document distribution can be written as

$$
P_{\mathrm{doc}}=C_{\#}P_{\mathrm{raw}},
$$

where $\#$ denotes pushing a distribution through a transformation. For a fixed document collection, if the sampler selects record $i$ with weight $w_i$, the empirical document distribution is

$$
P_{\mathrm{doc}}(r_i)=
\frac{w_i}{\sum_{j=1}^{n}w_j}.
$$

Uniform document sampling, length-weighted sampling, and source-level sampling therefore define different distributions. The number of files alone does not say what the model sees.

Document distribution and training-position distribution are also different objects. Uniformly selecting documents can still give longer documents more loss positions. Sampling token starts uniformly produces another weighting over short documents, long documents, and boundaries. Document count, token count, and supervised loss positions are not interchangeable statistics.

A tokenizer $\tau$ maps document text to a token sequence and pushes the document distribution into a token-sequence distribution:

$$
\tau(u_i)=z^{(i)}=(z^{(i)}_0,\ldots,z^{(i)}_{L_i}),
\qquad z^{(i)}_t\in\{0,\ldots,V-1\},
$$

where $V$ is the vocabulary size. A window sampler then applies sequence length, packing, EOS, and document-boundary rules. Let this operation be $\Pi$. It induces:

$$
P_{\mathrm{tok}}=\tau_{\#}P_{\mathrm{doc}},
\qquad
P_{\mathrm{win}}=\Pi_{\#}P_{\mathrm{tok}}.
$$

For document-local sampling, this can be written explicitly as:

$$
P_{\mathrm{win}}(X,Y,M)
=
\sum_i P_{\mathrm{doc}}(r_i)
\,P_{\Pi}(X,Y,M\mid \tau(u_i)).
$$

The complete chain can be summarized as:

$$
P_{\mathrm{raw}}
\xrightarrow{\,C\,}P_{\mathrm{doc}}
\xrightarrow{\,\tau\,}P_{\mathrm{tok}}
\xrightarrow{\,\Pi+\text{pack/shift/mask}\,}P_{\mathrm{win}}(X,Y,M)
\xrightarrow{\,p_\theta\,}\mathcal{L}.
$$

For a concatenated logical stream, $\Pi$ acts on the packed stream rather than one document, but the same idea applies: the window distribution is induced by the complete data protocol. In practice, $P_{\mathrm{win}}$ is the object closest to the “training data distribution”. It depends on source weights, cleaning, deduplication, tokenization, packing, window length, and masks.

## What exactly is prediction loss?

Let a window contain consecutive tokens

$$
z=(z_0,z_1,\ldots,z_L).
$$

A causal language model uses earlier tokens as input and the next token as target:

$$
X=(z_0,\ldots,z_{L-1}),
\qquad
Y=(z_1,\ldots,z_L).
$$

For model parameters $\theta$, the loss at position $t$ is the negative log probability assigned to the observed target token:

$$
\ell_t(\theta)=-\log p_\theta(y_t\mid x_{1:t}).
$$

If the model emits logits $a_\theta(x_{1:t})$ at position $t$, then

$$
p_\theta(v\mid x_{1:t})
=
\operatorname{softmax}(a_\theta(x_{1:t}))_v.
$$

If $M=(m_1,\ldots,m_L)$ is a loss mask, with $m_t=1$ for supervised positions and $m_t=0$ for ignored positions, the average window loss is

$$
\ell(\theta;X,Y,M)
=
-\frac{\sum_{t=1}^{L}m_t\log p_\theta(y_t\mid x_{1:t})}
{\sum_{t=1}^{L}m_t}.
$$

The population objective induced by the pipeline is

$$
\mathcal{L}(\theta;P_{\mathrm{win}})
=
\mathbb{E}_{(X,Y,M)\sim P_{\mathrm{win}}}
\left[\ell(\theta;X,Y,M)\right].
$$

A finite run sees $N$ sampled windows. If the loss is averaged over all unmasked tokens, the empirical objective is

$$
\widehat{\mathcal{L}}_N(\theta)
=
\frac{\sum_{k=1}^{N}\sum_{t=1}^{L_k}
m^{(k)}_t\ell^{(k)}_t(\theta)}
{\sum_{k=1}^{N}\sum_{t=1}^{L_k}m^{(k)}_t}.
$$

Minibatch optimization approximates gradients of this empirical objective. “The model learns from the data” therefore means that its parameters are updated so that average next-token negative log likelihood decreases on windows produced by the declared sampler.

Validation loss uses the same definition with an independent $P_{\mathrm{val}}$ in place of $P_{\mathrm{win}}$. Training loss asks whether the model fits the training distribution; validation loss asks how it predicts on another declared distribution. If validation contains near duplicates of training, the second question has not been measured reliably.

This is why the chapter treats data as part of the objective. Changing $P_{\mathrm{doc}}$, $\tau$, $\Pi$, or $M$ changes the target function even when architecture and optimizer remain unchanged.

## How a raw document becomes a training signal

The data pipeline is not just a file transfer to the GPU. It is a sequence of transformations that changes the mathematical object being optimized:

~~~mermaid
flowchart LR
    A[Raw file] --> B[Document record and metadata]
    B --> C[Clean, deduplicate, split]
    C --> D[Tokenizer: text → token IDs]
    D --> E[Packing and window sampling]
    E --> F[input_ids / labels / loss_mask]
    F --> G[Cross-entropy and gradients]
    G --> H[Parameter update]
~~~

For an illustrative token sequence:

| Object | Value |
|---|---|
| Continuous tokens | `[10, 11, 12, 2]` |
| `input_ids` | `[10, 11, 12]` |
| `labels` | `[11, 12, 2]` |
| `loss_mask` | `[1, 1, 1]` |

The window loss is

$$
-\frac{1}{3}
\left[
\log p_\theta(11\mid 10)
+\log p_\theta(12\mid 10,11)
+\log p_\theta(2\mid 10,11,12)
\right].
$$

Masking the final EOS changes both the numerator and denominator. Crossing a document boundary changes the available context. A tokenizer that produces more tokens increases the number of prediction positions and the cost. The training signal is therefore not raw text itself; it is the $(X,Y,M)$ object constructed by these rules.

## What each transformation changes

**Source selection changes support.** Web text, code, books, and papers expose different vocabularies, lengths, and structures. A model receives signal only from the regions sampled into training.

**Cleaning changes the sample set.** Removing navigation pages, corrupted text, and parser failures changes which records remain in $C_{\mathrm{train}}$. Over-filtering can remove code, mathematics, minority languages, or long documents.

**Deduplication changes weights.** Repeating one document one hundred times increases its probability under $P_{\mathrm{doc}}$. Near duplicates across training and validation also destroy evaluation independence.

**Splitting changes the evaluation object.** Document-group splits and contamination checks determine whether validation windows represent unseen content. A contaminated benchmark can turn low loss or high accuracy into memorization evidence.

**Tokenization changes the sample space and cost.** The same characters can produce different token sequences. Vocabulary size affects the embedding, output head, sequence length, attention, KV cache, and throughput.

**Packing and masks change conditional probabilities.** Cross-document packing, EOS retention, and supervised positions all change $P_{\mathrm{win}}$ or the window loss. Packing is part of the learning protocol, not only a storage optimization.

## A concrete mixture example

| Source | Documents | Average length | Total tokens |
|---|---:|---:|---:|
| Web | 900,000 | 1,000 | 900M |
| Code | 50,000 | 2,000 | 100M |
| Books | 10,000 | 10,000 | 100M |

Concatenation makes web text about 82% of the token stream. Uniform document sampling increases the probability of book documents. Both can be valid, but they define different $P_{\mathrm{doc}}$ and therefore different objectives.

If 30% of web tokens are boilerplate duplicates, cleaning may reduce effective web tokens to 630M. If the mixture is not recomputed, the effective weights of code and books rise automatically. Every stage must report its own denominator rather than only a final token total.

The current `BatchStream` samples windows from one logical token stream. It does not implement source-level mixture scheduling. The later mixture chapter defines that capability without claiming it already exists here.

## The pretraining data contract

Before a long run, the pipeline should answer:

1. **Is the source identifiable?** Can we locate the release, retrieval time, and license?
2. **Is quality explainable?** Does every filter have a version, threshold, count, and sample audit?
3. **Are splits trustworthy?** Are train, validation, and test separated by document or project boundary, with near-duplicate and contamination checks?
4. **Is representation consistent?** Do tokenizer, special tokens, packing, label shift, and masks agree with the model configuration?
5. **Is the version recoverable?** Can manifest, shard hashes, sampler state, and checkpoint identify the next batch?

Readable data is not necessarily acceptable data. Each requirement needs a saved evidence object and a later preflight check.

## Data objects and version boundaries

**Raw files** are immutable downloads or snapshots such as archives, HTML, JSONL, or repository states.

**Document records** contain extracted text and metadata for cleaning, deduplication, splitting, and boundary management.

**Token shards** contain integer sequences and supervision masks plus vocabulary, tokenizer version, dtype, shard count, and SHA256 metadata.

**Training windows** are fixed-length sequences sampled from the token stream and converted into `input_ids`, `labels`, `loss_mask`, and position state.

These objects must not share one vague “data version”. A fixed raw file can produce a new document version after cleaning changes; a fixed document set can produce a new token version after tokenizer changes; fixed shards can produce a new window protocol after sequence length or packing changes.

## What the repository currently implements

`ttt/data.py` provides a byte tokenizer and JSONL-to-token conversion. It can write one-dimensional NPY data or sharded token and mask files with a manifest.

`ttt/dataset.py` validates manifest version, `vocab_size`, paths, SHA256, token ranges, mask shapes, and mask values, then reads shards with memory mapping. `BatchStream` uses a NumPy generator to sample windows and stores RNG state, data digest, sequence length, batch size, and seed.

~~~bash
.venv/bin/python -m ttt.data \
  --input examples/book/clean_corpus.jsonl \
  --output outputs/book-data \
  --sharded \
  --tokenizer byte \
  --shard-tokens 256
~~~

`shard-tokens` is a storage-shard size, not the model sequence length. The current window can cross EOS and document boundaries, and the model has no general “reset attention at EOS” interface. The current loader should therefore be described as continuous-stream packing rather than document-isolated packing.

The manifest also lacks a complete document-offset sidecar, so arbitrary token-to-document tracing remains a requirement for a larger data pipeline. Separating implemented behavior from design requirements is part of the data standard.

## The route through the remaining data chapters

- `11_data_acquisition.md`: source versions, licenses, downloads, and provenance;
- `12_data_cleaning.md`: normalization, format, language, length, and quality filters;
- `13_deduplication.md`: exact/near deduplication and its effect on sampling weights;
- `14_split_and_contamination.md`: train, validation, test, and contamination boundaries;
- `15_tokenizer.md`: tokenizer, vocabulary, and parameter dimensions;
- `16_packing_and_masks.md`: packing, label shift, document boundaries, and loss masks;
- `17_dataset_and_manifest.md`: shards, manifests, mmap, and recoverable sampling;
- `18_data_mixture.md`: source mixtures and deterministic sampling;
- `19_data_preflight.md`: hard gates before training;
- `20_baseline_model.md`: whether these inputs actually support learning in a dense causal model.

From the next chapter onward, “data” is no longer a vague input name. We trace each object, how it changes the training distribution, and which evidence allows the next run to check it.
