---
hide:
  - navigation
---

# TTT-E2E technical report: from-scratch training, matched controls and long-context evaluation

**Version 1.0 · 18 November 2026**

!!! warning "Draft: 6 values are not in yet"
    This page was rendered from experiments that are still running. Places marked «?» or carrying a
    placeholder note fill in automatically once the corresponding run finishes.

## Abstract

To be written.


## 1 Introduction

To be written.


## 2 Method

### 2.1 Notation

A token sequence is \(x_1,\ldots,x_{L+1}\). At position \(t\) the model reads \(x_{1:t}\) and predicts \(x_{t+1}\); \(L\) is the number of predicted positions in one training window. The model has \(n_\ell\) residual blocks, hidden width \(d\), MLP inner width \(d_{\mathrm{ff}}\) and vocabulary size \(V\). Each sliding-window attention layer sees the most recent \(w\) positions.

Parameters are split in two. **Slow weights** \(\theta\) are fixed once training ends and shared by every sequence. **Fast weights** \(\phi\) belong to the sequence currently being read and change as it is read. The sequence is cut into update chunks of \(C\) tokens; chunk \(c\) covers positions \(\mathcal T_c=\{(c-1)C+1,\ldots,cC\}\). \(\phi_{c-1}\) denotes the fast weights used to predict chunk \(c\), and \(\phi_0\) is their initial value.

### 2.2 Architecture

The backbone is a pre-norm, decoder-only Transformer. Each block does two things in order:

$$
u=h+\operatorname{Attn}\bigl(\operatorname{RMSNorm}(h)\bigr),\qquad
h'=u+\operatorname{MLP}\bigl(\operatorname{RMSNorm}(u)\bigr).
$$

Attention uses rotary position embeddings (RoPE, base \(5\times10^5\)) and QK normalisation: queries and keys are RMS-normalised per head before the dot product, and scores are divided by \(\sqrt{d_h}\). The MLP is a bias-free SwiGLU, \(\operatorname{MLP}(v)=W_{\mathrm{down}}\bigl(\operatorname{SiLU}(W_{\mathrm{gate}}v)\odot W_{\mathrm{up}}v\bigr)\). Input embeddings and the output projection share one matrix.

TTT-E2E changes only the last \(n_s\) blocks (the **suffix**; the other \(n_\ell-n_s\) blocks are the **prefix**). Each suffix block gains a second MLP of the same shape, MLP′, between attention and the original MLP. Its three matrices are the fast weights:

$$
u=h+\operatorname{Attn}(\operatorname{RMSNorm}(h)),\qquad
v=u+\operatorname{MLP}'_{\phi}(\operatorname{RMSNorm}(u)),\qquad
h'=v+\operatorname{MLP}(\operatorname{RMSNorm}(v)).
$$

The original MLP stays a slow weight, so knowledge from pre-training keeps a home that test-time updates cannot overwrite. This choice, and updating only the last quarter of the blocks, follow the original paper[^ttte2e]. Embeddings, normalisation and attention are never updated at test time.

<figure markdown="span" class="report-figure">
  ![Forward pass and the two loops of TTT-E2E](assets/report/fig_method.light.svg#only-light){ width="100%" }
  ![Forward pass and the two loops of TTT-E2E](assets/report/fig_method.dark.svg#only-dark){ width="100%" }
  <figcaption>Figure 1. Forward pass of TTT-E2E. The prefix runs once over the whole sequence. The suffix runs chunk by chunk: each chunk is predicted with the current fast weights, and its loss then moves the fast weights one step. The outer loop sums the chunk losses and differentiates through the whole chain of updates.</figcaption>
</figure>

### 2.3 Inner loop: predict first, then update

The prefix does not depend on the fast weights, so it is computed once for the whole sequence, giving features \(g_t=f^{\mathrm{pre}}_\theta(x_{1:t})\). The suffix is computed chunk by chunk. After chunk \(c\) has been read, its mean loss is

$$
\ell_c(\theta,\phi_{c-1})=\frac1C\sum_{t\in\mathcal T_c}-\log p\bigl(x_{t+1}\mid x_{1:t};\theta,\phi_{c-1}\bigr).
$$

Chunk \(c\) is predicted with \(\phi_{c-1}\), which has only seen the targets of the first \(c-1\) chunks. Only once every target in chunk \(c\) has been observed do the fast weights take one gradient step:

$$
g_c=\nabla_{\phi}\,\ell_c(\theta,\phi)\Big|_{\phi=\phi_{c-1}},\qquad
\hat g_c=\min\!\Bigl(1,\frac{\tau}{\lVert g_c\rVert_2}\Bigr)g_c,\qquad
\phi_c=\phi_{c-1}-\eta\,\hat g_c .
$$

\(\eta\) is the inner learning rate and \(\tau\) caps the gradient norm taken over all fast weights together. The updated \(\phi_c\) is used only from chunk \(c+1\) onwards, so no token is ever scored by weights that have seen it, and evaluation has no label leakage. A trailing partial chunk is scored but triggers no update.

Suffix attention keeps a KV cache: chunk \(c\) sees the keys and values of earlier chunks inside the window. They were computed with the fast weights of their time and are not recomputed after an update.

### 2.4 Outer loop: learning to be updated

The training objective is the mean loss *after* reading and updating in this way:

$$
\mathcal L(\theta)=\frac{1}{n_c}\sum_{c=1}^{n_c}\ell_c\bigl(\theta,\phi_{c-1}(\theta)\bigr),\qquad \phi_0\subset\theta .
$$

\(\phi_0\) is part of the slow weights: the initial value of the fast weights is learned. Because \(\phi_{c-1}\) is a function of \(\theta\), the gradient of \(\mathcal L\) flows along the update chain as well as through ordinary back-propagation:

$$
\frac{d\phi_c}{d\theta}=\frac{d\phi_{c-1}}{d\theta}-\eta\,\frac{d\hat g_c}{d\theta},
$$

where \(d\hat g_c/d\theta\) contains second derivatives of \(\ell_c\). In code this is one more differentiation of the whole forward function, inner loop included (gradients of gradients). The slow weights therefore learn two things: how to predict the next token, and how to make the suffix predict better once it has been updated.

The **first-order** variant in the ablations treats \(\hat g_c\) as a constant, \(d\hat g_c/d\theta=0\). The forward pass is unchanged and training is cheaper, but the outer loop no longer knows how the slow weights shape the update.

### 2.5 Inference

Inference runs the same inner loop without the outer loop. A session's state has three parts: the current fast weights \(\phi\) (fixed size, independent of how much has been read), the KV cache of the last \(w\) positions in each layer, and the features and targets of the chunk still being filled. During generation one update runs each time \(C\) tokens have accumulated and the next token is known. No session modifies the released slow weights, and sessions do not share fast weights.

### 2.6 Implementation notes

The implementation is in JAX. The following points bear directly on how far the results can be trusted.

- **Outer-loop gradients.** The sequence loss runs the chunks in a `lax.scan`, calling `jax.value_and_grad` at each step to obtain \(g_c\); the whole function is then differentiated again. The chunk scan is two-level with checkpointing, so the backward pass keeps about \(\sqrt{n_c}\) snapshots of the fast weights and recomputes the rest.
- **Sliding-window attention.** During training the prefix applies windowed attention to the whole sequence. We cut the sequence into blocks of \(w\) positions and let each block attend densely to itself and the previous block, all blocks in parallel. In float64 it differs from the per-position masked reference by at most \(7\times10^{-7}\) (forward values and gradients with respect to queries and keys). It is a different algorithm for the same function, not an approximation.
- **Cross-entropy.** The vocabulary projection and log-softmax are computed in token blocks; the values equal the one-shot computation.
- **Precision.** Parameters are float32; matrix products use bfloat16; normalisation, softmax and the loss use float32.
- **Correctness tests.** The repository's tests cover causality (changing future tokens leaves earlier outputs unchanged), agreement of the outer-loop gradient with finite differences, agreement of streaming inference with the full forward pass, checkpoint restore, and the property that the first-order variant keeps the forward pass and changes the gradient. All pass on CPU.

### 2.7 Complexity

Let \(c_{\mathrm{lin}}\) be the cost of one token passing through the linear layers of one block. Full attention pays, per token, an attention cost proportional to the length already read, and its KV cache grows linearly. Sliding-window attention and TTT-E2E pay an attention cost that depends on \(w\) only. On top of that, TTT-E2E pays two more terms per token: the forward pass of MLP′ in the suffix, and one backward pass through the suffix per chunk (roughly twice a suffix forward pass, plus the output layer). Neither depends on the length already read, so inference cost and state size are constant in context length. The price is the second-order training pass through the inner loop. Measured numbers are in Section 4.10.


## 3 Experimental setup

### 3.1 Four models

The main experiment trains four models with 12 blocks and width 768. They share the tokenizer, the data, the exact batch order, the optimiser and the number of training tokens. They differ only in how they use context.

| Model | Attention | Blocks with MLP′ | Test-time updates | Parameters | Non-embedding parameters |
|:--|:--|:--|:--|--:|--:|
| Full attention | all 4,096 positions | – | – | 123.57M | 84.97M |
| Sliding window | last \(w=512\) positions | – | – | 123.57M | 84.97M |
| No-update control | last \(w=512\) positions | last 3 | none | 137.73M | 99.13M |
| TTT-E2E | last \(w=512\) positions | last 3 | one step every \(C=256\) tokens | 137.73M | 99.13M |

Each control answers one question.

- **Full attention** is the reference in which everything in context is visible without loss. Its cost grows linearly with context; this report treats it as the quality target, not as a cost competitor.
- **Sliding window** is the baseline whose cost is of the same order as TTT-E2E. Each layer sees only the last 512 positions directly; older information can arrive only by being relayed layer to layer (at most about 6,100 positions back with 12 layers), and the model has no dedicated long-range memory.
- **No-update control** has the same architecture and parameter count as TTT-E2E (both carry 14.16M more parameters in MLP′ than the first two) and runs no inner loop in training or inference. It separates the effect of the extra parameters from the effect of test-time updates.
- At evaluation the trained TTT-E2E model is also run with **updates off**, to see how the same slow weights behave without updating.

Everything else is identical across the four: \(d=768\), 12 attention heads, \(d_{\mathrm{ff}}=2048\), tied input and output embeddings, RoPE base \(5\times10^5\), QK normalisation.

### 3.2 Data

| Corpus | Use | Documents | Tokens | Bytes/token | Source and licence |
|:--|:--|--:|--:|--:|:--|
| FineWeb-Edu `sample/10BT`, shards 000–002 | training | 2,182,000 | 2.25B | 4.62 | HuggingFaceFW/fineweb-edu, revision `87f0914`, ODC-By 1.0[^fineweb] |
| PG-19 train | training | 28,602 | 3.07B | 3.73 | Project Gutenberg books released by DeepMind, file lists at revision `4d28bd7`, Apache-2.0[^pg19] |
| PG-19 validation | validation during training, ablations | 50 | 4.64M | 3.75 | as above, official split |
| PG-19 test | final evaluation | 100 | 10.58M | 3.82 | as above, official split |
| FineWeb-Edu `sample/10BT`, first 20,000 documents of shard 013 | final evaluation | 19,635 | 20.37M | 4.62 | as FineWeb-Edu; 365 documents whose full text also occurs in the training shards were removed |

The two corpora complement each other. FineWeb-Edu is web text filtered by a quality classifier; its documents are short (about 1,000 tokens on average) and it drives general language ability. Each PG-19 document is a whole book (about 100,000 tokens on average), so a 4,096-token window almost always lies inside one book. It is the data that actually needs long context.

**Tokenizer.** All models use the GPT-2 BPE vocabulary (\(V=50{,}257\)). Each document is encoded as `<|endoftext|>` followed by its text, and documents are concatenated into one token stream. We use an existing vocabulary rather than training one so that losses can be compared directly with public models that share it.

**Mixture and sampling.** Each training example first picks a source with probability 50%, then draws a window of \(L+1\) tokens uniformly at random from that source's token stream, with replacement. Windows may cross document boundaries. Replaying the sampler with the training seed gives the realised shares and coverage:

| Source | Corpus tokens | Target share | Windows drawn | Realised share | Share of corpus seen | Seen blocks drawn more than once |
|:--|--:|--:|--:|--:|--:|--:|
| fineweb_edu | 2.25B | 50% | 96,316 | 50.16% | 19.7% | 10.5% |
| pg19 | 3.07B | 50% | 95,684 | 49.84% | 14.8% | 7.8% |

Training touches only a small part of each corpus, and the great majority of training windows are seen once.

The four main models share this one sampler: instantiated from each of their own configurations, the tokens and masks of the first 20 batches are byte-identical (digest `bea8b574b5ba…`), as are the seed, step count, batch size, sequence length, learning rate and data fingerprint. Any difference between the models in Section 4 therefore comes from architecture and test-time updates, and cannot come from the data or its order.

### 3.3 Training

| Item | Value |
|:--|:--|
| Sequence length \(L\) / batch | 4,096 / 16 sequences (65,536 tokens per step) |
| Steps / tokens | 12,000 / 786,432,000 |
| Optimiser | AdamW, \(\beta_1=0.9\), \(\beta_2=0.95\), weight decay 0.1 (matrices only) |
| Learning rate | peak \(6\times10^{-4}\); linear warm-up over 480 steps, then cosine decay to 0 |
| Outer gradient clipping | global norm 1.0 |
| Inner loop (TTT-E2E only) | \(\eta=1.0\), \(\tau=1.0\); \(\eta\) rises linearly from 0.1 to 1.0 over the first 1,200 steps |
| Precision | float32 parameters, bfloat16 matrix products |
| Seed | 0 (same initialisation seed and data order for all four models) |
| Hardware | one NVIDIA H200 (143,771 MiB) per model, no multi-GPU training |

No hyper-parameter was tuned for any single model: the four share the table above, and the inner-loop \(\eta\) and \(\tau\) are the implementation defaults. The number of training tokens is about 31% of the original paper's basic recipe for its 125M model (2.5 billion tokens), bounded by one night on two GPUs. This is the main scale limitation of the report.

### 3.4 Evaluation protocol

**Held-out language modelling.** Each PG-19 test book is cut from its start into non-overlapping 4,096-token windows (a trailing partial window is dropped): «?» windows and «?» predicted positions. Every window starts from an empty state: the KV cache is cleared and the fast weights return to \(\phi_0\). The loss at position \(t\) of a window is therefore the loss with exactly \(t\) tokens of context, and binning by position gives loss as a function of context length. FineWeb-Edu held-out documents are short, so they are concatenated as in training and the first 2,048 windows are used. We report mean negative log-likelihood per token (NLL, in nats), perplexity and bits per byte.

**Statistics.** All four models are evaluated on the same windows, so differences between models are estimated as paired differences, with 95% confidence intervals from 2,000 bootstrap resamples over books for PG-19 and over windows for FineWeb-Edu. The intervals reflect sampling error of the evaluation set. They do not include the randomness of retraining, which Section 4.9 estimates separately with multiple seeds.

**Beyond the training length.** The window is lengthened to 32,768 tokens, eight times the training length, with the protocol otherwise unchanged and no fine-tuning.

**Public benchmarks.** Every task is zero-shot and scored by log-likelihood: for multiple choice, \(\log p(\text{choice}\mid\text{context})\) is computed for each choice and the largest wins. Prompt templates, the way choices are attached and length normalisation match the task of the same name in lm-evaluation-harness[^lmeval]. To confirm this we evaluated GPT-2 with our script and with lm-evaluation-harness and compared them metric by metric (Section 4.7). No few-shot examples are used, and no training, prompt search or checkpoint selection targets any benchmark; every model is evaluated at its final checkpoint only.

**Public reference models.** To anchor absolute levels, a set of public models of similar size is evaluated with the same script. Their training data exceeds ours by one to three orders of magnitude; these numbers locate our models and are not a budget-matched comparison.


## 4 Results

### 4.1 Training

<figure markdown="span" class="report-figure">
  ![Training loss and held-out loss against training tokens](assets/report/fig_training.light.svg#only-light)
  ![Training loss and held-out loss against training tokens](assets/report/fig_training.dark.svg#only-dark)
  <figcaption>Figure 2. Training. Left: training loss (exponential moving average). Centre and right: loss of the checkpoints saved every 1,000 steps on PG-19 validation (first 5 windows of each book, 250 in total) and on FineWeb-Edu held-out (256 windows). All four models read exactly the same batch at every step.</figcaption>
</figure>

To be written.


### 4.2 Main result: held-out loss at 4K context

<div class="report-table" markdown>

*(draft: table `main` has no data yet)*

</div>

Table 1. Loss of the final checkpoints on the two held-out sets (lower is better). Differences are paired differences from full attention on the same windows, with 95% confidence intervals in brackets. Bits per byte use the corpus-average bytes per token.

To be written.


### 4.3 Loss as a function of context length

<figure markdown="span" class="report-figure">
  ![Loss by context length on PG-19 test](assets/report/fig_context_4k.light.svg#only-light)
  ![Loss by context length on PG-19 test](assets/report/fig_context_4k.dark.svg#only-dark)
  <figcaption>Figure 3. Loss on PG-19 test against the number of tokens already read in the window (means over 64 positions). Left: loss of each model. Right: paired difference from full attention; bands are 95% confidence intervals from resampling books. The sliding window is 512 tokens and TTT-E2E updates every 256 tokens.</figcaption>
</figure>

<div class="report-table" markdown>

*(draft: table `ranges` has no data yet)*

</div>

Table 2. Loss on PG-19 test by range of context position, and paired differences between models (95% confidence intervals).

To be written.


### 4.4 Where the gain comes from

To be written.


### 4.5 Beyond the training length: 32K context

<figure markdown="span" class="report-figure">
  ![Loss beyond the training length](assets/report/fig_context_32k.light.svg#only-light)
  ![Loss beyond the training length](assets/report/fig_context_32k.dark.svg#only-dark)
  <figcaption>Figure 4. Loss on PG-19 test when the evaluation window is lengthened to 32,768 tokens, eight times the training length (means over 256 positions; the vertical line marks the training length). The dashed line is full attention extrapolated with no truncation; “last 4K only” restricts the same full-attention model to the range it saw in training. The right panel shows paired differences from sliding-window attention.</figcaption>
</figure>

<div class="report-table" markdown>

*(draft: table `long` has no data yet)*

</div>

Table 3. Loss by position range inside 32K windows, and paired differences (95% confidence intervals). Every model was trained at length 4,096 only.

To be written.


### 4.6 What compression loses: verbatim recall

<figure markdown="span" class="report-figure">
  ![Verbatim recall probe](assets/report/fig_recall.light.svg#only-light){ width="70%" }
  ![Verbatim recall probe](assets/report/fig_recall.dark.svg#only-dark){ width="70%" }
  <figcaption>Figure 5. Verbatim recall probe. Tokens 128–191 of a window are copied to a later position. The vertical axis is how much lower the loss on the middle 32 tokens of the repeat is than on the unmodified text; larger values mean the model makes more use of text it has already seen.</figcaption>
</figure>

<div class="report-table" markdown>

*(draft: table `recall` has no data yet)*

</div>

Table 4. Loss reduction on the repeated span (nats, 95% confidence intervals in brackets; first 4 windows of each PG-19 test book, 400 in total).

<div class="report-table" markdown>

| Model |
|:--|

</div>

Table 5. Needle in a haystack: one sentence, “One of the special magic numbers for &lt;key&gt; is: &lt;7-digit number&gt;.”, is hidden in the haystack text and the model is then asked for that key's number. The table reports how much lower the answer's log-likelihood per token is than when the needle is replaced by an unrelated key–value pair (nats; larger means the needle was used more). The haystack is PG-19 test book text, the needle is placed at ten evenly spaced depths, and the prompt and needle wording come from RULER.

To be written.


### 4.7 Public benchmarks and public models

<div class="report-table" markdown>

| Model | Params | HellaSwag | PIQA | ARC-E | ARC-C | WinoGrande | SciQ | OBQA | BLiMP | MMLU | LAMBADA | Mean |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Sliding window | 124M | 27.1 | 57.9 | 38.2 | 22.6 | 52.2 | 70.1 | 25.4 | 76.0 | 23.0 | 17.7 | 41.0 |
| 95% CI half-width |  | ±0.9 | ±2.3 | ±2.0 | ±2.4 | ±2.7 | ±2.8 | ±3.8 | ±0.3 | ±0.7 | ±1.0 |  |

</div>

Table 6. Zero-shot accuracy (%) of the four models in this report. HellaSwag, PIQA, ARC and OpenBookQA report accuracy normalised by choice length; the others report plain accuracy. “Mean” is the arithmetic mean of the ten tasks.

<div class="report-table" markdown>

| Model | Hugging Face repository | Parameters | Max context | Training data (from model card or paper) |
|:--|:--|--:|--:|:--|
| DistilGPT-2[^distil] | `distilbert/distilgpt2` | 81.9M | 1,024 | OpenWebText (38 GB of text), distilled from GPT-2; training tokens not stated |
| GPT-2[^gpt2] | `openai-community/gpt2` | 124.4M | 1,024 | WebText (40 GB of text); training tokens not stated |
| GPT-2 Medium[^gpt2] | `openai-community/gpt2-medium` | 354.8M | 1,024 | WebText (40 GB of text); training tokens not stated |
| Pythia-70M[^pythia] | `EleutherAI/pythia-70m` | 70.4M | 2,048 | The Pile; about 300B tokens seen in training |
| Pythia-160M[^pythia] | `EleutherAI/pythia-160m` | 162.3M | 2,048 | The Pile; about 300B tokens seen in training |
| GPT-Neo-125M[^gptneo] | `EleutherAI/gpt-neo-125m` | 125.2M | 2,048 | The Pile; 300B tokens |
| OPT-125M[^opt] | `facebook/opt-125m` | 125.2M | 2,048 | a 180B-token mixed corpus (800 GB); training tokens not stated explicitly |
| SmolLM-135M[^smollm2] | `HuggingFaceTB/SmolLM-135M` | 134.5M | 2,048 | Cosmopedia v2, Python-Edu, FineWeb-Edu; 600B tokens |
| SmolLM2-135M[^smollm2] | `HuggingFaceTB/SmolLM2-135M` | 134.5M | 8,192 | FineWeb-Edu, DCLM, The Stack and others; 2T tokens |

</div>

Table 7. The public models used for comparison. The training-data column is taken from each model card or paper (checked November 2026); their training budgets exceed ours by one to three orders of magnitude.

<div class="report-table" markdown>

| Model | Params | HellaSwag | PIQA | ARC-E | ARC-C | WinoGrande | SciQ | OBQA | BLiMP | MMLU | LAMBADA | Mean |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DistilGPT-2 | 82M | 27.5 | 59.8 | 36.6 | 23.0 | 50.4 | 71.6 | 26.8 | 79.5 | 22.9 | 25.0 | 42.3 |
| GPT-2 | 124M | 31.1 | 62.5 | 39.5 | 22.7 | 51.6 | 75.3 | 27.2 | 82.5 | 22.9 | 32.6 | 44.8 |
| GPT-2 Medium | 355M | 39.4 | 66.4 | 43.6 | 25.0 | 53.1 | 77.5 | 30.2 | 82.6 | 22.9 | 43.0 | 48.4 |
| Pythia-70M | 70M | 27.4 | 58.4 | 34.4 | 22.4 | 51.3 | 61.2 | 24.8 | 73.8 | 22.9 | 18.5 | 39.5 |
| Pythia-160M | 162M | 30.3 | 61.6 | 39.6 | 23.9 | 50.6 | 74.0 | 27.6 | 78.2 | 23.1 | 32.8 | 44.2 |
| GPT-Neo-125M | 125M | 30.4 | 62.5 | 39.4 | 23.1 | 50.4 | 76.6 | 26.2 | 79.3 | 23.1 | 37.4 | 44.8 |
| OPT-125M | 125M | 31.6 | 62.0 | 40.1 | 22.4 | 51.4 | 78.4 | 28.8 | 81.4 | 22.9 | 38.9 | 45.8 |
| SmolLM-135M | 135M | 42.5 | 68.6 | 56.2 | 28.9 | 53.1 | 83.7 | 34.0 | 79.8 | 25.2 | 37.5 | 50.9 |
| SmolLM2-135M | 135M | 43.1 | 68.4 | 58.6 | 29.4 | 52.6 | 84.2 | 33.2 | 80.3 | 24.2 | 42.8 | 51.7 |

</div>

Table 8. Public models evaluated with the same script and protocol (%). Parameter counts are taken from the loaded weights.

<div class="report-table" markdown>

| Model | Max context | WikiText-103 word PPL | WikiText-103 bits/byte | PG-19 test word PPL | PG-19 test bits/byte | LAMBADA PPL |
|:--|--:|--:|--:|--:|--:|--:|
| Sliding window | 4,096 | 73.05 | 1.158 | 106.09 | 1.158 | 382.44 |
| DistilGPT-2 | 1,024 | 52.96 | 1.071 | 370.39 | 1.469 | 96.85 |
| GPT-2 | 1,024 | 37.37 | 0.977 | 280.89 | 1.400 | 40.06 |
| GPT-2 Medium | 1,024 | 26.75 | 0.887 | 143.89 | 1.234 | 18.26 |
| Pythia-70M | 2,048 | 57.05 | 1.091 | 200.17 | 1.316 | 142.56 |
| Pythia-160M | 2,048 | 33.41 | 0.947 | 107.09 | 1.161 | 38.18 |
| GPT-Neo-125M | 2,048 | 32.29 | 0.937 | 105.24 | 1.156 | 30.27 |
| OPT-125M | 2,048 | 31.95 | 0.935 | 129.99 | 1.209 | 23.14 |
| SmolLM-135M | 2,048 | 26.57 | 0.885 | 102.91 | 1.151 | 32.39 |
| SmolLM2-135M | 8,192 | 21.07 | 0.822 | 64.17 | 1.033 | 19.05 |

</div>

Table 9. Language-modelling metrics (lower is better). WikiText-103 test and PG-19 test are cut into non-overlapping windows of each model's maximum context, as lm-evaluation-harness does. Word-level perplexity is normalised by the number of whitespace-separated words and can be compared across tokenizers.

To be written.


### 4.8 Ablations

<figure markdown="span" class="report-figure">
  ![Ablations](assets/report/fig_ablations.light.svg#only-light)
  ![Ablations](assets/report/fig_ablations.dark.svg#only-dark)
  <figcaption>Figure 6. Four design choices of TTT-E2E (models with 8 blocks, width 512 and 2,048 tokens of context; one choice changed at a time). Horizontal lines are full attention and sliding-window attention at the same size; in the rightmost panel the sliding-window baseline changes with the window.</figcaption>
</figure>

<div class="report-table" markdown>

| Setting | Training tokens | PG-19 val NLL | First 256 tokens | Last 512 tokens | Updates off | FineWeb-Edu NLL | Step time (s) |
|:--|--:|--:|--:|--:|--:|--:|--:|

</div>

Table 10. Ablation results (PG-19 validation, 2,048 tokens of context). “Updates off” is the loss of the same checkpoint evaluated without test-time updates.

To be written.


### 4.9 Seeds and model size

<div class="report-table" markdown>

| Quantity | seed 0 | seed 1 | seed 2 | Mean | SD |
|:--|--:|--:|--:|--:|--:|

</div>

Table 11. PG-19 validation loss under three seeds (initialisation and data order both change). Within a seed all models read the same data, so differences are paired by seed.

<div class="report-table" markdown>

| Layers × width | Full attention | Sliding window | TTT-E2E | Δ TTT − sliding window | Δ TTT − full attention |
|:--|--:|--:|--:|--:|--:|

</div>

Table 12. PG-19 validation loss of three model sizes at the same 164M training tokens (2,048 tokens of context, window 256).

<figure markdown="span" class="report-figure">
  ![Model size and amount of training](assets/report/fig_scaling.light.svg#only-light)
  ![Model size and amount of training](assets/report/fig_scaling.dark.svg#only-dark)
  <figcaption>Figure 7. Left: the three sizes of Table 10. Right: differences in PG-19 validation loss between models during the main training runs, against training tokens.</figcaption>
</figure>

To be written.


### 4.10 Cost

<div class="report-table" markdown>

| Model | Parameters | Fast-weight parameters | Training throughput (tokens/s) | Relative to sliding window | Step time (s) | Peak memory (GiB) |
|:--|--:|--:|--:|--:|--:|--:|
| Sliding window | 123.6M | – | 193,430 | 1.00× | 0.339 | 81.8 |
| No-update control | 137.7M | 14.2M | 125,550 | 0.65× | 0.522 | 13.2 |
| Full attention | 123.6M | – | 82,087 | 0.42× | 0.798 | 45.4 |
| TTT-E2E | 137.7M | 14.2M | 70,218 | 0.36× | 0.933 | 77.7 |

</div>

Table 13. Training cost on one H200 (batch of 16 × 4,096 tokens, random input, median of 4–5 steps). Full attention and the no-update control use per-block gradient checkpointing; without it full attention exceeds the memory of one GPU.

<figure markdown="span" class="report-figure">
  ![Training and forward throughput](assets/report/fig_cost.light.svg#only-light)
  ![Training and forward throughput](assets/report/fig_cost.dark.svg#only-dark)
  <figcaption>Figure 8. Left: training throughput from Table 11. Right: forward throughput at different context lengths (the whole window is read at once and the loss computed at every position, 32,768 tokens per batch; TTT-E2E includes its test-time updates).</figcaption>
</figure>

<div class="report-table" markdown>

| Model | Context read (tokens) | Fast weights (MiB) | KV cache (MiB) | Session state, total (MiB) | Token-by-token decoding (tokens/s) |
|:--|--:|--:|--:|--:|--:|

</div>

Table 14. State kept by one inference session after reading contexts of different lengths, and the speed of token-by-token decoding that follows (batch size 1). State sizes are counted from the arrays actually allocated.

To be written.


## 5 Data audit

<div class="report-table" markdown>

| Benchmark | Items probed | Items whose probe string occurs verbatim in the training text | Share |
|:--|--:|--:|--:|
| hellaswag | 9,793 | 0 | 0.00% |
| piqa | 395 | 0 | 0.00% |
| arc_easy | 2,079 | 13 | 0.63% |
| arc_challenge | 1,061 | 1 | 0.09% |
| winogrande | 1,267 | 0 | 0.00% |
| lambada_openai | 5,153 | 2 | 0.04% |
| sciq | 813 | 6 | 0.74% |
| openbookqa | 206 | 0 | 0.00% |
| mmlu | 12,477 | 94 | 0.75% |
| wikitext | 61 | 2 | 3.28% |

</div>

Table 15. Verbatim overlap between evaluation items and the training corpora. For each item, the first 80 characters of the normalised question text (lower-cased, whitespace collapsed) are matched exactly against all training text (21.9 GB).

To be written.


## 6 Discussion and limitations

To be written.



## 7 Scope and release notes

- **What the models are.** The four main models are small English base language models that have only been pre-trained. They have had no instruction tuning, chat tuning or safety alignment. They are suitable for research on sequence modelling, not for direct use by end users, and their output should not be treated as a source of facts.
- **Data licences.** Training uses two public corpora: FineWeb-Edu (ODC-By 1.0) and PG-19 (public-domain book text; the dataset card is Apache-2.0). The repository does not redistribute the corpora; it ships download scripts, pinned revisions and file digests. Evaluation sets are downloaded as evaluation splits only, under their own licences, and never enter training.
- **Content risks.** FineWeb-Edu comes from web pages and PG-19 from books published before 1919, some of which use language at odds with present-day norms. The models reproduce biases and outdated content in their training data. This report contains no toxicity, bias or jailbreak evaluation. Those are required before any user-facing release and are not claimed here.
- **Risks specific to test-time updates.** Fast weights exist only inside one session and are discarded when it ends; user input cannot rewrite the released slow weights, and the implementation isolates the fast weights and caches of different sessions. Within a session, however, the model's behaviour drifts with what it reads. That is the purpose of the method, and it also means that behaviour in a long session is harder to bound in advance than for a fixed-weight model. Section 4.5 reports how far the fast weights drift over 32K tokens.
- **What this report does not claim.** It does not claim the best results among models of this size (the public models in Section 4.7 are stronger on most tasks, with far more training). It does not claim the conclusions must hold for larger models, longer training or longer windows. It claims no comparability with any closed model.

## 8 Reproduction

Every experiment can be rerun with the scripts in the repository. Each step writes the SHA256 of its input files, its configuration and its environment into the output directory.

```bash
# Environment (Linux, CUDA 12, Python 3.12)
uv sync --frozen --no-install-project --python 3.12 --extra cuda --extra checkpoint
uv pip install --python .venv/bin/python tiktoken matplotlib pyahocorasick

# 1. Corpora: download, tokenize, de-duplicate, evaluation sets
python scripts/download_public_corpora.py --dataset pg19 --output data/raw/pg19
python scripts/download_public_corpora.py --dataset fineweb-edu --output data/raw/fineweb-edu --shards 0 1 2
python scripts/download_public_corpora.py --dataset fineweb-edu --output data/raw/fineweb-edu-heldout --shards 13
for split in train validation test; do
  python scripts/tokenize_corpus.py --source text-dir --input data/raw/pg19/$split --output data/tokens/pg19-$split
done
python scripts/tokenize_corpus.py --source parquet --input data/raw/fineweb-edu/00[012]_00000.parquet \
  --output data/tokens/fineweb-edu-train
python scripts/audit_report_data.py duplicates --fineweb-train data/raw/fineweb-edu/00[012]_00000.parquet \
  --fineweb-heldout data/raw/fineweb-edu-heldout/013_00000.parquet --pg19 data/raw/pg19 \
  --output outputs/audits/duplicates.json
python scripts/tokenize_corpus.py --source parquet --input data/raw/fineweb-edu-heldout/013_00000.parquet \
  --max-documents 20000 --exclude-digests outputs/audits/duplicates.json \
  --output data/tokens/fineweb-edu-heldout-dedup
python scripts/download_benchmarks.py --output data/benchmarks

# 2. Training and evaluation: one pipeline per GPU, no communication; rerun to resume after an interruption
python scripts/make_report_configs.py
scripts/report_pipeline.sh 0 baseline   # sliding window → full attention → throughput → ablation queue
scripts/report_pipeline.sh 1 ttt        # TTT-E2E → no-update control → naive-update baseline → ablation queue
scripts/run_reference_benchmarks.sh 0   # public reference models + lm-evaluation-harness cross-check

# 3. Audits, figures and the report
python scripts/audit_report_data.py sampler --recipe configs/data/fineweb_pg19_50_50.json \
  --config configs/report/main_ttt.json --output outputs/audits/sampler_main.json
python scripts/audit_report_data.py benchmarks --fineweb-train data/raw/fineweb-edu/00[012]_00000.parquet \
  --pg19-train data/raw/pg19/train --output outputs/audits/benchmark_overlap.json
python scripts/make_report_figures.py && python scripts/make_report_diagram.py
python scripts/render_report.py
```

| Item | Content |
|:--|:--|
| Code | this repository. The `ttt/` source used to train the sliding-window and TTT-E2E main models is commit `d54407d` (source digest `e3553bf5…`); all other training used commit `854484e` (digest `5610909c…`). Between the two, `ttt/` differs only by a first-order switch that is off by default, the way configurations are compared on resume, and import order; the numerics are identical. The source digest of every run is recorded under `provenance` in `results.json`; the algorithm is in `ttt/provenance.py`. |
| Configurations | `configs/report/*.json` (generated by `scripts/make_report_configs.py`); data mixture `configs/data/fineweb_pg19_50_50.json` |
| Raw results | `docs/assets/report/results.json` (the source of every number on this page); raw outputs of the evaluation scripts under `docs/assets/report/raw/` |
| Software | Python 3.12.16, JAX 0.4.38, Optax 0.2.4, Orbax 0.10.3, NumPy 2.5.4; reference models loaded with PyTorch 2.15 and Transformers 5.23; lm-evaluation-harness 0.4.13 |
| Hardware | NVIDIA H200 NVL (143,771 MiB, driver 580.196.07), AMD EPYC 9555, Ubuntu 22.04 |
| Randomness | initialisation and data order are set by `seed` in the configuration; evaluation windows are chosen deterministically, with no random numbers |

Model weights are large and are not stored in the Git repository. Checkpoints remain on the training server; the commands above regenerate them from scratch.

## 9 Version history

| Version | Date | Content |
|:--|:--|:--|
| 0.1–0.6 | 2026-11-16 to 11-17 | Pipeline validation: a 1M-parameter CPU run and 15M-parameter, byte-level paired runs on WikiText-2 on H200. These confirmed that training, restore and evaluation work end to end, but the models were at chance on public benchmarks and support no claim about capability. The records are kept in [CPU Pilot](reports/cpu_pilot_1m.md), [H200 15M Scaling](reports/h200_scaling.md) and [Benchmark protocol](reports/benchmark_protocol_en.md). |
| 1.0 | 2026-11-18 | This report: GPT-2 BPE, two public corpora with billions of tokens, a matched experiment with four models of about 130M parameters at 4K context, evaluations of long context, verbatim recall, public benchmarks, ablations, seeds, model size and cost, and data audits. |


## References

///Footnotes Go Here///

[^ttte2e]: A. Tandon et al. End-to-End Test-Time Training for Long Context. arXiv:2512.23675, 2025.
[^pg19]: J. W. Rae, A. Potapenko, S. M. Jayakumar, T. P. Lillicrap. Compressive Transformers for Long-Range Sequence Modelling. arXiv:1911.05507, 2019.
[^fineweb]: G. Penedo et al. The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. arXiv:2406.17557, 2024.
[^gpt2]: A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever. Language Models are Unsupervised Multitask Learners. OpenAI, 2019.
[^distil]: V. Sanh, L. Debut, J. Chaumond, T. Wolf. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv:1910.01108, 2019.
[^pythia]: S. Biderman et al. Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. arXiv:2304.01373, 2023.
[^gptneo]: S. Black, L. Gao, P. Wang, C. Leahy, S. Biderman. GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow. Zenodo, 2021. doi:10.5281/zenodo.5297715.
[^opt]: S. Zhang et al. OPT: Open Pre-trained Transformer Language Models. arXiv:2205.01068, 2022.
[^smollm2]: L. Ben Allal et al. SmolLM2: When Smol Goes Big – Data-Centric Training of a Small Language Model. arXiv:2502.02737, 2025.
[^lmeval]: S. Biderman et al. Lessons from the Trenches on Reproducible Evaluation of Language Models. arXiv:2405.14782, 2024.

