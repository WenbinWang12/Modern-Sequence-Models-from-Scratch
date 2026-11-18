---
hide:
  - navigation
---

# TTT-E2E technical report: from-scratch training, matched controls and long-context evaluation

**Version 1.0 · 18 November 2026**

## Abstract

We implement end-to-end test-time training (TTT-E2E) from scratch: while reading a context, the model runs gradient descent on one set of MLPs in its last few blocks using the text it has just read, and training differentiates through those updates so that the model learns to be updated this way. On two H200 GPUs we trained four models of about 138M parameters at 4,096 context, with identical data, batch order, optimiser and token budget: full attention, sliding-window attention (window 512), a control with TTT's architecture and parameter count but no updates, and TTT-E2E. The corpus is half FineWeb-Edu and half PG-19, 786,432,000 GPT-2 BPE tokens in total.

On PG-19 test, TTT-E2E's loss per token is 0.072 nats lower than sliding-window attention (95% CI [−0.076, −0.068]) and 0.051 nats lower than full attention. How the gap varies with available context is in Section 4.3, and results with the evaluation window stretched to eight times the training length (32,768 tokens) are in Section 4.5.
The report measures three factors separately: the extra parameters, the test-time updates, and the meta-learning during training. The no-update control answers the first, switching updates off in the same checkpoint answers the second, and running test-time gradient updates on a model that was never meta-trained (dynamic evaluation) answers the third. On cost, TTT-E2E's training throughput is 0.36× that of sliding-window attention, while its per-token inference cost and session state are independent of context length.

We measure the method's limits as well: a verbatim-recall probe and a RULER-style single-needle retrieval task ask how much detail survives compression into weights. Ten public zero-shot benchmarks use a protocol checked metric by metric against lm-evaluation-harness (the GPT-2 cross-check is in Section 4.7), alongside public models trained on one to three orders of magnitude more data. Data audits cover the sampling mixture, duplication between training and evaluation sets, and verbatim occurrences of benchmark items in the training corpus. Every number on this page is produced by a script from raw evaluation output, and that output ships with the repository.


## 1 Introduction

Long-context language models face a basic cost conflict. Full attention lets every token read all of the preceding text directly, but the compute per generated token and the state that must be kept both grow linearly with the length already read. Restricting attention to a recent window makes both constant, at the price of losing almost everything outside the window. Recurrent approaches compress the history into a fixed-size state, which turns the question into what to compress and how.

TTT-E2E[^ttte2e] answers it by compressing the history into part of the model's own weights, using gradient descent to do the compressing. As the model reads, it trains itself to predict the next token on the text it has just read, updating only one set of MLPs in the last few blocks; training then uses meta-learning so the model is prepared to be updated that way. At 3B parameters and 164B training tokens the original paper reports that within 128K of context, TTT-E2E's loss scales with context length the way full attention does, while inference latency does not grow with context.

This report implements the method from scratch and tests it at a scale that two GPUs can reproduce end to end. We train four models of about 138M parameters with identical data, batch order, optimiser and token budget, and ask five questions:

1. How much of the prediction quality lost to a limited window does TTT-E2E recover, and how far is it from full attention? (Sections 4.2 and 4.3)
2. How much of any gain comes from the test-time updates themselves rather than from the extra parameters, and is plain test-time gradient descent on a model that was never meta-trained enough? (Section 4.4)
3. What happens once the context exceeds the training length? (Section 4.5)
4. What is lost by compressing history into weights? (Section 4.6)
5. What does it cost in training and in inference? (Section 4.10)

Section 4.7 places these models next to public models of similar size under one zero-shot protocol to anchor absolute levels; Sections 4.8 and 4.9 cover ablations, seeds and model size; Section 5 is the data audit.

**Relation to prior work.** Continuing to apply gradient updates on the test sequence goes back at least to dynamic evaluation[^dyneval][^dynevaltr]; fast weights[^fastweights] supply the view of weights that change along a sequence; learning an initialisation by differentiating through an inner loop is MAML[^maml], with the first-order variant in Reptile[^reptile]. Test-time training for out-of-distribution generalisation is Sun et al.[^ttt], and building it into sequence-model layers appears in TTT layers[^tttlayers] and Titans[^titans]. State-space and linear recurrent models such as Mamba[^mamba] and Gated DeltaNet[^gdn] pursue the same constant inference cost; Transformers that restrict attention or reuse state across segments include Transformer-XL[^transformerxl], Longformer[^longformer], the Compressive Transformer[^pg19] and Block-Recurrent Transformers[^blockrecurrent]. This report implements and evaluates TTT-E2E and its controls only; it does not reimplement the other methods.

**Contributions.**

- A complete JAX implementation: training with second-order gradients, stateful streaming inference, checkpointing and restore, and tests that make correctness checkable (causality, outer-loop gradients, streaming consistency).
- A strictly matched set of controls that separates the extra parameters, the test-time updates and the meta-learning.
- Evaluation at eight times the training length, a verbatim-recall probe, public benchmarks, and a comparison with public models under one protocol.
- Full traceability: every number on this page is generated by a script from raw evaluation output, and both the output and the scripts are in the repository.


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

**Held-out language modelling.** Each PG-19 test book is cut from its start into non-overlapping 4,096-token windows (a trailing partial window is dropped): 2,531 windows and 10.37M predicted positions. Every window starts from an empty state: the KV cache is cleared and the fast weights return to \(\phi_0\). The loss at position \(t\) of a window is therefore the loss with exactly \(t\) tokens of context, and binning by position gives loss as a function of context length. FineWeb-Edu held-out documents are short, so they are concatenated as in training and the first 2,048 windows are used. We report mean negative log-likelihood per token (NLL, in nats), perplexity and bits per byte.

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

All four models eventually ran to 12,000 steps and 786,432,000 tokens with no divergence or loss spikes. During training, user-level NFS quota exhaustion interrupted checkpoint writes. The supervisor resumed each run from its latest checkpoint and recorded the event in `interruptions.jsonl`. The run recorded 3 automatic recovery event(s). Gradient clipping rarely bound: after warm-up, the outer-loop gradient norm exceeded the threshold of 1.0 on 0.4% of logged steps for sliding-window attention and 0.6% for TTT-E2E.

Training losses are not comparable across the four. Full attention can read the whole window at every position during training and so has a lower training loss by construction, while sliding-window attention sees only the last 512 positions; and TTT-E2E's training loss is measured *after* its fast weights have been updated the way they will be at test time, which is a different quantity again. The left panel of Figure 2 serves only to confirm that all four runs converged cleanly. The comparable quantities are in the middle and right panels: one evaluation protocol, one set of held-out windows.

The validation curves show something as important as the final numbers: TTT-E2E leads sliding-window attention from early in training, and the gap does not close as training proceeds (Figure 7, right). The gain is therefore not an advantage that merely reflects undertraining.
TTT-E2E's inner loop stays active throughout training. The fast-weight norm grows from 75.3 at initialisation to 132.5, each inner update has a norm of about 0.98, and the inner gradient norm sits near 1.22 — slightly above the clipping threshold \(\tau=1\), meaning most update steps are clipped to a fixed step size. What the outer loop learns is therefore which direction to move in at a fixed step size, rather than how far to move. Section 4.8's inner-learning-rate ablation shows the corresponding effect.


### 4.2 Main result: held-out loss at 4K context

<div class="report-table" markdown>

| Model | PG-19 test NLL | PPL | bits/byte | Δ vs full attention (95% CI) | FineWeb-Edu NLL | PPL |
|:--|--:|--:|--:|--:|--:|--:|
| Full attention | 3.052 | 21.16 | 1.151 | – | 3.469 | 32.11 |
| Sliding window | 3.073 | 21.62 | 1.159 | +0.021 [+0.017, +0.025] | 3.477 | 32.35 |
| No-update control | 3.050 | 21.13 | 1.151 | −0.002 [−0.005, +0.002] | 3.451 | 31.52 |
| TTT-E2E (updates off) | 3.080 | 21.75 | 1.162 | +0.027 [+0.021, +0.033] | 3.472 | 32.20 |
| TTT-E2E | **3.001** | 20.11 | 1.132 | −0.051 [−0.053, −0.049] | **3.415** | 30.41 |

</div>

Table 1. Loss of the final checkpoints on the two held-out sets (lower is better). Differences are paired differences from full attention on the same windows, with 95% confidence intervals in brackets. Bits per byte use the corpus-average bytes per token.

Table 1 gives the final results of all four models on both held-out sets. The paired differences are measured against full attention on the same windows.

On PG-19 test, TTT-E2E's loss per token is 0.051 nats below full attention (95% CI [−0.053, −0.049]) and 0.072 nats below sliding-window attention. FineWeb-Edu held-out points the same way with a smaller margin (0.054 nats below full attention).

The size of that 0.072 nat gap needs a yardstick. In the same table, **widening the attention window from 512 to all 4,096 positions** (sliding window → full attention) is worth only 0.021 nats. At this scale and on this data, test-time updates buy more than twice what letting the model see the whole context buys.
This is the report's main result, and its scope is narrow: one model size, one data recipe, a training length of 4,096. Section 4.9 repeats it across three seeds and three model sizes, and Section 6 states where the conclusion should not be expected to hold.


### 4.3 Loss as a function of context length

<figure markdown="span" class="report-figure">
  ![Loss by context length on PG-19 test](assets/report/fig_context_4k.light.svg#only-light)
  ![Loss by context length on PG-19 test](assets/report/fig_context_4k.dark.svg#only-dark)
  <figcaption>Figure 3. Loss on PG-19 test against the number of tokens already read in the window (means over 64 positions). Left: loss of each model. Right: paired difference from full attention; bands are 95% confidence intervals from resampling books. The sliding window is 512 tokens and TTT-E2E updates every 256 tokens.</figcaption>
</figure>

<div class="report-table" markdown>

| Context position (tokens) | 0–512 | 512–1K | 1K–2K | 2K–4K | All |
|:--|--:|--:|--:|--:|--:|
| Full attention | 3.189 | 3.056 | 3.028 | 3.029 | 3.052 |
| Sliding window | 3.158 | 3.052 | 3.050 | 3.070 | 3.073 |
| No-update control | 3.133 | 3.029 | 3.027 | 3.047 | 3.050 |
| TTT-E2E (updates off) | 3.141 | 3.056 | 3.059 | 3.080 | 3.080 |
| TTT-E2E | 3.126 | 3.004 | 2.980 | 2.980 | 3.001 |
| Δ TTT-E2E − Sliding window | −0.031<br>[−0.033, −0.030] | −0.048<br>[−0.050, −0.046] | −0.070<br>[−0.073, −0.067] | −0.090<br>[−0.095, −0.084] | −0.072<br>[−0.076, −0.068] |
| Δ TTT-E2E − Full attention | −0.062<br>[−0.066, −0.058] | −0.052<br>[−0.054, −0.050] | −0.048<br>[−0.051, −0.046] | −0.049<br>[−0.052, −0.047] | −0.051<br>[−0.053, −0.049] |
| Δ Sliding window − Full attention | −0.031<br>[−0.034, −0.028] | −0.004<br>[−0.006, −0.002] | +0.022<br>[+0.018, +0.026] | +0.040<br>[+0.034, +0.046] | +0.021<br>[+0.017, +0.025] |

</div>

Table 2. Loss on PG-19 test by range of context position, and paired differences between models (95% confidence intervals).

Figure 3 unfolds the loss by how many tokens have been read inside the window. All four curves nearly coincide over the first 256 tokens — exactly the update chunk \(C=256\). Before the first inner-loop update, TTT-E2E and its updates-off version *are* the same model. The moment the first update lands, the curves separate and the gap keeps widening. This is direct evidence that the mechanism works as designed: the gain appears exactly when updating begins.

Against sliding-window attention, TTT-E2E's advantage grows monotonically with available context: from 0.031 nats over the first 512 tokens to 0.090 nats over 2,048–4,096 (Table 2). That is what compressing history into weights predicts: the more that has been read, the more has been written into the fast weights.

The comparison with full attention has the opposite shape. The advantage is largest when context is shortest (0.062 nats), then narrows and settles near 0.049 nats. A gap at short context cannot come from remembering more history — there is hardly any history there. The likelier explanation is structural: TTT-E2E's suffix carries an extra MLP′, and it was trained under the condition that its weights would be updated. Section 4.4 isolates that part.

### 4.4 Where the gain comes from

Three things separate TTT-E2E from sliding-window attention: the extra MLP′ in the suffix (parameters), the gradient updates at inference (test-time compute), and the fact that training simulated those updates (meta-learning). Four models plus two dedicated evaluations pull them apart.

**The extra parameters alone.** The no-update control has TTT-E2E's architecture and parameter count and runs no inner loop in training or inference. It is 0.023 nats below sliding-window attention ([−0.024, −0.022]). The extra 14.2M parameters do help, but they account for only part of the total 0.072 nat gap.

**The test-time updates alone.** TTT-E2E is 0.049 nats below the no-update control ([−0.052, −0.046]). The two have identical architecture and parameter count; they differ only in whether the fast weights are updated at inference and whether training simulated that. This is the largest of the three terms.
**Are updates enough without meta-learning?** This is the decisive question: if plain test-time gradient descent on any ordinary model gave the same gain, the second-order training pass would be wasted effort. We ran the same inner loop directly on the no-update control, sweeping the test-time learning rate. It barely moves: the best setting reaches 3.0456 against 3.0552 without updates. The same machinery applied to the meta-trained TTT-E2E takes it from 3.0793 to 3.0078.

What helps, in other words, is **not the act of updating but having been trained to be updated** — which is exactly what the second-order training pass buys.
One observation points the other way and is worth recording. Evaluating the trained TTT-E2E with **updates switched off** puts it 0.029 nats *above* the no-update control ([+0.026, +0.032]). Slow weights trained with updates come to **depend** on them: part of the modelling work has been delegated to the inner loop, and removing it leaves a hole. TTT-E2E is therefore not a feature that can be switched off to save compute; it is a design that has to be adopted as a whole.

### 4.5 Beyond the training length: 32K context

<figure markdown="span" class="report-figure">
  ![Loss beyond the training length](assets/report/fig_context_32k.light.svg#only-light)
  ![Loss beyond the training length](assets/report/fig_context_32k.dark.svg#only-dark)
  <figcaption>Figure 4. Loss on PG-19 test when the evaluation window is lengthened to 32,768 tokens, eight times the training length (means over 256 positions; the vertical line marks the training length). The dashed line is full attention extrapolated with no truncation. The right panel shows paired differences from the matched sliding-window, no-update and TTT-no-adapt conditions.</figcaption>
</figure>

<div class="report-table" markdown>

| Context position (tokens) | 0–4K | 4K–8K | 8K–16K | 16K–32K | All |
|:--|--:|--:|--:|--:|--:|
| Sliding window | 3.000 | 3.085 | 3.072 | 3.073 | 3.065 |
| No-update control | 2.977 | 3.061 | 3.049 | 3.051 | 3.042 |
| TTT-E2E (updates off) | 3.002 | 3.095 | 3.083 | 3.084 | 3.075 |
| TTT-E2E | 2.928 | 2.978 | 2.954 | 2.936 | 2.945 |
| Δ TTT-E2E − Sliding window | −0.072<br>[−0.077, −0.067] | −0.107<br>[−0.116, −0.096] | −0.118<br>[−0.129, −0.107] | −0.137<br>[−0.153, −0.121] | −0.120<br>[−0.133, −0.108] |
| Δ TTT-E2E − No-update control | −0.049<br>[−0.053, −0.044] | −0.083<br>[−0.092, −0.073] | −0.095<br>[−0.105, −0.084] | −0.114<br>[−0.130, −0.099] | −0.097<br>[−0.109, −0.085] |
| Δ TTT-E2E − TTT-E2E (updates off) | −0.074<br>[−0.081, −0.066] | −0.117<br>[−0.129, −0.104] | −0.129<br>[−0.142, −0.115] | −0.148<br>[−0.166, −0.129] | −0.130<br>[−0.144, −0.115] |

</div>

Table 3. Loss by position range inside 32K windows, and paired differences (95% confidence intervals). Every model was trained at length 4,096 only.

The evaluation window is stretched from 4,096 to 32,768 tokens — eight times the training length — with no fine-tuning and no change to the models. Full attention cannot take part in this comparison: its position encoding was only ever trained within 4,096, so extrapolating to 32K leaves the training distribution. We measured it separately at 8,192 and 16,384; see `full_extrapolation` in `results.json`.

Inside 32K windows, TTT-E2E's advantage over sliding-window attention widens to 0.120 nats ([−0.133, −0.108]), up from 0.072 nats at 4K. Both models lose some ground past the training length (Figure 4, left), but TTT-E2E loses it more slowly.

One distinction matters here. Neither sliding-window attention nor TTT-E2E has a length-extrapolation problem: every layer sees only the last 512 positions, so the relative distances their position encoding meets at 32K are exactly those seen in training. This figure therefore does not measure whether longer inputs can be processed, but whether the fast weights keep accumulating useful information as more is read. The widening advantage says the inner loop still works, without diverging, at eight times the training length.
The fast-weight diagnostics agree. Across the 32K evaluation, the fast-weight norm after each update chunk changes smoothly and the inner gradient norm stays near the clipping threshold; over 128 update chunks neither diverges nor collapses. That is direct evidence of a stable inner loop far beyond the training length.

### 4.6 What compression loses: verbatim recall

<figure markdown="span" class="report-figure">
  ![Verbatim recall probe](assets/report/fig_recall.light.svg#only-light){ width="70%" }
  ![Verbatim recall probe](assets/report/fig_recall.dark.svg#only-dark){ width="70%" }
  <figcaption>Figure 5. Verbatim recall probe. Tokens 128–191 of a window are copied to a later position. The vertical axis is how much lower the loss on the middle 32 tokens of the repeat is than on the unmodified text; larger values mean the model makes more use of text it has already seen.</figcaption>
</figure>

<div class="report-table" markdown>

| Distance to first occurrence (tokens) | 320 | 640 | 1,152 | 2,176 | 3,712 |
|:--|--:|--:|--:|--:|--:|
| Full attention | +1.83<br>[+1.71, +1.94] | +1.53<br>[+1.42, +1.64] | +1.37<br>[+1.27, +1.46] | +0.99<br>[+0.89, +1.08] | +0.52<br>[+0.43, +0.62] |
| Sliding window | +1.51<br>[+1.40, +1.62] | −0.11<br>[−0.21, −0.01] | −0.15<br>[−0.26, −0.05] | −0.07<br>[−0.17, +0.03] | −0.12<br>[−0.21, −0.03] |
| No-update control | +1.58<br>[+1.47, +1.69] | −0.12<br>[−0.21, −0.02] | −0.16<br>[−0.26, −0.06] | −0.08<br>[−0.18, +0.02] | −0.12<br>[−0.21, −0.02] |
| TTT-E2E (updates off) | +1.15<br>[+1.04, +1.25] | −0.12<br>[−0.22, −0.03] | −0.14<br>[−0.25, −0.04] | −0.06<br>[−0.15, +0.04] | −0.08<br>[−0.17, +0.02] |
| TTT-E2E | +1.58<br>[+1.48, +1.68] | +0.61<br>[+0.52, +0.70] | +0.58<br>[+0.48, +0.67] | +0.65<br>[+0.56, +0.74] | +0.58<br>[+0.49, +0.66] |

</div>

Table 4. Loss reduction on the repeated span (nats, 95% confidence intervals in brackets; first 4 windows of each PG-19 test book, 400 in total).

<div class="report-table" markdown>

| Model | 1K | 2K | 4K | 8K | 16K | 32K |
|:--|--:|--:|--:|--:|--:|--:|
| Full attention | – | – | – | – | – | – |
| Sliding window | 2.05 | 1.08 | 0.51 | 0.52 | 0.41 | 0.37 |
| No-update control | 2.24 | 1.12 | 0.54 | 0.53 | 0.43 | 0.41 |
| TTT-E2E (updates off) | 2.10 | 1.05 | 0.51 | 0.53 | 0.43 | 0.39 |
| TTT-E2E | 2.59 | 2.00 | 1.53 | 1.54 | 1.56 | 1.32 |

</div>

Table 5. Needle in a haystack: one sentence, “One of the special magic numbers for &lt;key&gt; is: &lt;7-digit number&gt;.”, is hidden in the haystack text and the model is then asked for that key's number. The table reports how much lower the answer's log-likelihood per token is than when the needle is replaced by an unrelated key–value pair (nats; larger means the needle was used more). The haystack is PG-19 test book text, the needle is placed at ten evenly spaced depths, and the prompt and needle wording come from RULER.

The previous sections show that TTT-E2E uses context better, but not *how*. Compressing history into fixed-size weights must lose something; two probes ask what.

**Verbatim recall.** A 64-token span of the window is copied verbatim to a later position, and we measure how much lower the loss on the repeat is than on unmodified text. Attention can look the original up directly, so full attention gains 1.83 nats at distance 320, decaying slowly to 0.52 nats.

Sliding-window attention falls off a cliff: 1.51 nats at distance 320 (still inside its 512 window), then nothing — or slightly negative — once the span leaves the window (−0.11 at distance 640). Beyond the window it has no memory of text it has read.

TTT-E2E fills that cliff, but differently from attention: 0.61 nats at distance 640, and then **almost no decay with distance** (0.58 nats at 3,712). That is the signature of compression into weights — information is not ordered by recency, it is accumulated into a set of weights. The cost is equally clear: the absolute size is about half of full attention's. The compression is lossy.
**Single-needle retrieval.** Built after RULER's `niah` task: a sentence stating that some key's special number is a 7-digit value is hidden in haystack text, and the model is asked for it. These are 138M-parameter base models with no instruction tuning, and exact match is zero for every one of them — a row of zeros that says nothing. So we instead measure the answer's log-likelihood with the needle present against the same context with the needle replaced by an unrelated key–value pair: it asks whether the needle was used at all, even when the model cannot state the full answer.

At 32K context TTT-E2E scores 1.32 nats/token, against 0.37 for sliding-window attention and 0.41 for the no-update control. Those two decline steadily as context grows and the needle is pushed out of the window; TTT-E2E is essentially flat from 4K to 32K.
Both probes point the same way: TTT-E2E keeps a memory that is **stable in distance but limited in precision**, where attention keeps one that is **precise but bounded by window or cost**. They are not the same thing, which is also why the original paper's needle-retrieval results trail full attention.


### 4.7 Public benchmarks and public models

<div class="report-table" markdown>

| Model | Params | HellaSwag | PIQA | ARC-E | ARC-C | WinoGrande | SciQ | OBQA | BLiMP | MMLU | LAMBADA | Mean |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Full attention | 124M | 26.6 | 57.4 | 39.1 | 23.5 | 50.3 | 64.4 | 25.2 | 74.6 | 23.0 | 16.4 | 40.1 |
| Sliding window | 124M | 27.1 | 57.9 | 38.2 | 22.6 | 52.2 | 70.1 | 25.4 | 76.0 | 23.0 | 17.7 | 41.0 |
| No-update control | 138M | 27.5 | 57.9 | 40.6 | 23.5 | 52.0 | 68.4 | 26.4 | 76.9 | 23.0 | 17.4 | 41.4 |
| TTT-E2E (updates off) | 138M | 27.5 | 58.2 | 39.6 | 24.7 | 52.0 | 64.0 | 29.0 | 76.1 | 23.1 | 15.2 | 41.0 |
| TTT-E2E | 138M | 27.5 | 58.2 | 39.6 | 24.7 | 52.1 | 64.6 | 29.0 | 76.1 | 23.1 | 15.2 | 41.0 |
| 95% CI half-width |  | ±0.9 | ±2.3 | ±2.0 | ±2.4 | ±2.7 | ±3.0 | ±3.8 | ±0.3 | ±0.7 | ±1.0 |  |

</div>

Table 6. Zero-shot accuracy (%) for the four trained configurations in this report, with the TTT checkpoint also evaluated with test-time updates disabled. HellaSwag, PIQA, ARC and OpenBookQA report accuracy normalised by choice length; the others report plain accuracy. “Mean” is the arithmetic mean of the ten tasks.

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
| Full attention | 4,096 | 63.97 | 1.122 | 102.66 | 1.150 | 469.84 |
| Sliding window | 4,096 | 73.05 | 1.158 | 106.09 | 1.158 | 382.44 |
| No-update control | 4,096 | 70.61 | 1.149 | 102.45 | 1.150 | 382.97 |
| TTT-E2E (updates off) | 4,096 | 78.30 | 1.176 | 107.11 | 1.161 | 497.22 |
| TTT-E2E | 4,096 | 59.87 | 1.104 | 95.01 | 1.131 | 497.20 |
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

**Protocol cross-check.** Before quoting any score, our harness has to be shown to agree with an accepted implementation. We evaluated GPT-2 (124M) with our script and with lm-evaluation-harness and compared metric by metric: 13 of 17 are identical, and the other 4 differ by less than 3×10⁻⁴ (SciQ's 0.1% comes from a single borderline item). The comparisons in Tables 6–8 are therefore protocol-equivalent.
**Our four models.** Table 6 shows them clustered closely across the ten tasks, with most differences inside the 95% confidence intervals. That is expected: these are 138M-parameter base models trained on 786M tokens, near chance (25%) on tasks like MMLU. A stable 0.07 nat difference in held-out loss is below the resolution of these benchmarks.

**Against public models (Tables 7 and 8).** This is positioning, not a budget-matched comparison: the public models in Table 7 were trained on one to three orders of magnitude more data (SmolLM2-135M on 2T tokens, roughly 2,500× ours). They are clearly stronger on knowledge-heavy tasks, as expected.

**One "advantage" that must be explained.** In Table 9 our models reach far lower word-level perplexity on PG-19 test than any public model. That is **not** evidence of a better model: half of our training data is the PG-19 training split, the same domain as the test set, while most public models never trained on pre-1919 book text. Symmetrically, we are clearly worse on WikiText-103 because our training data contains no Wikipedia. Both gaps, in opposite directions, come from the data recipe rather than the architecture, and reading them as capability comparisons would be wrong.


<div class="report-table" markdown>

| Task | Adaptation | Valid samples | Score | Protocol |
|:--|--:|--:|--:|:--|
| ARC-Challenge | off | 100/100 | 20.0 | arc_piqa_loglikelihood_v1 |
| BBH sports_understanding | off | 100/100 | 0.0 | bbh_greedy_v1 |
| GPQA | off | 30/65 | 23.1 | arc_piqa_loglikelihood_v1 |
| GSM8K | off | 100/100 | 3.0 | gsm8k_generate_v1 |
| HumanEval | off | 100/100 | 0.0 | humaneval_generate_v1 |
| PIQA | off | 100/100 | 51.0 | arc_piqa_loglikelihood_v1 |
| Synthetic long-context preflight | off | — | 256: 0.0, 512: 0.0, 1024: 0.0, 2048: 0.0, 4096: 0.0, 8192: 0.0 | synthetic_kv_retrieval_v1 |
| ARC-Challenge | off | 1158/1165 | 20.1 | arc_piqa_loglikelihood_v1 |
| BBH sports_understanding | off | 250/250 | 0.0 | bbh_greedy_v1 |
| GPQA | off | 64/256 | 22.7 | arc_piqa_loglikelihood_v1 |
| GSM8K | off | 1319/1319 | 0.9 | gsm8k_generate_v1 |
| HumanEval | off | 164/164 | 0.0 | humaneval_generate_v1 |
| PIQA | off | 1800/1819 | 53.7 | arc_piqa_loglikelihood_v1 |
| Synthetic long-context preflight | off | — | 256: 0.0, 512: 0.0, 1024: 0.0, 2048: 0.0, 4096: 0.0, 8192: 0.0 | synthetic_kv_retrieval_v1 |
| ARC-Challenge | on | 1158/1165 | 20.4 | arc_piqa_loglikelihood_v1 |
| BBH sports_understanding | on | 250/250 | 0.0 | bbh_greedy_v1 |
| GPQA | on | 64/256 | 24.6 | arc_piqa_loglikelihood_v1 |
| GSM8K | on | 1319/1319 | 0.8 | gsm8k_generate_v1 |
| HumanEval | on | 164/164 | 0.0 | humaneval_generate_v1 |
| PIQA | on | 1800/1819 | 53.9 | arc_piqa_loglikelihood_v1 |
| Synthetic long-context preflight | on | — | 256: 0.0, 512: 0.0, 1024: 0.0, 2048: 0.0, 4096: 0.0, 8192: 0.0 | synthetic_kv_retrieval_v1 |
| ARC-Challenge | off | 100/100 | 24.0 | arc_piqa_loglikelihood_v1 |
| BBH sports_understanding | off | 100/100 | 0.0 | bbh_greedy_v1 |
| GPQA | off | 30/65 | 26.2 | arc_piqa_loglikelihood_v1 |
| GSM8K | off | 100/100 | 2.0 | gsm8k_generate_v1 |
| HumanEval | off | 100/100 | 0.0 | humaneval_generate_v1 |
| PIQA | off | 100/100 | 55.0 | arc_piqa_loglikelihood_v1 |
| Synthetic long-context preflight | off | — | 256: 0.0, 512: 0.0, 1024: 0.0, 2048: 0.0, 4096: 0.0, 8192: 0.0 | synthetic_kv_retrieval_v1 |

</div>

Table 9a. Extended diagnostics requested for this project. ARC, PIQA and GPQA use answer-text continuation scoring; GSM8K and HumanEval use greedy generation; BBH is the named single-task diagnostic. The synthetic long-context row is a RULER/LongBench engineering preflight, not an official suite score.

### 4.8 Ablations

<figure markdown="span" class="report-figure">
  ![Ablations](assets/report/fig_ablations.light.svg#only-light)
  ![Ablations](assets/report/fig_ablations.dark.svg#only-dark)
  <figcaption>Figure 6. Four design choices of TTT-E2E (models with 8 blocks, width 512 and 2,048 tokens of context; one choice changed at a time). Horizontal lines are full attention and sliding-window attention at the same size; in the rightmost panel the sliding-window baseline changes with the window.</figcaption>
</figure>

<div class="report-table" markdown>

| Setting | Training tokens | PG-19 val NLL | First 256 tokens | Last 512 tokens | Updates off | FineWeb-Edu NLL | Step time (s) |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Full attention | 164M | 3.666 | 3.836 | 3.632 | – | 4.148 | 0.131 |
| Sliding window (k=256) | 164M | 3.650 | 3.752 | 3.644 | – | 4.106 | 0.082 |
| Sliding window, same training time as TTT | 600M | 3.342 | 3.455 | 3.335 | – | 3.700 | 0.082 |
| No-update control | 164M | 3.635 | 3.737 | 3.629 | – | 4.087 | 0.104 |
| TTT-E2E (base: η=1, C=128, last 2 blocks, k=256) | 164M | 3.557 | 3.722 | 3.521 | 3.675 | 3.979 | 0.299 |
| First-order (no gradient through inner gradients) | 164M | 3.539 | 3.692 | 3.504 | 3.629 | 3.962 | 0.200 |
| Inner learning rate η=0.1 | 164M | 3.543 | 3.685 | 3.517 | 3.613 | 3.971 | 0.297 |
| Inner learning rate η=0.3 | 164M | 3.531 | 3.683 | 3.498 | 3.619 | 3.955 | 0.296 |
| Inner learning rate η=3 | 164M | 3.669 | 3.848 | 3.626 | 3.816 | 4.110 | 0.296 |
| Update chunk C=64 | 164M | 3.558 | 3.727 | 3.521 | 3.721 | 3.975 | 0.376 |
| Update chunk C=256 | 164M | 3.558 | 3.718 | 3.522 | 3.635 | 3.988 | 0.265 |
| Fast weights in the last block only | 164M | 3.567 | 3.729 | 3.531 | 3.675 | 3.991 | 0.253 |
| Fast weights in the last 4 blocks | 164M | 3.547 | 3.712 | 3.509 | 3.670 | 3.970 | 0.396 |
| Sliding window, k=128 | 164M | 3.679 | 3.752 | 3.677 | – | 4.142 | 0.081 |
| TTT-E2E, k=128 | 164M | 3.553 | 3.714 | 3.516 | 3.692 | 3.978 | 0.288 |
| Sliding window, k=512 | 164M | 3.648 | 3.778 | 3.636 | – | 4.107 | 0.099 |
| TTT-E2E, k=512 | 164M | 3.562 | 3.737 | 3.524 | 3.665 | 3.987 | 0.316 |

</div>

Table 10. Ablation results (PG-19 validation, 2,048 tokens of context). “Updates off” is the loss of the same checkpoint evaluated without test-time updates.

Ablations use smaller models (8 blocks, width 512, 2,048 tokens of context), changing one factor at a time (Table 9, Figure 6).

**Inner learning rate \(\eta\).** Values of 0.1, 0.3 and 1 give similar results (3.543, 3.531, 3.557), while \(\eta=3\) is clearly worse (3.669), falling behind the sliding-window baseline at the same size. This matches Section 4.1: the inner gradient norm sits near the clipping threshold, so most updates are clipped to a fixed step size and \(\eta\) matters little over a wide range — until the step is large enough to destabilise training.
**Update chunk \(C\).** 64, 128 and 256 are nearly indistinguishable (3.558, 3.557, 3.558). Smaller chunks mean more frequent updates and higher training cost without buying lower loss at this scale, so the main experiment uses the larger \(C=256\).
**Number of blocks with fast weights.** One, two and four blocks give 3.567, 3.557 and 3.547: more blocks help slightly, with diminishing returns and a backward pass through each one per chunk.
**Attention window \(k\).** TTT-E2E barely moves across 128, 256 and 512 (3.553, 3.557, 3.562), while the sliding-window baseline is markedly more sensitive (3.679 → 3.648). The fast weights take over work that the attention window would otherwise have to do, which is where the long-context cost advantage comes from.
**Parameters matched exactly.** Narrowing every MLP so the total parameter count equals the baseline without MLP′ gives 3.576, still well ahead of the sliding window's 3.650. The gain does not rest on the extra parameter budget.
**First-order approximation.** Treating the inner gradient as a constant leaves the forward pass unchanged and makes training cheaper. It scores 3.539, matching or slightly beating full second-order training at 3.557. At this scale the second-order term through the inner loop is **not required** — a useful negative result, though whether it survives at larger scale is beyond what this report can answer.
**Matched training time.** Every comparison above fixes the token count, but TTT-E2E is slower per step (Table 12). Using the measured step time, we gave the sliding-window baseline the same wall-clock budget: 18,300 steps and 600M tokens, about 3.7× the data. It reaches 3.342, **far better** than TTT-E2E's 3.557.

This is the most important negative result in the report. At a fixed token budget TTT-E2E is clearly better; but if the binding constraint is **time or compute** rather than data, spending that time on more data is the better deal at this scale. Section 6 takes this up.

### 4.9 Seeds and model size

<div class="report-table" markdown>

| Quantity | seed 0 | seed 1 | seed 2 | Mean | SD |
|:--|--:|--:|--:|--:|--:|
| Full attention NLL | 3.666 | 3.660 | 3.656 | 3.661 | 0.005 |
| Sliding window NLL | 3.650 | 3.658 | 3.656 | 3.655 | 0.004 |
| TTT-E2E NLL | 3.557 | 3.560 | 3.550 | 3.556 | 0.005 |
| Δ TTT-E2E − Sliding window | −0.093 | −0.098 | −0.105 | −0.099 | 0.006 |
| Δ TTT-E2E − Full attention | −0.109 | −0.100 | −0.105 | −0.105 | 0.004 |
| Δ Sliding window − Full attention | −0.016 | −0.002 | −0.000 | −0.006 | 0.008 |

</div>

Table 11. PG-19 validation loss under three seeds (initialisation and data order both change). Within a seed all models read the same data, so differences are paired by seed.

<div class="report-table" markdown>

| Layers × width | Full attention | Sliding window | TTT-E2E | Δ TTT − sliding window | Δ TTT − full attention |
|:--|--:|--:|--:|--:|--:|
| 6 × 384 | 3.761 | 3.769 | 3.659 | −0.109 | −0.102 |
| 8 × 512 | 3.666 | 3.650 | 3.557 | −0.093 | −0.109 |
| 12 × 768 | 3.520 | 3.532 | 3.420 | −0.112 | −0.100 |

</div>

Table 12. PG-19 validation loss of three model sizes at the same 164M training tokens (2,048 tokens of context, window 256).

<figure markdown="span" class="report-figure">
  ![Model size and amount of training](assets/report/fig_scaling.light.svg#only-light)
  ![Model size and amount of training](assets/report/fig_scaling.dark.svg#only-dark)
  <figcaption>Figure 7. Left: the three sizes of Table 10. Right: differences in PG-19 validation loss between models during the main training runs, against training tokens.</figcaption>
</figure>

**Seeds.** Table 10 gives results for three seeds, changing both initialisation and data order. The per-model standard deviation runs from 0.0040 to 0.0049, while the TTT-E2E − sliding-window difference is −0.099 ± 0.0061 and TTT-E2E − full attention is −0.105 ± 0.0042. The spread of the difference is comparable to a single model's spread while the difference itself is an order of magnitude larger, so the conclusion does not rest on one lucky initialisation.

The same table holds another point worth noting: sliding-window attention and full attention differ by only −0.006 ± 0.0084 — **indistinguishable within seed noise**. At 2,048 context and this data budget, widening the window from 256 to fully visible buys almost nothing, while test-time updates do.
**Model size.** Table 11 compares three sizes (6×384, 8×512, 12×768) at a fixed 164M-token budget. TTT-E2E beats both baselines at all three, with a margin from 0.109 nats at the smallest size to 0.112 nats at the largest; it does not vanish as size grows.

Three points cannot establish a trend, and all three sit in a narrow range at one data budget. The original paper reports how the gain varies from 125M to 3B parameters; this report can neither confirm nor contradict that.

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
| Full attention | 2,048 | – | 144.2 | 145.7 | 601 |
| Full attention | 8,192 | – | 576.8 | 578.3 | 512 |
| Full attention | 32,768 | – | 2307.0 | 2308.5 | 284 |
| Sliding window | 2,048 | – | 18.0 | 19.5 | 643 |
| Sliding window | 8,192 | – | 18.0 | 19.5 | 631 |
| Sliding window | 32,768 | – | 18.0 | 19.5 | 627 |
| TTT-E2E | 2,048 | 54.0 | 18.0 | 72.4 | 573 |
| TTT-E2E | 8,192 | 54.0 | 18.0 | 72.4 | 575 |
| TTT-E2E | 32,768 | 54.0 | 18.0 | 72.4 | 582 |

</div>

Table 14. State kept by one inference session after reading contexts of different lengths, and the speed of token-by-token decoding that follows (batch size 1). State sizes are counted from the arrays actually allocated.

**Training.** Table 12 gives measured throughput on one H200. Sliding-window attention is fastest (193K tokens/s), full attention reaches 82K, and TTT-E2E is slowest at 70K — 0.36× the sliding window. Two things cost: the extra MLP′ forward pass in the suffix, and one backward pass through the suffix per update chunk, with the outer loop then differentiating through that chain. The no-update control runs at 126K, which places about half the overhead on the extra MLP′ itself and half on the inner loop.

Section 4.8's matched-time comparison shows what this costs in practice: given the same wall-clock budget, the sliding-window baseline reads 3.7× more data and ends up with a lower loss.
**Inference.** Table 13 reports measured session state and decoding speed. TTT-E2E's session state is the fast weights (a fixed 54.0 MiB, independent of how much has been read) plus a sliding-window KV cache (also fixed); whether 2K or 32K of context has been read, the state size and the token-by-token decoding speed do not change. Full attention's KV cache grows linearly with context.

This is the method's selling point: what it pays in training buys inference cost that is constant in context length. At the scale of this report the trade is not worth it — the models are small and the contexts short, so attention is cheap anyway — and it would only pay off at longer contexts and larger serving volumes, which is beyond what this report can verify.

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

**Do evaluation items appear in the training corpus?** FineWeb-Edu comes from public web pages, and public benchmark items also circulate on the web, so this has to be checked rather than assumed. Table 15 reports: no verbatim overlap for HellaSwag, PIQA, WinoGrande or OpenBookQA; under 1% of items for ARC-Easy, ARC-Challenge, SciQ, LAMBADA and MMLU, the largest being MMLU (94 of 12,477, 0.75%). Even if every one of those items were answered correctly — or incorrectly — the task's accuracy would move by no more than that share, which changes none of the conclusions in Section 4.7. Of the 61 WikiText-103 test articles, 2 have a passage that occurs in the training corpus, which is expected: Wikipedia text is widely mirrored on the web.

The check has two known blind spots. First, it finds verbatim overlap only, not paraphrase, translation or answer-only overlap. Second, items whose normalised question text is shorter than 50 characters are not probed (which is why the "items probed" column is below the full item count), so PIQA and OpenBookQA are only partly covered.

**Duplication between training and held-out sets.** PG-19 uses the official splits. Comparing digests of normalised full texts, none of the 150 validation and test books is identical to any of the 28,602 training books, and no book shares a normalised "title + author" string with the training set in the PG-19 metadata. FineWeb-Edu's held-out set comes from a shard that training never used, but web corpora contain cross-shard duplicates: 365 of the 20,000 documents first drawn (1.8%) were identical to a training document. Those were removed, and every FineWeb-Edu number in this report is computed on the de-duplicated set. Near-duplicates, such as pages differing only in headers and footers, were not checked.

**What training actually read.** Replaying the sampler with the training seed gives the shares in Section 3.2: the two sources land within 0.2 percentage points of the 50/50 target. Training covered 19.7% of the FineWeb-Edu training shards and 14.8% of the PG-19 training set, and about nine tenths of the text that was read was read exactly once. Training is therefore close to single-pass, and the models had little opportunity to memorise training text by repetition.


## 6 Discussion and limitations

**What this report measured.** At a fixed token budget, with data and hyper-parameters strictly matched, TTT-E2E beat both full attention and sliding-window attention on two held-out sets, and the advantage reappeared across three seeds and three model sizes. Decomposing the gain, the largest term is the test-time updates themselves and the second is the extra parameters; running the same gradient updates on a model that was never meta-trained achieves almost nothing, so what matters is having been *trained to be updated*, not the act of updating. Two probes show the memory it keeps is stable with distance but less precise than attention's.

**What it did not measure.** Scale. All four main models have 138M parameters and 786M training tokens — about 31% of the original paper's smallest recipe and two orders of magnitude below its largest experiment. The three sizes in Section 4.9 sit in a narrow band. Context too: a training length of 4,096 and evaluation out to 32,768, against the original paper's claims at 128K. This report can neither confirm nor refute the conclusions at that scale.

**The most important caveat.** Section 4.8's matched-training-time comparison is the result least favourable to TTT-E2E: given the same wall-clock budget, the sliding-window baseline reads 3.7× more data and reaches a clearly lower loss. This is not an implementation-efficiency problem — differentiating through the inner loop has an inherent cost, and the original paper lists training latency as a main limitation too. The conclusion should therefore be stated as: **when data is the binding constraint (a fixed token budget), TTT-E2E trades extra compute for a better model; when compute is the binding constraint, that compute is better spent reading more data.** Both situations occur in practice, and which applies depends on the constraint rather than on the method.

**A useful negative result.** The first-order approximation — not differentiating through the inner gradient — matches or slightly beats full second-order training at this scale while being cheaper. If that holds at larger scale it would change the cost trade-off above substantially; this report's scale cannot test it.

**A property worth noting.** Slow weights trained with updates come to depend on them: with updates switched off, the model is worse than a matched control that never used them. TTT-E2E is therefore not a switch that can be turned off at inference to save compute, but a design that must be adopted whole. That has direct deployment consequences.

**Relation to the original paper.** Agreeing in direction: TTT-E2E beats the baselines at matched budget, its inference cost is independent of context length, its advantage widens with longer context, and it trails full attention on needle-style retrieval. This report reproduces those qualitative findings independently at a much smaller scale and adds decompositions the original does not report (the no-update control, the dynamic-evaluation baseline, the matched-time comparison, and the first-order variant). The original's quantitative claims at 128K context and 3B parameters are neither verified nor refuted here.



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
scripts/run_extended_benchmarks.sh 7 main-full no-adapt 100
scripts/run_extended_benchmarks.sh 7 main-ttt adapt 100
scripts/run_extended_benchmarks.sh 7 main-ttt no-adapt 100

The `0` and `1` values are examples of device IDs assigned to a job, not fixed machine-wide IDs. On the H200 run for this report, GPU 7 ran the baseline pipeline and GPU 5 ran the TTT pipeline. The two pipelines were independent and exchanged no cross-device data.

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
[^tttlayers]: Y. Sun et al. Learning to (Learn at Test Time): RNNs with Expressive Hidden States. arXiv:2407.04620, 2024.
[^ttt]: Y. Sun et al. Test-Time Training with Self-Supervision for Generalization under Distribution Shifts. ICML 2020. arXiv:1909.13231.
[^maml]: C. Finn, P. Abbeel, S. Levine. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. ICML 2017. arXiv:1703.03400.
[^reptile]: A. Nichol, J. Achiam, J. Schulman. On First-Order Meta-Learning Algorithms. arXiv:1803.02999, 2018.
[^dyneval]: B. Krause, E. Kahembwe, I. Murray, S. Renals. Dynamic Evaluation of Neural Sequence Models. arXiv:1709.07432, 2017.
[^dynevaltr]: B. Krause, E. Kahembwe, I. Murray, S. Renals. Dynamic Evaluation of Transformer Language Models. arXiv:1904.08378, 2019.
[^fastweights]: J. Ba et al. Using Fast Weights to Attend to the Recent Past. arXiv:1610.06258, 2016.
[^titans]: A. Behrouz, P. Zhong, V. Mirrokni. Titans: Learning to Memorize at Test Time. arXiv:2501.00663, 2024.
[^mamba]: A. Gu, T. Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752, 2023.
[^gdn]: S. Yang, J. Kautz, A. Hatamizadeh. Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR 2025. arXiv:2412.06464.
[^transformerxl]: Z. Dai et al. Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context. ACL 2019. arXiv:1901.02860.
[^longformer]: I. Beltagy, M. E. Peters, A. Cohan. Longformer: The Long-Document Transformer. arXiv:2004.05150, 2020.
[^pg19]: J. W. Rae, A. Potapenko, S. M. Jayakumar, T. P. Lillicrap. Compressive Transformers for Long-Range Sequence Modelling. arXiv:1911.05507, 2019.
[^blockrecurrent]: D. Hutchins et al. Block-Recurrent Transformers. arXiv:2203.07852, 2022.
[^fineweb]: G. Penedo et al. The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. arXiv:2406.17557, 2024.
[^gpt2]: A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever. Language Models are Unsupervised Multitask Learners. OpenAI, 2019.
[^distil]: V. Sanh, L. Debut, J. Chaumond, T. Wolf. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv:1910.01108, 2019.
[^pythia]: S. Biderman et al. Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. arXiv:2304.01373, 2023.
[^gptneo]: S. Black, L. Gao, P. Wang, C. Leahy, S. Biderman. GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow. Zenodo, 2021. doi:10.5281/zenodo.5297715.
[^opt]: S. Zhang et al. OPT: Open Pre-trained Transformer Language Models. arXiv:2205.01068, 2022.
[^smollm2]: L. Ben Allal et al. SmolLM2: When Smol Goes Big – Data-Centric Training of a Small Language Model. arXiv:2502.02737, 2025.
[^lmeval]: S. Biderman et al. Lessons from the Trenches on Reproducible Evaluation of Language Models. arXiv:2405.14782, 2024.

