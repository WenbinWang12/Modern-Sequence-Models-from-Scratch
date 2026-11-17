## 3 Experimental setup

### 3.1 Four models

The main experiment trains four models with 12 blocks and width 768. They share the tokenizer, the data, the exact batch order, the optimiser and the number of training tokens. They differ only in how they use context.

| Model | Attention | Blocks with MLP′ | Test-time updates | Parameters | Non-embedding parameters |
|:--|:--|:--|:--|--:|--:|
| Full attention | all 4,096 positions | – | – | {{ M(r.training_cost.full.parameters, 2) }} | {{ M(r.training_cost.full.parameters - 50257 * 768, 2) }} |
| Sliding window | last \(w=512\) positions | – | – | {{ M(r.training_cost.swa.parameters, 2) }} | {{ M(r.training_cost.swa.parameters - 50257 * 768, 2) }} |
| No-update control | last \(w=512\) positions | last 3 | none | {{ M(r.training_cost.ck.parameters, 2) }} | {{ M(r.training_cost.ck.parameters - 50257 * 768, 2) }} |
| TTT-E2E | last \(w=512\) positions | last 3 | one step every \(C=256\) tokens | {{ M(r.training_cost.ttt.parameters, 2) }} | {{ M(r.training_cost.ttt.parameters - 50257 * 768, 2) }} |

Each control answers one question.

- **Full attention** is the reference in which everything in context is visible without loss. Its cost grows linearly with context; this report treats it as the quality target, not as a cost competitor.
- **Sliding window** is the baseline whose cost is of the same order as TTT-E2E. Each layer sees only the last 512 positions directly; older information can arrive only by being relayed layer to layer (at most about 6,100 positions back with 12 layers), and the model has no dedicated long-range memory.
- **No-update control** has the same architecture and parameter count as TTT-E2E (both carry {{ M(r.training_cost.ttt.fast_parameters, 2) }} more parameters in MLP′ than the first two) and runs no inner loop in training or inference. It separates the effect of the extra parameters from the effect of test-time updates.
- At evaluation the trained TTT-E2E model is also run with **updates off**, to see how the same slow weights behave without updating.

Everything else is identical across the four: \(d=768\), 12 attention heads, \(d_{\mathrm{ff}}=2048\), tied input and output embeddings, RoPE base \(5\times10^5\), QK normalisation.

### 3.2 Data

| Corpus | Use | Documents | Tokens | Bytes/token | Source and licence |
|:--|:--|--:|--:|--:|:--|
| FineWeb-Edu `sample/10BT`, shards 000–002 | training | {{ n(r.audits.corpora["fineweb-edu-train"].documents) }} | {{ B(r.audits.corpora["fineweb-edu-train"].tokens) }} | {{ f2(r.audits.corpora["fineweb-edu-train"].bytes_per_token) }} | HuggingFaceFW/fineweb-edu, revision `87f0914`, ODC-By 1.0[^fineweb] |
| PG-19 train | training | {{ n(r.audits.corpora["pg19-train"].documents) }} | {{ B(r.audits.corpora["pg19-train"].tokens) }} | {{ f2(r.audits.corpora["pg19-train"].bytes_per_token) }} | Project Gutenberg books released by DeepMind, file lists at revision `4d28bd7`, Apache-2.0[^pg19] |
| PG-19 validation | validation during training, ablations | {{ n(r.audits.corpora["pg19-validation"].documents) }} | {{ M(r.audits.corpora["pg19-validation"].tokens, 2) }} | {{ f2(r.audits.corpora["pg19-validation"].bytes_per_token) }} | as above, official split |
| PG-19 test | final evaluation | {{ n(r.audits.corpora["pg19-test"].documents) }} | {{ M(r.audits.corpora["pg19-test"].tokens, 2) }} | {{ f2(r.audits.corpora["pg19-test"].bytes_per_token) }} | as above, official split |
| FineWeb-Edu `sample/10BT`, first 20,000 documents of shard 013 | final evaluation | {{ n(r.audits.corpora["fineweb-edu-heldout-dedup"].documents) }} | {{ M(r.audits.corpora["fineweb-edu-heldout-dedup"].tokens, 2) }} | {{ f2(r.audits.corpora["fineweb-edu-heldout-dedup"].bytes_per_token) }} | as FineWeb-Edu; {{ n(r.audits.corpora["fineweb-edu-heldout-dedup"].excluded_duplicate_documents) }} documents whose full text also occurs in the training shards were removed |

The two corpora complement each other. FineWeb-Edu is web text filtered by a quality classifier; its documents are short (about 1,000 tokens on average) and it drives general language ability. Each PG-19 document is a whole book (about 100,000 tokens on average), so a 4,096-token window almost always lies inside one book. It is the data that actually needs long context.

**Tokenizer.** All models use the GPT-2 BPE vocabulary (\(V=50{,}257\)). Each document is encoded as `<|endoftext|>` followed by its text, and documents are concatenated into one token stream. We use an existing vocabulary rather than training one so that losses can be compared directly with public models that share it.

**Mixture and sampling.** Each training example first picks a source with probability 50%, then draws a window of \(L+1\) tokens uniformly at random from that source's token stream, with replacement. Windows may cross document boundaries. Replaying the sampler with the training seed gives the realised shares and coverage:

{% table sampler %}

Training touches only a small part of each corpus, and the great majority of training windows are seen once.

### 3.3 Training

| Item | Value |
|:--|:--|
| Sequence length \(L\) / batch | 4,096 / 16 sequences (65,536 tokens per step) |
| Steps / tokens | 12,000 / {{ n(r.training.swa.tokens_seen) }} |
| Optimiser | AdamW, \(\beta_1=0.9\), \(\beta_2=0.95\), weight decay 0.1 (matrices only) |
| Learning rate | peak \(6\times10^{-4}\); linear warm-up over 480 steps, then cosine decay to 0 |
| Outer gradient clipping | global norm 1.0 |
| Inner loop (TTT-E2E only) | \(\eta=1.0\), \(\tau=1.0\); \(\eta\) rises linearly from 0.1 to 1.0 over the first 1,200 steps |
| Precision | float32 parameters, bfloat16 matrix products |
| Seed | 0 (same initialisation seed and data order for all four models) |
| Hardware | one NVIDIA H200 (143,771 MiB) per model, no multi-GPU training |

No hyper-parameter was tuned for any single model: the four share the table above, and the inner-loop \(\eta\) and \(\tau\) are the implementation defaults. The number of training tokens is about 31% of the original paper's basic recipe for its 125M model (2.5 billion tokens), bounded by one night on two GPUs. This is the main scale limitation of the report.

### 3.4 Evaluation protocol

**Held-out language modelling.** Each PG-19 test book is cut from its start into non-overlapping 4,096-token windows (a trailing partial window is dropped): {{ n(r.fig_context_4k.arms.swa.windows) }} windows and {{ M(r.fig_context_4k.arms.swa.tokens, 2) }} predicted positions. Every window starts from an empty state: the KV cache is cleared and the fast weights return to \(\phi_0\). The loss at position \(t\) of a window is therefore the loss with exactly \(t\) tokens of context, and binning by position gives loss as a function of context length. FineWeb-Edu held-out documents are short, so they are concatenated as in training and the first 2,048 windows are used. We report mean negative log-likelihood per token (NLL, in nats), perplexity and bits per byte.

**Statistics.** All four models are evaluated on the same windows, so differences between models are estimated as paired differences, with 95% confidence intervals from 2,000 bootstrap resamples over books for PG-19 and over windows for FineWeb-Edu. The intervals reflect sampling error of the evaluation set. They do not include the randomness of retraining, which Section 4.9 estimates separately with multiple seeds.

**Beyond the training length.** The window is lengthened to 32,768 tokens, eight times the training length, with the protocol otherwise unchanged and no fine-tuning.

**Public benchmarks.** Every task is zero-shot and scored by log-likelihood: for multiple choice, \(\log p(\text{choice}\mid\text{context})\) is computed for each choice and the largest wins. Prompt templates, the way choices are attached and length normalisation match the task of the same name in lm-evaluation-harness[^lmeval]. To confirm this we evaluated GPT-2 with our script and with lm-evaluation-harness and compared them metric by metric (Section 4.7). No few-shot examples are used, and no training, prompt search or checkpoint selection targets any benchmark; every model is evaluated at its final checkpoint only.

**Public reference models.** To anchor absolute levels, a set of public models of similar size is evaluated with the same script. Their training data exceeds ours by one to three orders of magnitude; these numbers locate our models and are not a budget-matched comparison.
