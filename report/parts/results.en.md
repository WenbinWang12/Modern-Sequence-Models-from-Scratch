## 4 Results

### 4.1 Training

<figure markdown="span" class="report-figure">
  ![Training loss and held-out loss against training tokens](assets/report/fig_training.light.svg#only-light)
  ![Training loss and held-out loss against training tokens](assets/report/fig_training.dark.svg#only-dark)
  <figcaption>Figure 2. Training. Left: training loss (exponential moving average). Centre and right: loss of the checkpoints saved every 1,000 steps on PG-19 validation (first 5 windows of each book, 250 in total) and on FineWeb-Edu held-out (256 windows). All four models read exactly the same batch at every step.</figcaption>
</figure>

{% include parts/results_4_1.en.md %}

### 4.2 Main result: held-out loss at 4K context

<div class="report-table" markdown>

{% table main %}

</div>

Table 1. Loss of the final checkpoints on the two held-out sets (lower is better). Differences are paired differences from full attention on the same windows, with 95% confidence intervals in brackets. Bits per byte use the corpus-average bytes per token.

{% include parts/results_4_2.en.md %}

### 4.3 Loss as a function of context length

<figure markdown="span" class="report-figure">
  ![Loss by context length on PG-19 test](assets/report/fig_context_4k.light.svg#only-light)
  ![Loss by context length on PG-19 test](assets/report/fig_context_4k.dark.svg#only-dark)
  <figcaption>Figure 3. Loss on PG-19 test against the number of tokens already read in the window (means over 64 positions). Left: loss of each model. Right: paired difference from full attention; bands are 95% confidence intervals from resampling books. The sliding window is 512 tokens and TTT-E2E updates every 256 tokens.</figcaption>
</figure>

<div class="report-table" markdown>

{% table ranges %}

</div>

Table 2. Loss on PG-19 test by range of context position, and paired differences between models (95% confidence intervals).

{% include parts/results_4_3.en.md %}

### 4.4 Where the gain comes from

{% include parts/results_4_4.en.md %}

### 4.5 Beyond the training length: 32K context

<figure markdown="span" class="report-figure">
  ![Loss beyond the training length](assets/report/fig_context_32k.light.svg#only-light)
  ![Loss beyond the training length](assets/report/fig_context_32k.dark.svg#only-dark)
  <figcaption>Figure 4. Loss on PG-19 test when the evaluation window is lengthened to 32,768 tokens, eight times the training length (means over 256 positions; the vertical line marks the training length). The dashed line is full attention extrapolated with no truncation; “last 4K only” restricts the same full-attention model to the range it saw in training. The right panel shows paired differences from sliding-window attention.</figcaption>
</figure>

<div class="report-table" markdown>

{% table long %}

</div>

Table 3. Loss by position range inside 32K windows, and paired differences (95% confidence intervals). Every model was trained at length 4,096 only.

{% include parts/results_4_5.en.md %}

### 4.6 What compression loses: verbatim recall

<figure markdown="span" class="report-figure">
  ![Verbatim recall probe](assets/report/fig_recall.light.svg#only-light){ width="70%" }
  ![Verbatim recall probe](assets/report/fig_recall.dark.svg#only-dark){ width="70%" }
  <figcaption>Figure 5. Verbatim recall probe. Tokens 128–191 of a window are copied to a later position. The vertical axis is how much lower the loss on the middle 32 tokens of the repeat is than on the unmodified text; larger values mean the model makes more use of text it has already seen.</figcaption>
</figure>

<div class="report-table" markdown>

{% table recall %}

</div>

Table 4. Loss reduction on the repeated span (nats, 95% confidence intervals in brackets; first 4 windows of each PG-19 test book, 400 in total).

<div class="report-table" markdown>

{% table niah %}

</div>

Table 5. Needle in a haystack: one sentence, “One of the special magic numbers for &lt;key&gt; is: &lt;7-digit number&gt;.”, is hidden in the haystack text and the model is then asked for that key's number. The table reports how much lower the answer's log-likelihood per token is than when the needle is replaced by an unrelated key–value pair (nats; larger means the needle was used more). The haystack is PG-19 test book text, the needle is placed at ten evenly spaced depths, and the prompt and needle wording come from RULER.

{% include parts/results_4_6.en.md %}

### 4.7 Public benchmarks and public models

<div class="report-table" markdown>

{% table benchmarks_ours %}

</div>

Table 6. Zero-shot accuracy (%) for the four trained configurations in this report, with the TTT checkpoint also evaluated with test-time updates disabled. HellaSwag, PIQA, ARC and OpenBookQA report accuracy normalised by choice length; the others report plain accuracy. “Mean” is the arithmetic mean of the ten tasks.

<div class="report-table" markdown>

{% table reference_models %}

</div>

Table 7. The public models used for comparison. The training-data column is taken from each model card or paper (checked November 2026); their training budgets exceed ours by one to three orders of magnitude.

<div class="report-table" markdown>

{% table benchmarks_reference %}

</div>

Table 8. Public models evaluated with the same script and protocol (%). Parameter counts are taken from the loaded weights.

<div class="report-table" markdown>

{% table perplexity %}

</div>

Table 9. Language-modelling metrics (lower is better). WikiText-103 test and PG-19 test are cut into non-overlapping windows of each model's maximum context, as lm-evaluation-harness does. Word-level perplexity is normalised by the number of whitespace-separated words and can be compared across tokenizers.

{% include parts/results_4_7.en.md %}

<div class="report-table" markdown>

{% table extended_benchmarks %}

</div>

Table 9a. Extended diagnostics requested for this project. ARC, PIQA and GPQA use answer-text continuation scoring; GSM8K and HumanEval use greedy generation; BBH is the named single-task diagnostic. The synthetic long-context row is a RULER/LongBench engineering preflight, not an official suite score.

### 4.8 Ablations

<figure markdown="span" class="report-figure">
  ![Ablations](assets/report/fig_ablations.light.svg#only-light)
  ![Ablations](assets/report/fig_ablations.dark.svg#only-dark)
  <figcaption>Figure 6. Four design choices of TTT-E2E (models with 8 blocks, width 512 and 2,048 tokens of context; one choice changed at a time). Horizontal lines are full attention and sliding-window attention at the same size; in the rightmost panel the sliding-window baseline changes with the window.</figcaption>
</figure>

<div class="report-table" markdown>

{% table ablations %}

</div>

Table 10. Ablation results (PG-19 validation, 2,048 tokens of context). “Updates off” is the loss of the same checkpoint evaluated without test-time updates.

{% include parts/results_4_8.en.md %}

### 4.9 Seeds and model size

<div class="report-table" markdown>

{% table seeds %}

</div>

Table 11. PG-19 validation loss under three seeds (initialisation and data order both change). Within a seed all models read the same data, so differences are paired by seed.

<div class="report-table" markdown>

{% table scaling %}

</div>

Table 12. PG-19 validation loss of three model sizes at the same 164M training tokens (2,048 tokens of context, window 256).

<figure markdown="span" class="report-figure">
  ![Model size and amount of training](assets/report/fig_scaling.light.svg#only-light)
  ![Model size and amount of training](assets/report/fig_scaling.dark.svg#only-dark)
  <figcaption>Figure 7. Left: the three sizes of Table 10. Right: differences in PG-19 validation loss between models during the main training runs, against training tokens.</figcaption>
</figure>

{% include parts/results_4_9.en.md %}

### 4.10 Cost

<div class="report-table" markdown>

{% table cost %}

</div>

Table 13. Training cost on one H200 (batch of 16 × 4,096 tokens, random input, median of 4–5 steps). Full attention and the no-update control use per-block gradient checkpointing; without it full attention exceeds the memory of one GPU.

<figure markdown="span" class="report-figure">
  ![Training and forward throughput](assets/report/fig_cost.light.svg#only-light)
  ![Training and forward throughput](assets/report/fig_cost.dark.svg#only-dark)
  <figcaption>Figure 8. Left: training throughput from Table 11. Right: forward throughput at different context lengths (the whole window is read at once and the loss computed at every position, 32,768 tokens per batch; TTT-E2E includes its test-time updates).</figcaption>
</figure>

<div class="report-table" markdown>

{% table sessions %}

</div>

Table 14. State kept by one inference session after reading contexts of different lengths, and the speed of token-by-token decoding that follows (batch size 1). State sizes are counted from the arrays actually allocated.

{% include parts/results_4_10.en.md %}

## 5 Data audit

<div class="report-table" markdown>

{% table overlap %}

</div>

Table 15. Verbatim overlap between evaluation items and the training corpora. For each item, the first 80 characters of the normalised question text (lower-cased, whitespace collapsed) are matched exactly against all training text ({{ GB(r.audits.corpora["fineweb-edu-train"].utf8_bytes + r.audits.corpora["pg19-train"].utf8_bytes) }} GB).

{% include parts/results_5.en.md %}

## 6 Discussion and limitations

{% include parts/discussion.en.md %}
