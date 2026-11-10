# 12. Data mixtures: deciding what the model reads

## Storage size cannot choose the recipe

Once web text, code, and books are ready, one question remains: how much of the next billion training tokens should come from each source? Putting all files in one directory does not answer it.

Suppose web text occupies 900 GB and code and books occupy 50 GB each. Web text accounts for **90% of bytes**, not necessarily 90% of tokens. Languages, formats, and tokenizers can have different token counts per byte. We must tokenize consistently before discussing training shares. For a concrete teaching example, instead assume the sources contain 800, 150, and 50 million usable tokens. Their concatenated token shares are then \(0.8,0.15,0.05\).

A large source is not necessarily more useful; it may simply be easier to obtain. Equalizing sources is not automatically better either, since a small collection may be read repeatedly. A recipe must address both coverage and repeated use.

## Specify the sampling unit

“Code is 20%” could mean 20% of documents, windows, or valid training positions. These coincide only under additional assumptions.

Suppose we select a source and then sample one window from it. If every window contains exactly \(L\) valid targets, source-selection probability \(p_j\) also determines the long-run valid-token share. If source \(j\) contributes an average of \(k_j\) valid targets per window, its valid-token share is instead

$$
q_j=\frac{p_j k_j}{\sum_i p_i k_i}.
$$

This follows from counting: over \(n\) selections, source \(j\) is selected \(np_j\) times in expectation and contributes approximately \(np_jk_j\) valid positions. Dividing by the total cancels \(n\).

For example, two sources may each receive half the windows, while contributing 100 and 25 valid positions per window. Their valid-token shares become \(50/(50+12.5)=80\%\) and \(20\%\). The sampler has not failed; the mistake is equating half the windows with half the valid tokens. Both recipe and consumption report must specify their unit.

## Control source proportions with an exponent

Normalize the available token counts:

$$
w_j=\frac{N_j}{\sum_i N_i},
$$

where \(N_j\) is source \(j\)'s usable token count. One explicit way to reduce the dominance of large sources is

$$
p_j(\alpha)=\frac{w_j^\alpha}{\sum_i w_i^\alpha},
\qquad \alpha>0.
$$

We call \(\alpha\) the mixture exponent rather than relying on an ambiguous temperature convention. At \(\alpha=1\), proportions remain unchanged. Values between zero and one compress disparities, while values above one amplify them. For two positive weights,

$$
\frac{p_i}{p_j}=\left(\frac{w_i}{w_j}\right)^\alpha.
$$

If \(w_i>w_j\) and \(0<\alpha<1\), the new ratio is smaller than the original ratio, although the larger source remains larger. As \(\alpha\) approaches zero from above, positive-weight sources approach equal probability. An empty source still cannot be sampled.

For \(w=(0.8,0.15,0.05)\) and \(\alpha=0.5\),

$$
w^\alpha\approx(0.8944,0.3873,0.2236),\qquad
p\approx(0.5942,0.2573,0.1485).
$$

Books rise from 5% to about 14.85%. The cost is also visible: a billion-token budget would consume about 148.5 million book tokens from a collection containing only 50 million, or 2.97 times its available volume. This estimates average coverage; overlapping random windows do not read every position exactly three times.

Some systems use \(w_j^{1/T}\) instead. Their temperature \(T\) is the reciprocal of our exponent, so \(T<1\) has the opposite effect. Reproduction requires the formula and parameter, not just “temperature 0.5.”

## Budget and source exhaustion

With total valid-token budget \(N_{\mathrm{budget}}\), the ideal allocation is \(q_jN_{\mathrm{budget}}\). If this exceeds the available data, decide in advance whether to stop training, sample with replacement, or renormalize among surviving sources.

These choices define different experiments. Stopping changes the total budget, replacement changes repetition, and renormalization changes the later distribution. Silently switching sources after exhaustion makes the original recipe an inaccurate description of the run.

Logs should distinguish selected windows, input tokens, and summed valid mask weights. With binary masks the last quantity counts valid positions; fractional masks instead require the term weight sum. Source identity must also survive packing: a window crossing sources cannot be assigned wholly to the source of its first token.

## A recipe calculation to inspect and save

The current trainer samples from one logical token stream and does not consume a source-mixture recipe. The following standalone calculation runs on the target Linux machine after environment setup. It saves the example above without changing training behavior.

~~~bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

counts = {"web": 800_000_000, "code": 150_000_000, "books": 50_000_000}
alpha = 0.5
total = sum(counts.values())
weights = {name: (count / total) ** alpha for name, count in counts.items()}
normalizer = sum(weights.values())
recipe = {
    "kind": "teaching_plan_not_connected_to_trainer",
    "formula": "p_j = (N_j / sum(N)) ** alpha / sum_i((N_i / sum(N)) ** alpha)",
    "alpha": alpha,
    "selection_unit": "equal_valid_length_window",
    "target_tokens": 1_000_000_000,
    "exhaustion_policy": "with_replacement",
    "sources": [
        {"name": name, "available_tokens": counts[name], "probability": value / normalizer}
        for name, value in weights.items()
    ],
}
path = Path("outputs/mixture-example/recipe.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
for source in recipe["sources"]:
    print(source["name"], f'{source["probability"]:.4f}')
PY
~~~

The formula predicts printed probabilities of approximately 0.5942, 0.2573, and 0.1485, in that order. The JSON lets us inspect the calculation and budget. It does not establish that training used the recipe: that requires a sampler consuming the definition and logs recording actual consumption.

## Why short-run proportions fluctuate

A source probability of \(p=0.2\) does not force exactly one selection in every five. Assume \(n\) independent selections with unchanged probability. Let \(I_k\) be 1 when selection \(k\) chooses the source and 0 otherwise. The observed proportion is

$$
\hat p=\frac{1}{n}\sum_{k=1}^{n} I_k.
$$

Since \(\mathbb{E}[I_k]=p\) and \(\operatorname{Var}(I_k)=p(1-p)\), independence lets us add variances:

$$
\mathbb{E}[\hat p]=p,\qquad
\operatorname{Var}(\hat p)=\frac{p(1-p)}{n}.
$$

At \(n=100\), the standard deviation is 0.04, or four percentage points. Observing 17% once does not establish a sampler bug. This calculation counts independent source selections; 1,000 correlated tokens within one window are not 1,000 independent selections.

## Restoring the same distributed plan

Processes on multiple GPUs need assigned sample identities. One design derives the source and window from a global sample index, then distributes disjoint indices across processes:

$$
(\text{source},\text{window start})
=f(\text{seed},\text{recipe version},\text{data version},\text{sample index}).
$$

This describes a proposed source-level design. The current code already has stateless sampling for single-stream windows, but lacks the complete source selection, exhaustion handling, and consumption log. Restoring only the integer step is insufficient: the function and its version, data fingerprint, global batch, and index assignment must also agree.

Compare the source and start-index sequence of an uninterrupted run with an interrupted-and-resumed run. Sample indices must not be skipped or duplicated. Sampling with replacement may legitimately assign the same window to different indices; that is a revisit, not a restoration error. The next chapter brings these recipe, data, and model conditions together before training.
