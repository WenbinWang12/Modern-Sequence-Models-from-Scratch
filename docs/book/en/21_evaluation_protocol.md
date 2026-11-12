# Evaluation and statistics: making loss support a claim

## Evaluation is more than the last training-log column

Training loss asks whether the model fits the distribution it repeatedly sees. It does not by itself establish generalization to held-out text, freedom from contamination, task ability, or whether a new module is worth its additional computation.

This chapter organizes evidence into data, metrics, statistics, and cost. Ablations and Frontier pages should reuse this protocol. The repository's tiny runs and retrieval probes establish entry points; they do not automatically establish a formal quality result.

## Fix the evaluation object first

| Object | Question it can answer | Question it cannot answer alone |
|---|---|---|
| train | Did the optimizer reduce its training objective? | Generalization, contamination, broad ability |
| validation | How should a configuration or training stage be selected? | An unbiased final report |
| test | How does a frozen choice perform? | Repeated tuning |
| task probe | How does one declared task format behave? | All language ability |
| long-context probe | What happens under one context and memory protocol? | General reasoning |

Once a test set influences model selection, it has taken the role of validation. Every evaluation should record the data version, tokenizer identity, context length, mask rule, and whether TTT updates are allowed.

## Token-weighted NLL and perplexity

For target \(y_t\), the negative log-likelihood is

$$
\ell_t=-\log p_\theta(y_t\mid x_{\le t}).
$$

Let \(m_t\in[0,1]\) be the supervision weight and \(W=\sum_t m_t\). The token-weighted NLL over an evaluation set is

$$
\mathcal L_{\mathrm{tok}}
=\frac{\sum_t m_t\ell_t}{\sum_t m_t}
\qquad W>0.
$$

Perplexity is its exponential transform:

$$
\operatorname{PPL}=\exp(\mathcal L_{\mathrm{tok}}).
$$

Do not average per-window losses equally unless every window has the same valid weight. Save the loss numerator, denominator, window count, and supervised count. A fractional mask reports a weight sum, not an invented integer token count.

## Document boundaries and contamination

A continuous token stream can cross EOS and can make training and evaluation share neighboring content. The protocol must state:

1. whether documents are encoded independently;
2. whether EOS, BOS, and padding contribute to loss;
3. whether windows can cross document boundaries;
4. whether fast weights reset between documents;
5. whether the evaluation tokenizer was trained only on the training split and then frozen.

No exact duplicate inside validation does not prove that it has no training mirror or near duplicate. Reports should list the exact, normalized, long-substring, or approximate checks that ran, along with checks that were not performed.

## Generation, retrieval, and long-context tasks

Generation fixes the prompt, sampling parameters, maximum new tokens, stop token, and random seed. Temperature, top-k, and top-p change the output distribution, so “the same model generated better text” is not a complete protocol.

Retrieval probes may use exact match, normalized string match, or token-level F1, but the answer normalization must be written down. Answers must not leak through training data or tokenizer-training text. A small custom probe answers only its own question; it is not a complete benchmark.

For long-context TTT evaluation, pair two runs using the same model and checkpoint. Run the same context with updates disabled and enabled, compare later unseen targets or task scores, and record extra inner-loop time, update count, and state size. This measures the effect of updates for one checkpoint and protocol; it is not a comparison of two methods independently trained from scratch.

## Multiple seeds and uncertainty

A single run can differ because of initialization, data order, or device reductions. Record the seed, data fingerprint, configuration digest, and hardware. With multiple seeds, preserve every raw result and the aggregate:

$$
\bar x=\frac1n\sum_{i=1}^{n}x_i,
\qquad
s^2=\frac1{n-1}\sum_{i=1}^{n}(x_i-\bar x)^2.
$$

The mean and standard deviation describe seed variation; they are not automatically a per-token confidence interval. For document- or sample-level metrics, bootstrap intervals can resample independent units. Do not treat correlated tokens from one window as independent documents.

## A shared result record

A minimal result object should lead back to its inputs and code:

~~~json
{
  "source_commit": "...",
  "config_sha256": "...",
  "data_manifest_sha256": "...",
  "tokenizer": {"name": "...", "revision": "...", "digest": "..."},
  "split": "validation",
  "sequence_length": 2048,
  "adaptation": {"enabled": false, "inner_steps": 0},
  "metrics": {
    "nll_numerator": 0.0,
    "supervision_weight": 0.0,
    "perplexity": 0.0
  },
  "seed": 0,
  "status": "not_checked"
}
~~~

The zero values are schema examples, not measurements. A formal artifact should also preserve raw per-seed JSON, log summaries, failure reasons, and cost fields such as steady-state step time, supervised tokens per second, peak memory, and inner-loop work.

## Current repository boundary

The current eval_retrieval.py and Frontier pages provide checkpoint loss, small retrieval probes, and an entry point for paired long-context evaluation. They do not implement complete RULER, a shared multi-seed aggregator, a contamination database, or an automatic bootstrap report. The scaling fitter also does not produce training-quality results.

The code can therefore support the claim “this declared evaluation entry point ran.” It cannot support “the model is stronger on a formal benchmark” or “the architecture is broadly better at equal cost” without the missing evidence. When evidence is absent, write not_checked instead of a predicted number.

## Summary

Evaluation fixes the object and contamination boundary, defines an effective-weight metric, and records task protocol, seeds, uncertainty, and cost. Only when these fields travel with the configuration, data, and tokenizer identity does a loss become an auditable result.
