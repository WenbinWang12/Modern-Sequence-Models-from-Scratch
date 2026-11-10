# 12. Data mixtures and deterministic sampling

## File size is an accidental training decision

Imagine 900 GB of web text, 50 GB of code, and 50 GB of books. Concatenating files gives approximately 90%, 5%, and 5% of the token stream. That may be a sensible choice, but it is a consequence of storage size, not an explicit scientific hypothesis.

The opposite mistake is to sample documents uniformly. A short code snippet and a 200-page book chapter then receive equal document weight even though their token counts differ greatly. A mixture must state what is being weighted: documents, tokens, or examples.

## A recipe is a probability distribution plus a budget

Write the intended mixture as data, for example:

~~~
{
  "sources": [
    {"name": "web", "weight": 0.70},
    {"name": "code", "weight": 0.20},
    {"name": "books", "weight": 0.10}
  ],
  "sampling": "temperature",
  "temperature": 0.7,
  "seed": 17,
  "target_tokens": 1000000000
}
~~~

Here <code>weight</code> is the intended probability before exhaustion and filtering. <code>target_tokens</code> states how much training data the plan can consume. The plan must also specify what happens when a source runs out: stop, resample with replacement, or renormalize among remaining sources.

Keep two tables. The **recipe table** records the target distribution. The **consumption table** records actual source tokens, valid masked tokens, padding, and dropped examples. A 70/20/10 recipe can produce a different observed ratio when one source is exhausted or many of its tokens are masked.

## Temperature needs a declared formula

“Temperature sampling” is ambiguous unless the exponent is written down. One common convention is

$$
p_i=\frac{w_i^{1/\tau}}{\sum_j w_j^{1/\tau}},
$$

while another uses $p_i\propto w_i^{\tau}$. They have opposite behavior for $\tau<1$. With weights $(0.8,0.15,0.05)$ and the first convention at $\tau=0.5$, the unnormalized values are $(0.64,0.0225,0.0025)$, which makes the largest source more dominant. With the second convention, the unnormalized values are approximately $(0.894,0.387,0.224)$, which raises the relative share of smaller sources after normalization. These numbers illustrate why a paper must name the formula rather than only reporting “temperature 0.5”.

The mixture distribution is not the same as the final loss distribution. If each sampled window has a mask $m$, the effective token count from source $i$ is

$$
N_i^{\mathrm{effective}}=\sum_{t\in i}m_t,
$$

not simply the number of windows selected from that source.

## Determinism across ranks

Distributed training adds a choice that is easy to miss. Do we select a source globally and then partition its samples across ranks, or let every rank select independently? The first avoids some duplicated choices but requires coordinated indices; the second is simpler but changes the global distribution and communication pattern.

A clear design derives a source and window from a global sample index, seed, and recipe version, then assigns that index to a rank. The function can be described as

$$
(\text{source},\text{window})=f(\text{seed},\text{recipe},\text{global\_index}).
$$

Restoring a run then means restoring the global step and recipe fingerprint, rather than hoping that each rank's private RNG happens to resume in the same order.

The current <code>BatchStream</code> samples windows from one logical token stream. It does not implement source-level mixture scheduling, temperature sampling, exhaustion policies, or rank-aware global indices. The JSON above is a next-stage design contract, not an existing feature. Experiments must not claim that the current sampler enforces it.

## A validation plan

Before training, generate a fixed sample plan with a fixed seed and count planned source tokens. During training, log cumulative source tokens, effective mask tokens, padding, and dropped examples. After restore, concatenate the pre-restore and post-restore global sample indices and check for a duplicate or a gap.

Also report uncertainty over a finite window. If the target is 70/20/10, a 1,000-token window can deviate substantially even when the long-run ratio is correct. Use both cumulative proportions and a sliding-window interval so that a short burst of a small source is not mistaken for a recipe failure.

## Reproducible command skeleton

The repository does not yet expose a source-mixture command. A reproducible experiment can still store the proposed recipe next to the resolved experiment configuration:

~~~
python -c "import json; p='runs/recipe.json'; print(json.dumps(json.load(open(p, encoding='utf-8')), indent=2))"
~~~

Treat this as an inspection step until a sampler implementation is added. A recipe file without a sampler and consumption log is documentation, not evidence that the mixture was used.

The next chapter turns these contracts into a preflight gate before compilation and a long training run.

## Summary and next question

A mixture is an explicit probability distribution and token budget, not a side effect of file size. State the temperature convention, derive rank and recovery behavior from a global sample index, and record consumption rather than only weights. The next chapter combines tokenizer, manifest, masks, and recipe into a preflight gate.
