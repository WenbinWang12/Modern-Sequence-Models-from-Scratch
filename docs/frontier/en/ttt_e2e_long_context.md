# TTT-E2E and long context: from correct state to quality evidence

**Source snapshot: 2026-11-11. Code review baseline: <code>567fef3</code>. Status: training and runtime paths implemented; 32K/128K quality reproduction incomplete.**

## Accepting a long input is not the conclusion

Suppose an answer appears at the start of a book and the question arrives at its end. A model may accept the whole book yet lose the answer. Even a correct answer may come from training contamination. A long-context experiment must establish useful access to history, absence of future leakage, and acceptable resource cost.

This section uses the state definitions from [inner and outer loops](../../book/en/05_inner_outer_loop.md) and [KV caches](../../book/en/06_attention_cache.md). Its goal is an evidence chain from tiny correctness checks to target-length quality evaluation, not an unmeasured promise about sufficient GPU hardware.

## Source and hypothesis

[End-to-End Test-Time Training for Long Context, v2, 2025-12-31](https://arxiv.org/abs/2512.23675v2) studies next-token learning on test context and training an initialization suited to that learning. The reported 3B model, 164B training tokens, and 128K setting belong to the authors' experiments; they are not reproduced results from this repository.

The local hypothesis is that enabling updates for the same model and checkpoint improves predictions of subsequent unseen tokens or task scores on the same context, with value remaining after accounting for inner-loop computation. This is a paired evaluation, not a substitute for independently training and comparing two complete methods.

## Connecting state, length, and resources

Divide inputs into chunks \(c\), with static parameters \(\theta\) and fast weights \(\phi_c\). Here \(X_c,Y_c,M_c\) are the chunk's inputs, shifted targets, and loss mask. Predict using current state, then update from observed targets. When gradient clipping is inactive, SGD is

$$
\mathcal L_c=\mathcal L(X_c,Y_c,M_c;\theta,\phi_c),\qquad
\phi_{c+1}=\phi_c-\eta_{\mathrm{inner}}\nabla_{\phi_c}\mathcal L_c.
$$

The actual <code>sgd_update</code> also multiplies the gradient by \(\min(1,\tau/\sqrt{\|g\|_2^2+10^{-12}})\), where \(g=\nabla_{\phi_c}\mathcal L_c\) and \(\tau\) is <code>clip_norm</code>. A large gradient retains its direction but takes a smaller step. Only subsequent chunks use \(\phi_{c+1}\). Updating on an answer before scoring its own prediction leaks the label.

Fixed fast-weight dimensions do not imply fixed whole-model memory: KV history, prefix activations, convolution state, pending chunks, and compiler buffers also count.

For a single sample, ideal KV storage for one full-attention layer is approximately

$$
M_{\mathrm{KV}}=2L n_{\mathrm{kv}}d_h b_{\mathrm{dtype}},
$$

where \(n_{\mathrm{kv}}\) counts KV heads and \(b_{\mathrm{dtype}}\) is bytes per element. With \(n_{\mathrm{kv}}=12,d_h=64,b_{\mathrm{dtype}}=2\), 32,768 tokens require about 96 MiB per layer; 131,072 tokens require about 384 MiB. These figures count only KV, excluding weights, gradients, optimizer state, and activations. They are not measured peak memory.

Sliding-window layers can bound retained KV history, but allocation also depends on implementation. The first layer of <code>configs/runs/tiny_gpu.json</code> is full attention. A sliding-window suffix does not remove prefix history growth. The tiny script therefore cannot establish context-independent latency reported for the paper's setting.

## Implemented paths and gaps

| Item | Status | Code and scope |
|---|---|---|
| Chunk prediction and gradient updates | Implemented | <code>ttt/inner_loop.py</code>; incomplete chunks are scored without committing their update |
| Session prefill/decode | Implemented | <code>ttt/runtime.py</code>; carries fast weights, caches, and pending chunks |
| Functional checks at 64/512 | Entry point implemented | <code>ttt/validate.py</code>; random tiny weights, not quality data |
| Checkpoint loss and retrieval probes | Scaffold | <code>ttt/eval_retrieval.py</code>; custom probes are not the complete RULER suite |
| 32K/128K training and formal evaluation | Planned | Requires fixed configurations, data splits, resource measurements, and raw reports |

Local from-scratch training does not require official weights. Official-checkpoint comparison does require the files, access, tokenizer, and conversion validation. Record these as distinct tracks.

## Minimal run, outputs, and failure diagnosis

First run functional checks on a prepared remote GPU node:

~~~bash
mkdir -p outputs/frontier-context
git rev-parse HEAD > outputs/frontier-context/source-commit.txt
.venv/bin/python -m ttt.validate --require-gpu --lengths 64 512 \
  --output outputs/frontier-context/functional.json
~~~

Expect <code>passed: true</code> and cases <code>context_64</code> and <code>context_512</code> with finite gradients, streaming-logit errors, and state bytes. Current code fails when maximum logit error exceeds \(10^{-3}\), with separate causality assertions. Failure reports include <code>error_type</code> and <code>error</code>. This command was not executed during this editing pass.

After training produces a real checkpoint, the following template performs paired evaluation. Replace both paths with existing inputs; they are not artifacts claimed to exist in the repository.

~~~bash
CHECKPOINT=/absolute/path/to/trained-checkpoint
VALIDATION=/absolute/path/to/heldout-manifest.json
.venv/bin/python -m ttt.eval_retrieval --checkpoint "$CHECKPOINT" \
  --data "$VALIDATION" --mode loss --sequence-length 64 --samples 8 \
  --output outputs/frontier-context/adapt.json
.venv/bin/python -m ttt.eval_retrieval --checkpoint "$CHECKPOINT" \
  --data "$VALIDATION" --mode loss --sequence-length 64 --samples 8 --no-adapt \
  --output outputs/frontier-context/no-adapt.json
~~~

The difference answers only the effect of enabling updates for this checkpoint and sampling protocol. Training sample text is not held-out validation. Eight samples check the entry point but cannot establish robust quality. Position-limit, vocabulary, or manifest mismatches must be fixed rather than bypassed.

## Admission conditions for 32K and 128K

| Stage | Evidence required first | Typical failure | Next action |
|---|---|---|---|
| Tiny, 64/512 | Causality, gradients, full/streaming agreement | Nonfinite values, misaligned state | Repair correctness |
| 32K, specified model | Legal positions, real long examples, warmed throughput, peak memory | OOM, long compilation, insufficient data | Change model or budget and record a new configuration |
| 128K, specified model | Passing 32K, length-extension policy, complete budget, uncontaminated tasks/splits | Growing memory, retrieval degradation, timeout | Preserve failure; do not extrapolate success |
| Formal quality comparison | Multiple seeds/intervals, matched data and training cost | A single fluctuation presented as a gain | Add independent runs |

No GPU is declared “certainly sufficient”: precision, full-attention layer count, batch, compilation, and training versus inference all change requirements. Setting <code>--lengths 131072</code> still enlarges a tiny functional case; it does not run a 3B model or reproduce the paper.

## Metrics, decisions, and the next run

Record valid tokens, held-out loss, retrieval scores, warmed prefill/decode time, compilation time, peak memory, and state size together; missing values remain “not measured.” <code>ttt.profile</code> supplies some timing/state metrics, but profiling synthetic tokens does not measure task quality.

| Research conclusion | Current evidence | Decision |
|---|---|---|
| Does local code support TTT training and streaming state? | Code and checks exist; no new run report in this section | Proceed to tiny checks |
| Does 32K/128K improve quality or cost? | No results under this topic's fixed protocol | Unresolved |
| Is the official 3B result reproduced? | Complete weight/data/evaluation alignment is missing | No reproduction claim |

The next run saves the raw 64/512 report, then selects a trained small model for paired held-out evaluation with adaptation toggled. Increase length only after both steps have explainable inputs, outputs, and errors.
