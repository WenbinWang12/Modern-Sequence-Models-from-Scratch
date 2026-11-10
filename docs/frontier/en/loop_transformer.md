# Looped Transformer: what exactly gets repeated?

**Source snapshot: 2026-11-11. Code review baseline: <code>567fef3</code>. Status: three local forms of shared repetition implemented; paper protocol and quality comparison not reproduced.**

## Why apply the same parameters twice?

A second reading of a draft may reveal a problem missed on the first pass. Shared-depth computation has a similar intuition: the same transformation acts on a representation that has changed, potentially extracting something its first application missed. This motivates a question; it does not prove that a second application helps. A transform that discards information or amplifies error can become worse when repeated.

An ordinary deep Transformer stores different parameters for different layers. A looped Transformer reuses parameters over part of its depth. It may reduce the number of independent parameters, but repeated computation still consumes time and may require saved activations in training. The question is whether that effective depth is worthwhile under a stated compute budget. Parameter count, speed, and quality need separate measurements.

## Source and local hypothesis

This section pins [Reasoning with Latent Thoughts: On the Power of Looped Transformers, v1, 2025-02-24](https://arxiv.org/abs/2502.17416v1), which studies the expressive and computational capabilities of looped Transformers. It motivates our local experiment; the repetition implemented below is not presented as a complete reproduction of the paper architecture or protocol.

Let the initial representation at a token position be \(h^{[0]}\), and let \(F_\theta\) be a shared module. Repeating it \(r\) times along depth gives

$$
h^{[j+1]}=F_\theta(h^{[j]}),\qquad j=0,\ldots,r-1.
$$

Here \(j\) counts repetitions; token position is still \(t\). The module may act on a whole sequence, whose dimension is suppressed. Each application uses the same \(\theta\), so its independent parameter count does not grow with \(r\). It still performs \(r\) transformations and may save intermediate values for each during training. Sharing parameters is different from sharing activations.

TTT instead updates fast weights \(\phi_c\) as chunks \(c\) arrive. Loop changes depth computation; TTT changes weight state while consuming context. They can be combined, but each needs a separate baseline before a combined gain can be attributed. Readers unfamiliar with gradients and state can first read [inner and outer loops](../../book/en/05_inner_outer_loop.md).

## What the code already implements

<code>ttt/models/ttt_e2e/modeling_ttt_e2e.py</code> repeats the same parameter object for the operations below. The earlier instruction to “validate <code>loop_mode</code>, then add real shared layers” was inaccurate. Sharing already exists; what remains is the full architecture, training controls, and quality evidence corresponding to a specified paper.

| Configuration | Implemented computation | Repeated scope |
|---|---|---|
| <code>none</code> | Each layer executes once | None |
| <code>early_ffn</code> | Repeat the static FFN branch within a layer | FFN in each selected layer |
| <code>middle_attention</code> | Repeat the attention branch within a layer | Attention in each selected layer |
| <code>middle_block</code> | Repeat a complete individual block | Each selected layer separately |

Repeating each layer differs from repeating a stack of layers. Selecting layers 0 and 1 with two repeats currently gives \(0\to0\to1\to1\) in <code>middle_block</code>; looping the entire two-layer module twice gives \(0\to1\to0\to1\). Equal layer and parameter counts do not remove this difference in execution order. The current mode cannot automatically stand in for an arbitrary recurrent-depth paper.

The configuration uses the half-open interval <code>[loop_start, loop_end)</code>, restricted to the static prefix. Attention and complete-block repetition store separate caches for different visits, preventing histories from different depth positions from being mixed. Shared parameters therefore do not imply a shared KV cache; more repetitions can increase cache and activation cost.

## An auditable minimal baseline

First disable fast weights and inner updates, then change only the first layer's repeat count. Run the following from the repository root on a prepared remote GPU node. These commands were not executed during this editing pass. Sample text checks the training pipeline; it is not held-out evaluation.

~~~bash
mkdir -p outputs/frontier-loop
git rev-parse HEAD > outputs/frontier-loop/source-commit.txt
bash scripts/prepare_sample.sh outputs/frontier-loop/sample.npy
.venv/bin/python - <<'PY'
import copy
import json
from pathlib import Path

root = Path("outputs/frontier-loop")
base = json.loads(Path("configs/tiny.json").read_text())
base["model"]["use_fast_mlp"] = False
base["model"]["ttt_suffix_length"] = 0
base["inner"]["enabled"] = False
for repeats in (1, 2, 4):
    config = copy.deepcopy(base)
    config["model"].update(
        loop_mode="none" if repeats == 1 else "middle_block",
        loop_start=0, loop_end=1, loop_repeats=repeats,
    )
    (root / f"r{repeats}.json").write_text(json.dumps(config, indent=2))
PY
for r in 1 2 4; do
  bash scripts/run_gpu_train.sh \
    --config "outputs/frontier-loop/r$r.json" \
    --data outputs/frontier-loop/sample.npy \
    --output "outputs/frontier-loop/run-r$r"
done
~~~

All three models have two independent blocks. Only the first executes 1, 2, or 4 times, giving 2, 3, or 5 total block visits, not 2, 4, or 8. The <code>none</code> group is a functional baseline with neither TTT nor depth repetition. Equal training tokens generally give unequal FLOPs across these groups; this command does not yet match compute budgets.

Each <code>run-r*</code> directory should contain <code>experiment.json</code>, <code>environment.json</code>, <code>metrics.jsonl</code>, <code>status.json</code>, <code>latest.json</code>, and checkpoints. With the current four-step NPZ recipe in <code>configs/tiny.json</code>, the final checkpoint is <code>step_00000004.npz</code>. If steps or format change, consult <code>latest.json</code>. Expect status <code>complete</code>, not a predetermined loss value.

Separately, <code>bash scripts/validate_gpu.sh outputs/frontier-loop/functional.json</code> checks gradients, causality, and streaming consistency for the three Loop modes. This tool uses its own tiny configurations, so its result is not a direct validation of each training configuration above. Neither check supplies a reasoning-task score.

## What a fair comparison controls

The first formal comparison fixes data splits, tokenizer, independent parameters, initialization policy, training tokens, optimizer, and precision, recording the cost of repetition. It asks what happens when the same model performs more computation. The second matches total training FLOPs, permitting changed step counts while explicitly reporting the resulting token counts. It asks how a fixed budget should be allocated. These are separate questions and need separate reports.

Also fix sequence length, batch, device, cache policy, residual and normalization placement, and any supervision at intermediate repetitions. The commands here use the final language-model loss without extra per-repeat supervision. Changing repeat counts during training and changing them only at evaluation are also different experiments.

Quality measurements include held-out cross-entropy and task scores. Resource measurements include independent parameters, training tokens, FLOPs, compilation time, warmed throughput, peak memory, and cache size. Four sample steps can reveal execution errors but cannot establish generalization. A lower loss with more repeats must be reported alongside its extra computation.

| Question | Current evidence | Decision |
|---|---|---|
| Does <code>loop_mode</code> really share parameters? | The same parameter object is called repeatedly; source is inspectable | Implemented; execution report still needed |
| Are the specified paper's loop and results reproduced? | Current repetition is local to individual layers; complete protocol missing | No reproduction claim |
| Does repetition improve quality under fixed compute? | No training/evaluation result under this protocol | Unresolved |
| Do TTT and Loop complement one another? | Composable components exist; four-way comparison incomplete | Planned experiment |

## Failure criteria, gaps, and the next run

Nonfinite training, failure to write a complete checkpoint, or disagreement between streaming and corresponding full execution is an implementation or configuration failure to resolve first. If a complete budget-matched comparison shows no stable quality gain, it does not support “repetition is more cost-effective” in that setting. Selecting only a longer run or a lucky seed cannot repair that conclusion.

This protocol still lacks multiple-seed quality reports, accurate cross-configuration FLOPs matching, and an implementation and reproduction record for looping a whole multilayer stack. The next run preserves all three tiny outputs, verifies shared-mode parameter counts and execution order, then selects real training and held-out data. The TTT×Loop four-way comparison follows those steps.
