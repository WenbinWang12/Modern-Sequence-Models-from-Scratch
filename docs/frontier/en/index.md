# Frontier Lab: turning new claims into testable questions

**Source snapshot: 2026-11-11. Code review baseline: <code>0614bd5</code>. Status: research protocols and partial implementations; paper reproduction is not claimed.**

A lower loss in a paper does not imply that inserting its module into an existing model produces the same gain. Data, training budget, update order, and inference implementation may differ. Frontier Lab makes these differences explicit so readers can distinguish a mathematical relationship, an implemented relationship, and a gain supported by training experiments.

This part relies on next-token loss, gradients, causal attention, and experiment records. Read the [baseline](../../book/en/20_baseline_model.md), [inner and outer loops](../../book/en/05_inner_outer_loop.md), and [attention caches](../../book/en/06_attention_cache.md) first when needed. The minimal runs establish a research starting point; they are neither exercises nor reproduced result tables.

## Sources, claims, and local status

Paper links are pinned to versions so a later update does not silently change the referenced argument. Status below comes from source inspection, not execution during this editing pass.

| Topic | Versioned source | Question | Current status |
|---|---|---|---|
| [TTT and linear attention](ttt_linear_attention.md) | [TTT v4](https://arxiv.org/abs/2407.04620v4), [KV binding v4](https://arxiv.org/abs/2602.21204v4) | Can a particular fast model and update be expressed as an explicit recurrence? | Gated Delta and inner-loop code exist; an independent equivalence experiment is pending |
| [TTT-E2E and long context](ttt_e2e_long_context.md) | [TTT-E2E v2](https://arxiv.org/abs/2512.23675v2) | Does adapting weights improve long-context quality and cost? | Training, sessions, and tiny validation exist; a formal quality comparison is pending |
| [Looped Transformer](loop_transformer.md) | [Loop v1](https://arxiv.org/abs/2502.17416v1) | Is the effective depth from shared parameters worth its added computation? | Per-layer shared repetition exists; the paper protocol and quality comparison are not reproduced |

**Implemented** means an executable code path exists. **Scaffold** means an entry point or configuration exists but controls and evidence are incomplete. **Planned** means the experiment has no corresponding implementation yet. Apply these labels to individual capabilities, rather than marking an entire topic “reproducing.”

## Why a separate research track is useful

The main book builds a trustworthy experiment system. Frontier topics ask which new hypotheses that system should investigate. For example, both TTT and Loop add computation. TTT changes weight state as context arrives; Loop repeats computation along depth. Calling both “thinking longer” hides their different state and budget semantics.

Each topic therefore starts with a small question and an implementation that matches it. A one-token identity can check an update rule; a tiny model can check gradients and streaming state. Comparing quality requires a complete training and evaluation protocol. Passing the first two levels does not establish the third.

## A common minimal starting point

Run these commands from the repository root on a remote GPU node prepared using the [environment chapter](../../book/en/01_environment.md). CPU checks belong on the authorized remote CPU machine. These experiments were not executed during this editing pass.

~~~bash
mkdir -p outputs/frontier-smoke
git rev-parse HEAD > outputs/frontier-smoke/source-commit.txt
bash scripts/validate_gpu.sh outputs/frontier-smoke/validation.json
~~~

The script uses tiny architectures defined in code, with default context checks at 64 and 512 tokens. It also includes Gated Delta, three Loop modes, MoE, and streaming consistency cases. It does not load a 125M recipe or perform formal data evaluation. The internal case named <code>dense</code> is not automatically a “TTT-disabled training baseline”: inspect the configuration to determine whether fast weights and updates are active.

Expected artifacts are <code>environment.json</code>, <code>source-commit.txt</code>, and <code>validation.json</code>. The report should have <code>passed: true</code>; cases contain <code>finite_gradients</code>, <code>logits_max_error</code>, compilation time, and state bytes. Causality is checked by an assertion, not a separate success field.

On failure, inspect <code>error_type</code> and <code>error</code>, then the backend, GPU, and failing case. If the process is killed before writing a report, preserve scheduler logs. A missing JSON file is not a pass.

## Controls, metrics, and decision records

Quality experiments fix data splits, tokenizer, initialization policy, valid-token budget, and optimizer. Cost experiments also fix hardware, precision, batch size, sequence length, and timing window. Equal parameter counts do not imply equal FLOPs, and equal FLOPs do not imply equal wall time; record all three.

| Evidence level | Required measurements | Supported conclusion | Unsupported conclusion |
|---|---|---|---|
| Algebra | State/output errors and assumptions | Correspondence under those assumptions | Every TTT variant is equivalent |
| Function | Finite gradients, causality, streaming error | Consistent execution for this configuration | A quality gain |
| Cost | Compilation, steady throughput, state/peak memory | Resource cost on this machine | The same ratio on other hardware |
| Quality | Fixed held-out loss, task scores, seed variation | Comparison under this protocol | Universal architecture superiority |

Every formal result must also identify its configuration, data fingerprints, and raw reports. No new measurements are reported here; missing results remain “not run,” rather than predicted values. The next run completes each topic's minimal check before deciding whether to scale up.
