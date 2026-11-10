# Looped Transformer: recurrence along depth

**Snapshot:** 2026-11-07. **Status:** reproducing.

## Claim

Looped Transformer reuses the same layers to increase effective computation depth without adding an independent parameter set for every repetition. The reported benefit depends on supervision, loop count, residual and normalization placement, and a fair compute budget.

## Relation to TTT

TTT adapts state along the time axis as tokens arrive. Looping repeats computation along the depth axis:

```text
TTT:  time-axis state learning, updating fast weights per token or block
Loop: depth-axis repeated computation, executing a shared block several times
```

| Model | Time update | Depth recurrence |
|---|---:|---:|
| Dense baseline | no | no |
| Looped Transformer | no | yes |
| TTT-E2E | yes | optional |
| TTT + Loop | yes | yes |

## Controls and evidence

Fix unique parameters, total training FLOPs, effective depth, loop count, sequence length, training tokens, supervision at each loop, residual/norm placement, and whether inference may change the loop count. Record compile time, steady-state time, memory, loss, and quality for every loop count. Parameter matching alone overstates loop efficiency; one-forward matching understates its runtime cost.

The current configuration has `loop_mode` for functional checks. A true shared-layer recurrent-depth experiment still needs an explicit configuration and a compute-matched report.

## Decision and next step

First validate the loop state and accounting in isolation. Then combine loop recurrence with TTT fast-weight updates and test whether the two mechanisms help or interfere.
