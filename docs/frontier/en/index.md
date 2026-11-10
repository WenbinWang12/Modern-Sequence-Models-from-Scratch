# Frontier Lab

Frontier Lab is a dated research radar rather than a fixed textbook sequence. Each topic answers three questions: what new claim is being made, how it relates to the current TTT-E2E code, and whether it can be reproduced or challenged under a fixed budget.

## Topic lifecycle

```text
Radar → paper reading → minimal reproduction → compute-matched comparison
  ↑                                                        ↓
Archive ← no evidence / topic cooled ← decision record
```

Every topic records a snapshot date, paper version, code status, controls, evidence, and decision. A claim enters the mainline only after it reuses the main data, training, and evaluation contract.

## Current topics

| Topic | Question | Status |
|---|---|---|
| TTT and linear attention | Can TTT state updates be understood as linear attention, KV binding, or a delta rule? | reproducing |
| TTT-E2E and long context | Can test-time learning compress context into weights while remaining scalable? | mainline direction |
| Looped Transformer | Is repeated shared depth more cost-effective than adding independent parameters? | reproducing |
| TTT × Loop | Are time-axis fast-weight updates complementary to depth-axis recurrence? | controls pending |

The stable code remains under `ttt/` and `configs/`; topic pages live under `docs/frontier/`, and formal reports live under `docs/reports/`.
