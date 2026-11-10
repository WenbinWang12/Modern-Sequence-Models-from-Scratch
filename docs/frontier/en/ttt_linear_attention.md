# TTT and linear attention: similar, equivalent, or complementary?

**Snapshot:** 2026-11-07. **Status:** reproducing.

## Claim

TTT treats a small model's parameters as a state that is updated on the test sequence with a self-supervised loss. Later work suggests that some TTT forms can be reinterpreted as learned linear attention with a richer state parameterization. We treat that as a structural hypothesis to test, not as a general mathematical equivalence.

## Relation to the mainline

| Component | Linear attention | TTT |
|---|---|---|
| State | explicit key/value statistics | fast weights updated by loss gradients |
| Update | usually a recurrence | SGD or an approximation to it |
| Capacity | feature map and state shape | fast model and inner-loop steps |
| Cost | fixed recurrent update | extra gradient and update work |
| Training target | learn the recurrence | learn the post-update prediction |

## Controls and reproduction

Implement a linear-attention or delta-rule baseline first. Then match hidden size, state parameters, training tokens, actual FLOPs, chunk size, and hardware before comparing TTT-Linear. Report loss, long-context quality, first-token latency, steady-state tokens/s, memory, and state norms. A strict equivalence claim additionally requires checking the update equations under a linear model and the same loss.

The repository already contains `gated_delta.py`, chunked attention, and TTT inner-loop code. The independent linear-attention baseline and equivalence experiment remain to be implemented.

## Decision and next step

Do not merge this topic into the stable architecture until the compute-matched comparison and the equation-level check agree. The next step is a minimal state-size-matched baseline with a written manifest.
