# TTT-E2E and long context

**Snapshot:** 2026-11-07. **Status:** mainline direction.

## Claim

TTT-E2E describes long-context modeling as continual learning: while reading a test context, the model continues to update from next-token targets and compresses context into a weight state. The reported 3B and long-context results motivate this direction, but are not yet reproduced by this repository.

## Relation to the mainline

This topic has four evidence layers:

- **Function:** inner loop, caches, gradients, and restore behavior;
- **Scale:** first-step compilation and steady-state throughput on fixed hardware;
- **Long context:** real 32K/128K data, memory, stability, and quality;
- **Quality:** RULER or another formal benchmark.

The current repository covers small-scale function and runtime checks. Official 3B, 128K, and full RULER results require external weights, data, and hardware.

## Controls

Fix tokenizer, data manifest, sequence length, effective training tokens, hidden size, layers, initialization, optimizer, precision, actual FLOPs, chunk size, and hardware. Keep KV cache, fast weights, and session snapshots in explicit state schemas.

## Evidence and decision

Record numerical forward checks, causality, first-token latency, steady-state tokens/s, memory, fast-weight norms, post-restore loss, and long-context quality. A shape check on a tiny model supports only a functional claim. TTT-E2E enters the mainline as a quality conclusion only after function, cost, and quality evidence reproduce under the same contract.

## Next step

Compare dense, KV-cache, and TTT paths at the same budget; extend to 32K; then request 128K and RULER resources. Commit the configuration, data summary, and report for each step.
