# 19. Ablations, cost, and scaling

## Write the comparison protocol first

Names such as A, D, E, E-plus, S/C/F, and Loop have meaning only after their structural differences are explicit. A fair experiment fixes everything except the declared axis.

| Field | Meaning |
|---|---|
| variant | structural switch combination |
| tokens | effective training tokens |
| params | total and active parameters |
| flops | theoretical or profiled training FLOPs |
| compile_s | first compilation time |
| steady_tokens_s | steady throughput |
| peak_memory | peak memory |
| loss | training and held-out loss |
| status | success, pending, or failed with reason |

Equal parameter counts do not imply equal compute, memory, communication, or compile cost.

## Three axes

The structure axis changes TTT, MLP-prime placement, Loop placement, or MoE. The scale axis changes width, depth, context, tokens, or device count. The runtime axis changes prefill/decode, full/streaming forward, batch, or cache length.

Change one axis at a time. If several variables must change together, make the joint change explicit in the manifest.

## Code and reports

ttt/experiments.py creates ablation configurations and cost summaries. ttt/profile.py records parameter, state, compile, and execution cost. ttt/scaling.py aggregates N/D scaling data and marks missing inputs.

~~~bash
python3 -m ttt.experiments \
  --output configs/ablations/generated \
  --sizes 125m 760m \
  --include-moe

python3 -m ttt.profile \
  --checkpoint CHECKPOINT \
  --context-length 1024 \
  --decode-tokens 16 \
  --training-flops

python3 -m ttt.scaling \
  --input scaling.jsonl \
  --output outputs/scaling.json
~~~

Without profiler or XLA cost inputs, keep the result pending rather than filling in estimates. A publishable result links config, commit, manifest, tokens, loss, hardware, measurement method, and failure records.

## Transition

The next chapter separates functional GPU checks, distributed restore, and target-scale training.
