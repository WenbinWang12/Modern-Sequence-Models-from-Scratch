# 20. Dense Causal-LM Baseline

TTT, linear attention, loops, and MoE need a reliable control group. Start with a dense causal language model and fix the data, tokenizer, optimizer, seed, training tokens, and hardware.

```text
num_experts = 0
use_fast_mlp = false
inner.enabled = false
loop_mode = none
```

Verify that loss decreases, gradients are finite, checkpoints recover, causal outputs agree between training and streaming inference, and shape constraints remain valid. Parameter count alone is not a fair compute comparison; record FLOPs, compile time, steady-state throughput, and peak memory.
