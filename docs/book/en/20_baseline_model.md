# 20. Dense causal-language-model baseline

## Why the baseline comes first

TTT, linear attention, Loop, and MoE need a trustworthy dense reference. The baseline fixes data, tokenizer, optimizer, training budget, and evaluation before adding new state or routing.

~~~text
num_experts = 0
use_fast_mlp = false
inner.enabled = false
loop_mode = none
~~~

Verify finite decreasing next-token loss, stable gradients, checkpoint recovery, shape changes, causal consistency, and a held-out distribution that is distinct and traceable.

## Code and fair comparison

The dense model lives under ttt/models; training and checkpoints are handled by ttt/training.py and ttt/checkpoint.py. Use the same manifest and preflight as later models. Start with tiny, then scale to 125M by changing only declared configuration fields.

For every comparison, fix initialization rules, tokenizer, manifest, effective tokens, optimizer, seed, and hardware. Report parameters, active parameters, FLOPs, memory, communication, compilation, and inference cost. A lower loss alone is not an efficiency claim.

The next part returns to the TTT forward path with a baseline that has already earned trust.
