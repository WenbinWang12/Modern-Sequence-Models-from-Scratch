# Why Train a Base Model from Scratch?

A base model is the starting point for later SFT, preference optimization, and deployment. Training one from scratch gives us control over data, tokenizer, architecture, training recipe, and system cost.

The book follows this pipeline:

```text
Research goal → Data → Tokenizer → Model → Objective → Optimizer
→ GPU/runtime → Checkpoints → Evaluation
```

We begin with tiny and small configurations because they make bugs and causal mistakes observable. Larger runs are introduced only after the data, training, recovery, and evaluation contracts are stable.

TTT is one frontier in a broader sequence-model story. Attention, linear attention, delta rules, TTT, recurrent depth, and MoE each change a different part of the memory or compute trade-off.

The Chinese edition contains the longer conceptual introduction and the LaTeX/TikZ overview figure.
