# From Scratch: Modern Sequence Models

This is the English edition of the experiment book. It follows the same code, configurations, remote runs, and evidence as the Chinese edition.

Every chapter uses the same narrative:

```text
Question → Theory → Code → Remote experiment → Evidence → Next question
```

The stable path starts with the training system and data, establishes a dense causal-language-model baseline, and then moves to TTT-E2E, streaming state, sparse models, scaling, and GPU validation. The Frontier Lab tracks fast-moving topics such as TTT and linear attention, recurrent-depth Transformers, and long-context learning. A chapter is complete when its code path, command, and evidence are clear, including the limits of what has been verified.

## Book map

| Part | Question | Main artifacts |
|---|---|---|
| Prologue | Why train a base model from scratch? | system map and research boundary |
| Part I | How do we make a run start, resume, and explain itself? | experiment contract, environment, config, checkpoints, provenance |
| Part II | What data distribution does the model learn? | source records, cleaning, tokenizer, manifests, preflight |
| Part III | Is the dense causal baseline trustworthy? | fixed-budget baseline |
| Part IV | Where do TTT fast weights update? | forward path and inner/outer loops |
| Part V | How do caches, streaming, and experts change the system? | runtime, MoE, mesh, restore reports |
| Part VI | How do we separate architecture gains from cost? | ablation matrix, profiling, GPU validation |

The stable path establishes the protocol first. Frontier Lab can move faster, but its claims should reuse the mainline data, training, and evaluation contract before they are compared with the baseline.

Start with [Why train a base model](00_why_pretrain.md), then read [The experiment contract](01_experiment_contract.md).
