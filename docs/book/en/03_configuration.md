# 14. Configuration: one contract for every tensor shape

## Why configuration must be explicit

Suppose a model silently chooses the number of heads from a tensor shape inside `forward`. The JSON may say `hidden_size=768`, while a later edit changes the actual head layout. Two runs then appear to share a configuration but instantiate different computation graphs. A configuration object should be an architecture contract: it lists every field that changes parameter shapes, state, cost, or comparison fairness, validates it immediately, and saves the completed defaults.

## Vocabulary fixes the output space

Let \(V\) be the tokenizer vocabulary size and \(d\) the hidden width. The input embedding is

$$
E\in\mathbb{R}^{V\times d}.
$$

At each position the residual stream is a vector in \(\mathbb{R}^{d}\). The MLP can temporarily use an intermediate width \(d_{\mathrm{ff}}\), but its output must return to \(d\) before the residual addition. The language-model head maps the final hidden state back to the vocabulary:

$$
\mathbf z_{b,t}=\mathbf h_{b,t}W_{\mathrm{out}},\qquad
W_{\mathrm{out}}\in\mathbb{R}^{d\times V}.
$$

The current implementation uses `bias=False`. With tied embeddings, \(W_{\mathrm{out}}=E^{\mathsf T}\); without tying, the model owns a second \(d\times V\) matrix. Either way, the last logits dimension is \(V\). This is why choosing a tokenizer also fixes the classification-head output dimension. Enlarging an MLP changes representation capacity, not the number of token classes.

For example, if \(V=259\) and \(d=16\), the tied embedding/head has \(259\times16=4{,}144\) scalar entries. A batch of hidden states with shape `[B, L, 16]` produces logits with shape `[B, L, 259]`. A target token is an integer in \([0,V)\), not a vector of arbitrary length.

## Four groups of fields

| Group | Examples | What changes |
|---|---|---|
| Model size | `hidden_size`, `num_layers`, `num_attention_heads` | parameter and activation shapes |
| Vocabulary | `vocab_size`, special-token IDs, tied embeddings | embedding and output projection |
| Adaptive state | `use_fast_mlp`, suffix layers, inner steps and learning rate | fast weights and update path |
| Training/system | batch, sequence length, optimizer, dtype, mesh | cost, memory, and recovery |

The control \(C_k\) and adaptive variant \(F_k\) must use the same static architecture. \(C_k\) disables test-time updates; \(F_k\) enables them. Changing width or vocabulary creates a different model and is a structural comparison, not a clean update ablation.

Sequence fields such as `max_position_embeddings`, `sliding_window`, and `ttt_chunk_size` are also architectural: they determine legal positions, cache capacity, and where fast-state updates occur. Keep them visible rather than burying them in a training script.

## How the code consumes the contract

`ttt/models/ttt_e2e/configuration_ttt_e2e.py` defines model, MLP-prime, MoE, loop, and dtype fields. `ttt/models/base_config.py` serializes configurations. `ttt/models/auto.py` and `ttt/models/registry.py` construct the requested class without a growing list of training-script branches.

~~~python
from ttt.models import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("configs/runs/tiny_gpu.json")
model = AutoModelForCausalLM.from_config(config, mode="train")
~~~

Loading should validate positive dimensions, head divisibility, vocabulary ranges, layer positions, special-token IDs, and compatible dtype/backend choices before initialization or XLA compilation. The expanded configuration belongs in `experiment.json`; otherwise a future default change can silently alter the architecture while leaving the original JSON untouched.

## Defaults and early errors

Some fields are derived. `num_key_value_heads=None` defaults to the query-head count, `head_dim=None` is derived from hidden width, and `ttt_suffix_length=None` is inferred from layer count and `use_fast_mlp`. Convenience is safe only when the resolved values are recorded.

Examples of errors that should happen during Python initialization:

- `hidden_size` is not divisible by `num_attention_heads`, so a head width cannot be defined;
- query heads are not divisible by key/value heads, so GQA replication is ambiguous;
- `ttt_chunk_size` exceeds the sliding window needed by a suffix update;
- `use_fast_mlp=false` is paired with a nonzero suffix, contradicting the control semantics;
- `eos_token_id` is outside `[0,V)`, so a generated label cannot index the head.

An early `ValueError` is cheaper and easier to diagnose than a compiler failure after several minutes.

## A tiny shape audit

Before launching training, write down the resolved values and trace one token:

~~~text
vocab_size = 259
hidden_size = 16
query_heads = 2, kv_heads = 1
layers = 2
ttt_suffix_length = 1
fast_intermediate_size = intermediate_size
sequence_length = 16
ttt_chunk_size = 4
~~~

Then draw `embedding -> attention -> MLP/MLP-prime -> residual -> lm_head` and annotate each shape. If one number has no source, the configuration is not yet an auditable contract. The next chapter follows the same shapes through a complete forward pass.
