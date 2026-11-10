# 14. Configuration: one contract for every tensor shape

## Overview: give every number a source

In <code>tiny-contract-001</code>, the vocabulary has 259 entries, the residual width is 16, and attention has two query heads. A reader should be able to follow those numbers through the embedding, each head, the residual stream, and the logits. If a number is only discoverable deep inside the code, the configuration is not yet an experiment contract.

After this chapter, you should be able to load a configuration, list its resolved defaults, hand-calculate a batch shape, and explain why the tokenizer vocabulary fixes the last dimension of the language-model head. Whether the model learns is the next chapter's question.

## Prerequisites and tiny target

You only need the idea that a matrix maps one collection of numbers to another. The running input is <code>input_ids</code> with shape <code>[B,L]=[2,4]</code>. The completion standard is to write down every shape after embedding, attention/MLP, and <code>lm_head</code>.

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

The current implementation has no output-projection bias. With tied embeddings, \(W_{\mathrm{out}}=E^{\mathsf T}\); without tying, the model owns a second \(d\times V\) matrix. Either way, the last logits dimension is \(V\). This is why choosing a tokenizer also fixes the classification-head output dimension. Enlarging an MLP changes representation capacity, not the number of token classes.

For example, if \(V=259\) and \(d=16\), the tied embedding/head has \(259\times16=4{,}144\) scalar entries. A batch of hidden states with shape `[B, L, 16]` produces logits with shape `[B, L, 259]`. A target token is an integer in \([0,V)\), not a vector of arbitrary length.

## Four groups of fields

| Group | Examples | What changes |
|---|---|---|
| Model size | `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads` | parameter and activation shapes |
| Vocabulary | `vocab_size`, special-token IDs, tied embeddings | embedding and output projection |
| Adaptive state | <code>use_fast_mlp</code>, <code>ttt_suffix_length</code>, <code>inner.enabled</code>, <code>inner.learning_rate</code> | fast weights and update path |
| Training/system | batch, sequence length, optimizer, dtype, mesh | cost, memory, and recovery |

The control C and adaptive variant F use the same static architecture. C disables inner updates; F enables them. Changing width or vocabulary creates a different model and is a structural comparison rather than an isolated update ablation.

Sequence fields such as `max_position_embeddings`, `sliding_window`, and `ttt_chunk_size` are also architectural: they determine legal positions, cache capacity, and where fast-state updates occur. Keep them visible rather than burying them in a training script.

## How the code consumes the contract

`ttt/models/ttt_e2e/configuration_ttt_e2e.py` defines model, MLP-prime, MoE, loop, and dtype fields. `ttt/models/base_config.py` serializes configurations. `ttt/models/auto.py` and `ttt/models/registry.py` construct the requested class without a growing list of training-script branches.

~~~python
from ttt.models import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("configs/book/tiny_control.json")
model = AutoModelForCausalLM.from_config(config, mode="train")
~~~

This example demonstrates architecture registration only. <code>AutoConfig</code> returns the model configuration, not the experiment's <code>inner</code> or <code>training</code> settings; <code>mode="train"</code> selects a model class and does not carry the control's disabled-update setting. Load the full experiment with <code>load_experiment</code> and pass its inner configuration into the training or sequence-loss path to control adaptation.

Loading should validate positive dimensions, head divisibility, vocabulary ranges, layer positions, and compatible dtype/backend choices before initialization or XLA compilation. The expanded experiment configuration belongs in <code>experiment.json</code>; otherwise a future default change can silently alter the architecture while leaving the original JSON untouched.

## Defaults and early errors

Some fields are derived. `num_key_value_heads=None` defaults to the query-head count, and `head_dim=None` divides hidden width by query-head count. When fast MLP or <code>adaptation_target="router"</code> is enabled, an unspecified suffix covers the final quarter of layers, with a minimum of one; otherwise its length is zero. Convenience is safe only when resolved values are recorded.

Examples of errors that should happen during Python initialization:

- `head_dim` is unspecified and `hidden_size` is not divisible by `num_attention_heads`, so the default head width cannot be derived;
- query heads are not divisible by key/value heads, so GQA replication is ambiguous;
- fast MLP or router adaptation is enabled, the layer types include sliding attention, and `ttt_chunk_size` exceeds `sliding_window`;
- under the default <code>adaptation_target="mlp"</code>, `use_fast_mlp=false` is paired with a nonzero suffix; router adaptation is a different case and can retain a suffix without a fast MLP;
- `eos_token_id` is outside `[0,V)`, so a generated label cannot index the head.

An early `ValueError` is cheaper and easier to diagnose than a compiler failure after several minutes.

## A tiny shape audit

Before launching training, write down the resolved values and trace one token:

~~~text
vocab_size = 259
hidden_size = 16
num_attention_heads = 2
num_key_value_heads = 1
num_hidden_layers = 2
ttt_suffix_length = 1
intermediate_size = 32
fast_intermediate_size = 32
inner.enabled = false    # tiny_control; the corresponding tiny.json value is true
sequence_length = 16
ttt_chunk_size = 4
~~~

This is an audit extract, not loadable JSON. Model fields belong to <code>model</code>, sequence length belongs to <code>training</code>, and the update switch belongs to <code>inner</code>. The next step connects these numbers to arrays.

### Start with a two-feature lookup and projection

A vector is an ordered collection of numbers; a matrix arranges numbers into rows and columns. To calculate every entry, temporarily use four tokens with two embedding features each:

$$
\mathbf E=\begin{pmatrix}1&0\\0&1\\2&1\\-1&1\end{pmatrix}.
$$

Input IDs 0 and 2 select rows \((1,0)\) and \((2,1)\), forming an array of shape \([2,2]\). The first dimension counts positions and the second counts features per position. With tied output weights, the hidden vector \(\mathbf h=(1,0)\) takes a dot product with each embedding row:

$$
\mathbf h\mathbf E^{\mathsf T}
=(1\times1+0\times0,\ 1\times0+0\times1,\ 1\times2+0\times1,\ 1\times(-1)+0\times1)
=(1,0,2,-1).
$$

Two features produce four candidate-token scores. Every position uses the same projection; adding the batch axis gives \([B,L,d]\to[B,L,V]\). This example shows only lookup and projection, omitting the intervening model layers.

### Walk through the tiny dimensions

Use the tiny model widths \(V=259\), \(d=16\), and \(d_{\mathrm{ff}}=32\), with a teaching batch \(B=2\) and length \(L=4\). These illustrative dimensions do not change the actual training configuration \(B=1,L=16\). Integer inputs have shape \([2,4]\); embedding lookup produces \([2,4,16]\). The ordinary SwiGLU branch projects features to width 32 and returns them to width 16 before the residual addition. The tied head multiplies by \(E^{\mathsf T}\in\mathbb R^{16\times259}\), giving logits of shape \([2,4,259]\). A target is one integer, such as \(y=42\), selecting one of those 259 candidate scores.

### Head dimensions and the MLP intermediate width

The batched layout above explains how samples stack. The current model's <code>forward</code> handles one sequence: input \([L]\), hidden states \([L,d]\), and logits \([L,V]\). Training applies that function across samples with <code>jax.vmap</code> to obtain \([B,L,V]\). Do not pass the teaching table's \([B,L]\) array directly to the single-sequence interface.

For tiny's \(d=16,n_h=2,n_{\mathrm{kv}}=1\), the default head width is \(d_h=8\). Query projection produces \(2\times8=16\) values per position, reshaped to \([L,2,8]\). Key and Value each produce \(1\times8=8\) values, shaped \([L,1,8]\). Two query heads share one KV head: this is grouped-query attention (GQA). It reduces KV projection and storage, rather than halving the residual output width. The attention chapter develops the operation itself.

SwiGLU's intermediate width of 32 also does not make its final output 32-dimensional. Each row \(h\in\mathbb R^{16}\) undergoes two projections, elementwise gating, and an output projection:

$$
g=hW_g,\quad u=hW_u,\quad
\operatorname{SwiGLU}(h)=(\operatorname{SiLU}(g)\odot u)W_d,
$$

Here \(W_g,W_u\in\mathbb R^{16\times32}\), \(W_d\in\mathbb R^{32\times16}\), \(\odot\) denotes elementwise multiplication, and \(\operatorname{SiLU}(a)=a/(1+e^{-a})\) is applied elementwise. Both intermediate vectors have width 32; the result returns to width 16 for residual addition. This establishes the calculation and shapes. Whether this gating choice is useful is a model-design experiment.

## Minimal configuration audit

On the remote node, parse the resolved configuration before launching training:

~~~bash
.venv/bin/python -c "from ttt.config import load_experiment; import json; print(json.dumps(load_experiment('configs/book/tiny_control.json').to_dict(), ensure_ascii=False, indent=2))"
~~~

The output should contain <code>model</code>, <code>inner</code>, <code>training</code>, and <code>tokenizer</code>, including resolved derived fields. If head divisibility, special-token ranges, or sequence-length limits fail, repair the configuration before waiting for XLA compilation. This command proves that the contract parses; it does not prove that the forward pass or training is correct.

## Summary

- The configuration contract fixes parameter shapes, state boundaries, cost, and comparison fairness.
- \(V\) determines the vocabulary axis of the embedding and logits; \(d\) determines residual width.
- Resolved defaults must be written into the experiment record.
- A tiny shape walk connects abstract fields to real tensors.
- The next chapter computes logits, cross-entropy, and gradients from the same configuration.
