# Prologue: Why train a base model from scratch

[中文版](../00_why_pretrain.md)

“Train a language model by putting text through a Transformer and minimizing cross-entropy.” That sentence is correct, but incomplete. It says nothing about where the text comes from, how the tokenizer changes the budget, what a resumed run means, or which evidence can support an architectural claim.

This book builds a complete experiment rather than a single model class. We begin with a controlled corpus, tokenizer, causal objective, and dense baseline. We then study TTT, linear attention, recurrent depth, MoE, and long-context state under the same data and reporting contract.

## The objective

For tokens `x_1,\ldots,x_T`, an autoregressive model minimizes

$$
\mathcal L(\theta)=-\frac{1}{T}\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t}).
$$

The next-token task is local, but useful prediction often requires syntax, code dependencies, discourse, or facts from the context. A low training loss is still not a guarantee of reasoning, factuality, or instruction following. It measures prediction on a data distribution, and duplicated or contaminated data can make that measurement misleading.

A base model learns sequence statistics. SFT and preference training add task and interaction behavior later. Keeping these stages separate makes the scope of this project clear.

## Why from scratch

Existing checkpoints are the right starting point for many applications. From-scratch training is valuable when we need to control data, vocabulary, architecture, optimizer, token budget, and system cost. Only then can a change in TTT or recurrent state be compared to a stable control.

The cost is real: data processing, licensing, storage, monitoring, GPU scheduling, and recovery all become part of the work. The book therefore starts with tiny runs, then scales only after the evidence is useful.

## Six coupled parts

| Part | Question |
|---|---|
| Data | What distribution does the model see? |
| Representation | How does text become token IDs and masks? |
| Architecture | How does context change the hidden state? |
| Optimization | Which state receives which gradient? |
| Systems | Can the computation fit and recover? |
| Evaluation | What conclusion does each metric support? |

A vocabulary change affects embedding and lm_head. A context change affects memory and attention cost. TTT changes state and gradient boundaries. MoE changes communication. These dependencies explain the order of the chapters.

## The book loop

The stable route is:

~~~text
source data → cleaning → tokenizer → dense baseline
→ TTT and streaming state → MoE and loops
→ controlled ablations → GPU validation
~~~

Frontier Lab moves faster around TTT, linear attention, recurrent depth, and long context. A frontier claim enters the mainline only after it has a minimal reproduction and a compute-matched control.

The first chapter fixes the environment. The next data chapter defines the contract that every later model must consume.
