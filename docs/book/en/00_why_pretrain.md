# Prologue: Why train a base model from scratch

## What the reader should be able to do

The reader should finish this book able to build an auditable training chain from scratch: identify the data a model saw, locate where its state was updated, measure where the budget was spent, and state exactly what an experiment does and does not prove.

The central question is:

> Under fixed data, representation, training budget, and system conditions, how do modern sequence models store, update, and use context?

“From scratch” is the method for answering that question. It keeps data, tokenization, optimization, state, and checkpoints visible before we compare attention, compressed state, TTT, recurrent depth, and MoE.

“Train a language model by putting text through a Transformer and minimizing cross-entropy.” That sentence is correct, but incomplete. It says nothing about where the text comes from, how the tokenizer changes the budget, what a resumed run means, or which evidence can support an architectural claim.

This book builds a complete experiment rather than a single model class. We begin with a controlled corpus, tokenizer, causal objective, and dense baseline. We then study TTT, linear attention, recurrent depth, MoE, and long-context state under the same data and reporting contract.

The progression is deliberately causal:

~~~text
motivation → experiment contract → data and representation → dense baseline
→ context memory → compute and capacity → evidence and boundaries
~~~

## The objective

For tokens $x_1,\ldots,x_L$, an autoregressive model minimizes

$$
\mathcal L(\theta)=-\frac{1}{L}\sum_{t=1}^{L}
\log p_\theta(x_t\mid x_{<t}).
$$

The next-token task is local, but useful prediction often requires syntax, code dependencies, discourse, or facts from the context. A low training loss is still not a guarantee of reasoning, factuality, or instruction following. It measures prediction on a data distribution, and duplicated or contaminated data can make that measurement misleading.

### A small example of next-token loss

Suppose the context is `the cat` and the next token is `sat`. The model does not output one answer; it outputs a score for every vocabulary item. Softmax turns those scores into probabilities:

$$
p_\theta(v\mid x_{<t})=
\frac{\exp z_{t,v}}
{\sum_{u=1}^{V}\exp z_{t,u}}.
$$

If the correct token receives probability $0.8$, its contribution is $-\log(0.8)\approx0.223$. If it receives $0.01$, the contribution is about $4.605$. The logarithm makes confident wrong predictions expensive and lets losses from different positions add; averaging then makes runs with different numbers of valid labels comparable. A mask $m_t$ removes padding or prompt labels from this average:

$$
\mathcal L=
\frac{\sum_{t=1}^{L}m_t[-\log p_\theta(x_t\mid x_{<t})]}
{\max(\sum_{t=1}^{L}m_t,1)}.
$$

This is why “the loss went down” is a precise statement about token prediction under a distribution, but not by itself a statement about broad intelligence. The data, tokenizer, split, and mask determine what the number means.

A base model learns sequence statistics. SFT and preference training add task and interaction behavior later. Keeping these stages separate makes the scope of this project clear.

## Why from scratch

Existing checkpoints are the right starting point for many applications. From-scratch training is valuable when we need to control data, vocabulary, architecture, optimizer, token budget, and system cost. Only then can a change in TTT or recurrent state be compared to a stable control.

The cost is real: data processing, licensing, storage, monitoring, GPU scheduling, and recovery all become part of the work. The book therefore starts with tiny runs, then scales only after the evidence is useful.

## A budget example

Three quantities should be kept separate. $N_{\mathrm{param}}$ is the number of trainable parameters; $N_{\mathrm{tok}}$ is the number of training tokens; and $r_{\mathrm{tok}}$ is the measured token throughput. A simple wall-time estimate is

$$
T_{\mathrm{wall}}\approx
\frac{N_{\mathrm{tok}}}{r_{\mathrm{tok}}},
$$

before adding compilation, evaluation, checkpointing, and failed jobs. For example, $5\times10^9$ tokens at $10^5$ tokens per second is about $5\times10^4$ seconds, or 13.9 hours of steady compute. If the tokenizer produces twice as many tokens for the same characters, the character coverage and cost comparison changes even when the model configuration is unchanged. If a vocabulary grows from 32K to 128K at hidden size 768, one embedding grows from about 24.6M to 98.3M parameters. A fair architecture comparison therefore declares both the model budget and the data budget.

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

For a beginner, the important habit is to ask which object changed. A new tokenizer changes the input coordinate system; a new attention rule changes how positions communicate; a new optimizer changes the update path; a new checkpoint format changes what “resume” means. Calling all of them a “model improvement” hides the cause of a result.

## The book loop

The stable route is:

~~~text
source data → cleaning → tokenizer → dense baseline
→ TTT and streaming state → MoE and loops
→ controlled ablations → GPU validation
~~~

Frontier Lab moves faster around TTT, linear attention, recurrent depth, and long context. A frontier claim enters the mainline only after it has a minimal reproduction and a compute-matched control.

The first chapter defines the experiment contract. The second fixes the environment, the third fixes reproducible runs, and the fourth data chapter defines the contract that every later model must consume.
