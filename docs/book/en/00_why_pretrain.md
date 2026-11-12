# Prologue: Why train a base model from scratch

If tensor shapes, gradients, or softmax are new to you, read the [mathematics and deep-learning toolkit](00_prerequisites.md) first. This prologue then explains why we choose to train from scratch.

## Overview

If the goal is to ship an application, downloading a checkpoint is usually the shortest route. This book takes the longer route: start with data, build the tokenizer, construct the training windows and loss, save checkpoints, and then study TTT, linear attention, recurrent depth, MoE, and long-context state.

The central question is:

> Under controlled data, representation, training budget, and system conditions, how do sequence models store, update, and use context?

“From scratch” is a way to make the answer auditable. It does not claim that a tiny model is useful. It gives us evidence that the data path, objective, state transitions, and recovery protocol are correct before we spend a larger budget.

## A concrete scene: predicting “sat”

When a person sees “the cat”, “sat” is a plausible continuation. A model does not receive the words directly. A tokenizer maps them to integer coordinates. For hand calculation, suppose a short sentence becomes:

~~~text
text token:    the   cat   sat   <EOS>
token ID:        17    42    31      2
~~~

At the first position, the model receives 17 and assigns a score to every vocabulary ID. It does not return only “sat” or “not sat”; it returns a vector of \(V\) logits. Softmax turns the scores into probabilities, and cross-entropy evaluates the probability assigned to the observed target. This small scene contains the whole book: data chooses the sentence, tokenization chooses the integers, architecture transforms context, optimization changes state, systems determine whether the run fits, and evaluation limits the claim.

## Prerequisites and outcomes

You need only basic arrays, functions, and command-line usage. Linear algebra, probability, and gradients are introduced when they first appear. By the end of this chapter, you should be able to:

1. distinguish token IDs, embeddings, hidden states, logits, and probabilities;
2. construct \(X\), \(Y\), and \(M\) from a token stream;
3. derive the logits gradient from softmax and negative log likelihood;
4. name which of the six training-system parts changed;
5. produce a tiny token file and manifest without mistaking that artifact for evidence of language ability.

## What a base model learns

First consider prediction after observing information. A common word may become implausible in a particular context. Conditional probability \(p(b\mid a)\) assigns probability to \(b\) after \(a\) is known. Its definition, \(p(b\mid a)=p(a,b)/p(a)\) for \(p(a)>0\), gives \(p(a,b)=p(a)p(b\mid a)\). Applying the same identity to three events,

$$
p(a,b,c)=p(a,b)p(c\mid a,b)
=p(a)p(b\mid a)p(c\mid a,b).
$$

This does not assume independence; dependencies remain in the conditions. Replacing events with tokens at successive positions gives a language model's factorization. Let an already supplied BOS be \(s_0\), followed by the \(L\) tokens \(s_1,\ldots,s_L\) to predict, possibly including EOS. BOS itself is not one of these targets. Then

$$
p_\theta(s_{1:L}\mid s_0)
=
\prod_{t=1}^{L}p_\theta(s_t\mid s_{0:t-1}).
$$

Here \(s_{0:t-1}\) contains BOS and the preceding tokens, and \(\theta\) denotes model parameters. If three observed targets receive conditional probabilities \(0.5,0.8,0.25\), their sequence probability is \(0.5\times0.8\times0.25=0.1\). Many probabilities below one produce a very small product in a long sequence. The natural logarithm converts multiplication into addition:

$$
\log p_\theta(s_{1:L}\mid s_0)
=\log\prod_{t=1}^{L}p_\theta(s_t\mid s_{0:t-1})
=\sum_{t=1}^{L}\log p_\theta(s_t\mid s_{0:t-1}).
$$

Throughout the book, \(\log\) denotes the natural logarithm, with base \(e\approx2.718\). It is increasing, so maximizing probability also maximizes its logarithm. Negating it turns maximization into minimization; dividing by the number of targets \(L\) gives average negative log likelihood:

$$
\mathcal{L}(\theta)
=
-\frac{1}{L}
\sum_{t=1}^{L}
\log p_\theta(s_t\mid s_{0:t-1}).
$$

The task is local, but useful prediction may require syntax, code dependencies, discourse, or facts from far earlier in the context. Lower local prediction error can therefore encourage the model to represent those structures.

It does not guarantee reasoning, factuality, or instruction following. Duplicated training data can lower loss through memorization; contaminated validation can make the number look too good; and a different tokenizer can change token counts without changing the characters. Later chapters make these effects explicit.

## From logits to probability, loss, and gradient

Suppose the context is “the cat” and the correct next token is “sat”. For three candidates, let the logits be

$$
z=(2,1,0).
$$

Softmax is:

$$
p_j=\frac{\exp z_j}{\sum_u\exp z_u}.
$$

The probabilities are approximately \((0.665,0.245,0.090)\). If the correct token is the second candidate, its negative log likelihood is

$$
\ell=-\log(0.245)\approx1.407.
$$

If the correct probability rises to \(0.8\), the contribution becomes \(-\log(0.8)\approx0.223\). Thus “loss went down” first means that more probability moved to the observed token; it is not a claim about general intelligence.

Expanding the loss:

$$
\ell
=
-\log\frac{\exp z_y}{\sum_v\exp z_v}
=
-z_y+\log\sum_v\exp z_v.
$$

Differentiating component \(j\):

$$
\frac{\partial\ell}{\partial z_j}
=
p_j-\mathbf{1}[j=y].
$$

The indicator \(\mathbf{1}[j=y]\) equals 1 for the correct class and 0 otherwise. If logits were independent parameters, gradient descent would raise the correct score and lower the others. This is a local analysis: shared network parameters affect many predictions at once, so an actual update need not move every example in that direction. The baseline chapter connects this derivative to the chain rule for model parameters.

With padding or prompt masks, let \(W=\sum_{t=1}^{L}m_t\) be the total supervision weight. Only positive-weight positions contribute:

$$
\mathcal{L}
=
\frac{\sum_{t=1}^{L}
m_t[-\log p_\theta(y_t\mid x_{\le t})]}
{\begin{cases}
W,&W>0,\\
1,&W=0.
\end{cases}}
$$

The mask may contain fractional weights; a positive weight below \(1\) remains part of the denominator. An all-zero mask has no supervision; the safe denominator only prevents division by zero and does not turn that window into a useful training example. A wrong input-label shift trains a different task. Counting padding trains the model to predict padding. Those are data-pipeline errors, not architecture errors.

## Why train from scratch

Existing checkpoints are often the right choice for applications. From-scratch training is valuable when we need to control data, vocabulary, architecture, optimizer, token budget, and system cost, and ask whether a change in TTT or recurrent state improves an unseen-context metric under the same conditions.

The cost is real: licensing, cleaning, storage, stability, GPU scheduling, communication, and recovery become our responsibility. The book therefore starts with tiny runs and proves, in order, that the graph runs, the objective learns, and the comparison is interpretable. A tiny result is debugging evidence, not a large-model capability claim.

## The six coupled parts

| Part | Question | Typical failure |
|---|---|---|
| Data | What distribution did the model actually see? | near-duplicate validation and an unrealistically low loss |
| Representation | How did text become IDs, positions, and masks? | an off-by-one label shift or inconsistent EOS rule |
| Architecture | How does context change hidden state? | a cache mismatch or suffix reading the future |
| Optimization | Which state receives which gradient? | treating fast weights as ordinary optimizer parameters |
| Systems | Can the computation fit and recover? | missing compile, communication, or memory costs |
| Evaluation | What does each metric support? | treating one loss value as a universal capability score |

Changing vocabulary size \(V\) changes the embedding and output head. Changing sequence length \(L\) changes activation and attention cost. TTT changes state and gradient boundaries. MoE changes communication and load. The chapters are ordered so that these dependencies are visible rather than hidden behind a model name.

## Budget: parameters, tokens, throughput, and wall time

Let \(N_{\mathrm{param}}\) be trainable parameters, \(N_{\mathrm{tok}}\) consumed tokens, and \(r_{\mathrm{tok}}\) measured steady throughput. A common dense-transformer estimate is:

$$
F_{\mathrm{rough}}
\approx
6N_{\mathrm{param}}N_{\mathrm{tok}}.
$$

This does not fully cover long-context attention, special kernels, communication, or TTT inner updates. It cannot be turned directly into a price or completion time. A more useful engineering estimate uses measured throughput:

| Quantity | Example | Meaning |
|---|---:|---|
| training tokens | \(10^9\) | a budget, not file bytes |
| steady throughput | \(50{,}000\) token/s | measured at the target configuration |
| pure compute time | \(20{,}000\) s | about 5.6 hours |
| compile, evaluation, saves, retries | unknown | must be recorded separately |

Model and data budgets must be reported together. With \(d=768\), increasing vocabulary from 32K to 128K increases one embedding from about \(24.6\)M to \(98.3\)M parameters. The same hidden width therefore does not imply the same model or memory budget.

## First runnable checkpoint

On the designated remote CPU machine, create the tiny data artifacts:

~~~bash
bash scripts/prepare_sample.sh outputs/sample-tokens.npy
.venv/bin/python -m ttt.data \
  --input examples/sample.jsonl \
  --output outputs/sample-shards \
  --tokenizer byte \
  --sharded \
  --shard-tokens 256
~~~

You should see <code>outputs/sample-tokens.npy</code>, shard files, and <code>manifest.json</code>. Check <code>vocab_size</code>, shard lengths, and the input digest. If the files are absent, check the Python environment, input path, and output permissions first. This proves that the data object can be built; it does not prove that the model learns or that GPU training scales.

The book uses three completion levels: **runnable** means the entry point and dependencies work; **learnable** means loss, valid-token count, and updates behave plausibly; **interpretable** means controls, metrics, and evidence support a claim. Completing one level never silently implies the next.

## Mainline and Frontier Lab

The mainline follows dependencies: notation, experiment contract, data, dense baseline, state and compute, ablations, and validation. Frontier Lab follows fast-moving questions about TTT and linear attention, recurrent depth, and long context. Each frontier page must state the paper version, implementation status, minimal reproduction entry point, and missing evidence.

A frontier page may propose a hypothesis, but it cannot reuse a paper's numbers as this project's result. It enters the mainline only after a runnable baseline, controlled variables, metrics, and failure criteria exist.

## Summary and next question

This chapter established why we train, what the model predicts, how softmax/NLL/gradients connect, and how budget and evidence limit the claim. The next chapter turns these objects into an experiment contract: what must remain fixed, what must be saved, and how to prove that an interrupted run resumed the same process.
