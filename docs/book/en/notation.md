# Notation and conventions

## Overview

This book discusses data, tensors, probability, optimization, and runtime state in one chain. If a symbol changes meaning between chapters, the equations become harder to audit than the code. We therefore begin with one short sentence, use it to build a causal-language-model example, and then fix the notation used by every chapter.

After this chapter, you should be able to turn text into token IDs, construct \(X\), \(Y\), and \(M\), and explain how logits become probabilities and loss. Mathematical positions start at 1; array indices in code start at 0.

## A running example: “the cat sits”

For hand calculation, suppose a tokenizer maps a short sentence and an end marker to the following **teaching IDs**:

~~~text
text token:    the   cat   sits   <EOS>
token ID:        17    42     31      2
math position:    1     2      3      4
code index:       0     1      2      3
~~~

Real tokenizers may assign different IDs. An ID is only an integer coordinate in the vocabulary. The model sees the stream \(s=(17,42,31,2)\), not the human meaning of the words. The conversion is:

$$
\text{mathematical position }t
\quad\longleftrightarrow\quad
\text{code index }t-1.
$$

Thus <code>tokens[2]</code> is mathematical position \(t=3\), not position 2. One-based sums are convenient in derivations; Python, NumPy, and JAX use zero-based arrays.

## From a token stream to \(X,Y,M\)

A causal language model predicts the next token from the context available so far. For the four-token stream, use the first three positions as input:

$$
X=(x_1,x_2,x_3)=(17,42,31),
\qquad
Y=(y_1,y_2,y_3)=(42,31,2).
$$

Position 1 predicts “cat” after “the”; position 2 predicts “sits” after “the cat”; position 3 predicts EOS after the full visible prefix. The code fields are:

~~~python
input_ids = [17, 42, 31]
labels = [42, 31, 2]
loss_mask = [1, 1, 1]
~~~

The code field <code>input_ids</code> is mathematical \(X\), <code>labels</code> is \(Y\), and <code>loss_mask</code> is \(M\). If two documents are packed together, whether the last token of one document may predict the first token of the next is a protocol choice. For example, \(M=(1,1,0)\) masks a cross-document target, while a continuous-corpus protocol may use \(M=(1,1,1)\). The mask defines which predictions enter the objective.

## Indices, shapes, and batches

| Object | Symbol | Convention |
|---|---|---|
| batch sample | \(b\) | \(b\in\{1,\ldots,B\}\) |
| token position | \(t\) | \(t\in\{1,\ldots,L\}\) |
| Transformer layer | \(\ell\) | \(\ell\in\{0,\ldots,n_\ell-1\}\) |
| chunk | \(c\) | a consecutive part of a token window |
| outer training step | \(k\) | one optimizer update of static parameters |
| expert | \(e\) | \(e\in\{1,\ldots,E\}\) |
| batch size | \(B\) | samples in one training step |
| sequence length | \(L\) | tokens in one window |
| vocabulary size | \(V\) | number of token IDs |

A batch of inputs is

$$
X\in\{0,\ldots,V-1\}^{B\times L}.
$$

For \(B=1,L=3\), <code>input_ids</code> has shape <code>[1, 3]</code>. Looking up each ID produces hidden vectors with shape <code>[1, 3, d]</code>; the vocabulary projection produces logits with shape <code>[1, 3, V]</code>. Unless a chapter says otherwise, axes are batch, sequence, features.

## Embeddings, hidden states, logits, and probabilities

These terms name different objects:

1. A **token ID** is an integer coordinate, such as 17.
2. An **embedding** is the \(d\)-dimensional vector looked up for that ID.
3. A **hidden state** is the \(d\)-dimensional residual-stream vector after model layers.
4. **Logits** are unnormalized scores for the \(V\) candidate tokens.
5. **Probabilities** are the normalized distribution obtained from logits.

Let the embedding matrix be \(E\in\mathbb{R}^{V\times d}\). Token 17 selects \(E_{17,:}\). If a model produces \(h\in\mathbb{R}^{d}\) and the output matrix is \(W_{\mathrm{out}}\in\mathbb{R}^{d\times V}\), then:

$$
z=hW_{\mathrm{out}}\in\mathbb{R}^{V}.
$$

The component \(z_v\) is a score, not a probability. Softmax gives:

$$
p_\theta(v\mid x_{1:t})
=
\frac{\exp z_v}{\sum_{u=0}^{V-1}\exp z_u}.
$$

The denominator sums every candidate, so the probabilities sum to one. Vocabulary size \(V\) therefore fixes both the final logits dimension and the number of classes in the distribution.

## Loss, gradient, and two clocks

For target \(y_t\), the per-token negative log likelihood is:

$$
\ell_t^{\mathrm{tok}}(\theta)
=
-\log p_\theta(y_t\mid x_{1:t}).
$$

With a mask \(m_t\in\{0,1\}\), the window loss is:

$$
\mathcal{L}
=
\frac{\sum_{t=1}^{L}m_t\ell_t^{\mathrm{tok}}}
{\max(\sum_{t=1}^{L}m_t,1)}.
$$

Why does the gradient become \(p-\text{one-hot}\)? For logits \(z\) and correct class \(y\):

$$
\ell
=
-\log\frac{\exp z_y}{\sum_v\exp z_v}
=
-z_y+\log\sum_v\exp z_v.
$$

Differentiating component \(j\), the first term contributes \(-\mathbf{1}[j=y]\), and the log-sum-exp term contributes \(p_j\):

$$
\frac{\partial\ell}{\partial z_j}
=
p_j-\mathbf{1}[j=y].
$$

The correct class receives a negative gradient, so gradient descent raises its score; incorrect classes receive positive gradients and are lowered. This local derivation reappears in the dense baseline and the TTT inner loop.

Static parameters are updated across training steps:

$$
\theta_{k+1}
=
\operatorname{Opt}_{\mathrm{outer}}
\left(\theta_k,\nabla_\theta\mathcal{L}_{\mathrm{outer}}\right).
$$

TTT fast weights can change within a sequence or chunk:

$$
\phi_{t+1}
=
\phi_t-\eta_{\mathrm{inner}}
\nabla_\phi\mathcal{L}_{\mathrm{inner}}.
$$

The indices are different clocks: \(k\) describes long-lived training state, while \(t\) describes short-lived adaptation.

## Datasets, distributions, and state owners

A dataset \(\mathcal{D}\) is a checkable object with files, manifests, and hashes. A distribution \(D\) is what the sampler actually shows to the model. Two files may contain the same number of documents but different numbers of tokens, so document-uniform sampling is not token-uniform sampling.

State also needs an owner and lifetime:

| State | Symbol | Owner | Lifetime |
|---|---|---|---|
| static parameters | \(\theta\) | model/checkpoint | many batches |
| optimizer state | \(O_k\) | trainer/checkpoint | training steps |
| sampler/RNG | \(R_k\) | data stream/checkpoint | resume protocol |
| KV cache | \(\mathbf c_t\) | request session | one request |
| TTT fast weights | \(\phi_t\) | chunk/session | protocol-defined |

“Model state” is too broad to audit. Loading parameters without optimizer and sampler state may succeed as a file operation, but it is not the same training process.

## Code checkpoint and evidence

Notation itself does not require a GPU. On the remote CPU machine, generate a tiny token file and shard:

~~~bash
bash scripts/prepare_sample.sh outputs/sample-tokens.npy
python3 -m ttt.data --input examples/sample.txt --output outputs/sample-shards --tokenizer byte --sharded --shard-tokens 256
~~~

Success should produce a token file, shard files, and <code>manifest.json</code>. Inspect <code>vocab_size</code>, shard token counts, and the input digest. This proves that the data object and fields can be produced; it does not prove that the model has learned language or that GPU throughput is adequate.

## Summary and next question

This chapter fixed positions, shapes, \(X/Y/M\), embeddings, hidden states, logits, probabilities, loss, gradients, and the state boundary between \(\theta\) and \(\phi\). The next chapter uses these symbols to define an experiment contract and ask whether a tiny run is genuinely comparable.

## Writing rules

Scalars use italic letters, vectors bold lowercase, matrices bold uppercase, and sets or losses calligraphic letters. Code fields <code>d_model</code>, <code>num_heads</code>, and <code>vocab_size</code> correspond to \(d\), \(n_h\), and \(V\); <code>input_ids</code>, <code>labels</code>, and <code>loss_mask</code> correspond to \(X\), \(Y\), and \(M\). Mathematical variables and probabilities use LaTeX; backticks are reserved for real code names.
