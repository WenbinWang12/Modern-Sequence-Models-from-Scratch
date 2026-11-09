# Notation and conventions

This book discusses data, models, optimization, and runtime state together. A symbol that changes meaning from chapter to chapter makes the equations harder to audit than the implementation. The main text, equations, and experiment reports therefore use the notation below.

Mathematical positions are one-based; array indices in code are zero-based. Thus mathematical $x_t$ corresponds to <code>tokens[t - 1]</code> in code.

## 1. Indices and shapes

| Object | Symbol | Convention |
|---|---|---|
| batch sample | $b$ | $b\in\{1,\ldots,B\}$ |
| token position | $t$ | $t\in\{1,\ldots,L\}$ |
| Transformer layer | $\ell$ | $\ell\in\{0,\ldots,n_\ell-1\}$ |
| chunk | $c$ | a window may contain several consecutive chunks |
| outer training step | $k$ | one optimizer update of the static parameters |
| expert | $e$ | $e\in\{1,\ldots,E\}$ |
| batch size | $B$ | samples in one training step |
| sequence length | $L$ | tokens in one window |
| vocabulary size | $V$ | number of token IDs |

A batch of token IDs is $X\in\{0,\ldots,V-1\}^{B\times L}$. Shapes such as <code>[B, L]</code> and <code>[B, L, d]</code> are ordered as batch, sequence, features unless stated otherwise.

## 2. Inputs, targets, and masks

- $x_{b,t}$: input token ID at position $t$ in sample $b$;
- $y_{b,t}$: next-token target;
- $m_{b,t}\in\{0,1\}$: whether that target contributes to the loss;
- $X,Y,M$: the corresponding batch tensors.

Standard causal language modeling uses $y_{b,t}=x_{b,t+1}$. Whether the final input position has a target depends on the window and padding protocol.

## 3. Model and representations

| Object | Symbol | Shape or meaning |
|---|---|---|
| hidden/residual width | $d$ | feature dimension of the residual stream |
| FFN width | $d_{\mathrm{ff}}$ | intermediate MLP or SwiGLU width |
| number of layers | $n_\ell$ | Transformer blocks |
| attention heads | $n_h$ | number of query heads |
| head width | $d_h$ | usually $d_h=d/n_h$ |
| embedding | $E$ | $E\in\mathbb{R}^{V\times d}$ |
| hidden state | $\mathbf{h}^{(\ell)}_{b,t}$ | $d$-dimensional state at layer $\ell$ |
| hidden tensor | $\mathbf{H}^{(\ell)}$ | shape $[B,L,d]$ |
| logits | $\mathbf{z}_{b,t}$ | $\mathbf{z}_{b,t}\in\mathbb{R}^{V}$ |
| probability distribution | $p_\theta$ | next-token distribution |

The forward function is written as $f_\theta$. With TTT or another test-time adaptation mechanism, it becomes $f_{\theta,\phi_t}$: $\theta$ are static parameters learned across batches, while $\phi_t$ are fast weights that may change within a sequence.

## 4. Losses and optimization

- $\ell^{\mathrm{tok}}_{b,t}$: per-token negative log likelihood;
- $\mathcal{L}_{\mathrm{batch}}$: batch loss weighted by valid-token count;
- $\mathcal{L}_{\mathrm{inner}}$: loss used to update fast weights;
- $\mathcal{L}_{\mathrm{outer}}$: loss used to update static parameters;
- $\mathcal{L}_{\mathrm{val}}$: aggregate validation loss;
- $\eta_{\mathrm{outer}}$ and $\eta_{\mathrm{inner}}$: outer and inner learning rates.

The static-parameter update is:

$$
\theta_{k+1}
=
\operatorname{Opt}_{\mathrm{outer}}
\left(\theta_k,\nabla_\theta\mathcal{L}_{\mathrm{outer}}\right).
$$

The fast-weight update is:

$$
\phi_{t+1}
=
\phi_t-\eta_{\mathrm{inner}}
\nabla_\phi\mathcal{L}_{\mathrm{inner}}.
$$

The indices represent different time scales: $k$ advances across training steps, while $t$ advances within a sequence or chunk.

## 5. Data, budgets, and experiments

| Object | Symbol | Meaning |
|---|---|---|
| dataset or manifest | $\mathcal{D}_{\mathrm{train}}$, $\mathcal{D}_{\mathrm{val}}$, $\mathcal{D}_{\mathrm{test}}$ | concrete readable data objects |
| training data distribution | $D$ | distribution actually induced by the sampler |
| consumed tokens | $N_{\mathrm{tok}}$ | token count in the training budget |
| parameter count | $N_{\mathrm{param}}$ | total trainable parameters |
| compute | $F$ | FLOPs or another declared estimate |
| wall-clock time | $T_{\mathrm{wall}}$ | elapsed execution time |

A dataset $\mathcal{D}$ and a data distribution $D$ are different: the former is an input object with hashes, manifests, and splits; the latter is the probability distribution produced by sampling.

## 6. State boundaries

- $\mathbf{c}_t$: attention KV cache;
- $\mathbf{r}_t$: recurrent or linear-attention state;
- $\phi_t$: TTT fast weights;
- $\mathbf{S}_k$: recoverable training state associated with step $k$, including optimizer, sampler, and random state.

Parameters, optimizer state, data-sampler state, and request/session state should not all be called “model state.” Each chapter identifies the owner and lifetime of the state it introduces.

## 7. Writing rules

1. Scalars use italic letters, vectors use bold lowercase letters, matrices use bold uppercase letters, and sets or losses use calligraphic letters.
2. Code fields <code>d_model</code>, <code>num_heads</code>, and <code>vocab_size</code> correspond to $d$, $n_h$, and $V$.
3. <code>input_ids</code>, <code>labels</code>, and <code>loss_mask</code> correspond to $X$, $Y$, and $M$.
4. Code and logs may spell the symbols as <code>theta</code> and <code>phi</code>; equations use $\theta$ and $\phi$.
5. $L$ means sequence length, $\mathcal{L}$ means a loss, and $B$ means batch size.
6. Scaling uses $N_{\mathrm{param}}$ and $N_{\mathrm{tok}}$ rather than ambiguous $N$ and $D$.
7. Equations use one-based positions; code uses zero-based array indices, and the conversion must be stated when it matters.

Mathematical variables, probabilities, sets, and tensor shapes must use rendered LaTeX, such as \(x\), \(\theta\), \(P_{\mathrm{train}}\), and \(X\in\mathbb{R}^{B\times L\times d}\). Backticks are reserved for actual code names; mathematical variables must not use code styling.
