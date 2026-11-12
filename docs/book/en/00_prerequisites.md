# A mathematics and deep-learning toolkit

This chapter collects the smallest mathematical tools used throughout the book. The goal is to read tensor shapes, calculate one probability and gradient by hand, and explain one parameter update. It is a bridge for beginners, not a replacement for a full linear-algebra or calculus course.

Mathematical positions in the book start at 1; code arrays start at 0. Let \(B\) be the batch size, \(L\) the sequence length, \(d\) the hidden size, and \(V\) the vocabulary size.

## From scalars to tensors

A scalar is one number, such as a learning rate \(\eta=0.1\). A vector is an ordered list:

$$
\mathbf z=(2,1,0)\in\mathbb R^3.
$$

A matrix has two axes. An output projection can have shape \(W_{\mathrm{out}}\in\mathbb R^{d\times V}\). If one hidden row is \(h\in\mathbb R^{1\times d}\), then \(hW_{\mathrm{out}}\in\mathbb R^{1\times V}\) is the logit vector for one position.

| Object | Shape | Meaning |
|---|---|---|
| token IDs | \(B\times L\) | one integer coordinate per position |
| embedding | \(B\times L\times d\) | one vector per token |
| hidden state | \(B\times L\times d\) | representation passed between layers |
| logits | \(B\times L\times V\) | vocabulary scores at every position |
| loss mask | \(B\times(L-1)\) | targets included in the loss |

Changing \(B\) mainly changes parallel samples. Changing \(L\) changes context and attention cost. Changing \(d\) changes representation width. Changing \(V\) changes the embedding and output-head class dimension.

## How shapes move through the model

The embedding matrix is

$$
E\in\mathbb R^{V\times d}.
$$

Input \(X\in\{0,\ldots,V-1\}^{B\times L}\) uses each integer as a row coordinate:

$$
E[X]\in\mathbb R^{B\times L\times d}.
$$

An ID is an index, not a meaningful real number. Applying \(W\in\mathbb R^{d\times d'}\) at every position changes only the last axis:

$$
H\in\mathbb R^{B\times L\times d}
\quad\longrightarrow\quad
HW\in\mathbb R^{B\times L\times d'}.
$$

With \(W_{\mathrm{out}}\in\mathbb R^{d\times V}\),

$$
Z=HW_{\mathrm{out}}\in\mathbb R^{B\times L\times V}.
$$

This is why changing \(V\) changes the output head even when \(d\) and the layer count stay fixed.

### Attention shapes

Attention projects hidden states into queries, keys, and values:

$$
Q=HW_Q,\qquad K=HW_K,\qquad V_h=HW_V.
$$

For one head, the projection matrices can have shape \(W_Q,W_K,W_V\in\mathbb R^{d\times d_h}\). Multiple heads are concatenated along the last axis and then projected back to width \(d\). For head width \(d_h\), position scores have shape

$$
S=\frac{QK^{\mathsf T}}{\sqrt{d_h}}\in\mathbb R^{B\times L\times L}.
$$

The \(L\times L\) table is a table of position-to-position scores, not vocabulary logits. A causal mask blocks future positions; softmax normalizes the permitted positions before they weight the values.

## Derivatives, gradients, and the chain rule

A derivative is a local slope. If \(f(a)=a^2\), then

$$
\frac{df}{da}=2a.
$$

The partial derivatives of a vector or matrix form a gradient. The simplest gradient-descent update is

$$
\theta'=\theta-\eta\nabla_\theta\mathcal L.
$$

\(\theta\) is the parameter state that persists across training steps, and \(\eta\) is the learning rate. If \(a=f(b)\) and \(b=g(c)\), the chain rule gives

$$
\frac{da}{dc}=\frac{da}{db}\frac{db}{dc}.
$$

Backpropagation repeatedly applies this rule along a computation graph. TTT also updates fast weights \(\phi\); the outer loss may differentiate through that update, creating a second-order path. A central difference,

$$
\frac{f(a+\varepsilon)-f(a-\varepsilon)}{2\varepsilon}
\approx\frac{df}{da},
$$

can check a small example numerically, but it is not a training implementation.

## From logits to softmax and cross-entropy

For logits \(z_{t,v}\) at position \(t\), softmax is

$$
p_\theta(v\mid x_{\le t})
=\frac{\exp z_{t,v}}{\sum_{u=0}^{V-1}\exp z_{t,u}}.
$$

If \(y_t\) is the observed next token, its negative log-likelihood is

$$
\ell_t=-\log p_\theta(y_t\mid x_{\le t}).
$$

For \(z=(2,1,0)\) with the second class correct, the probability is about \(0.245\) and the loss about \(1.407\). Differentiating the logits gives

$$
\frac{\partial\ell_t}{\partial z_{t,v}}
=p_{t,v}-\mathbf 1[v=y_t].
$$

The correct class has a negative local gradient; other classes have positive local gradients. Shared network parameters affect many positions, so a real update need not increase every example's correct logit monotonically.

For a next-token loss mask \(m_t\in[0,1]\), let \(W=\sum_{t=1}^{L-1}m_t\). A positive-weight window uses

$$
\mathcal L=\frac{\sum_{t=1}^{L-1}m_t\ell_t}{W}\qquad(W>0).
$$

When \(W=0\), a safe denominator only prevents division by zero; it does not create supervision. A positive fractional weight below \(1\) remains part of the denominator.

## Optimizers and two time scales

The outer loop can be written as

$$
\theta_{k+1}=\theta_k-\eta_{\mathrm{outer}}\nabla_\theta\mathcal L_k.
$$

AdamW adds first- and second-moment statistics and decoupled weight decay around this basic direction. A TTT inner loop may be written as

$$
\phi_{c+1}
=\phi_c-\eta_{\mathrm{inner}}
\nabla_\phi\mathcal L_{\mathrm{inner},c}.
$$

\(k\) counts long-lived training steps; \(c\) counts chunks inside a sequence. Label availability, tail-chunk commits, and whether the outer loop keeps the second-order path are protocol choices. The two learning rates are different quantities.

## Numerical stability and dtype

Softmax commonly subtracts the largest logit:

$$
\operatorname{softmax}(z)_v
=\frac{\exp(z_v-\max_u z_u)}
{\sum_j\exp(z_j-\max_u z_u)}.
$$

The probability is unchanged, but exponentials are less likely to overflow. Low-precision dtypes save memory while reducing numerical range. Record the dtypes used for parameters, gradients, activations, fast weights, and reductions. When NaN or Inf appears, preserve the preceding loss, gradients, learning rate, dtype, and batch information.

## A reading order for model code

1. Check the ID range, \(B\), and \(L\) in \(X\).
2. Check that embedding produces \(B\times L\times d\).
3. Check Q/K/V and the causal mask for future access.
4. Check the input and output widths of residual paths, normalization, and the MLP.
5. Check that logits have shape \(B\times L\times V\).
6. Check the shift from \(X\) to \(Y\) and the supervised positions in \(M\).
7. Check which state is updated: loss, gradients, optimizer state, or parameters.
8. If TTT is enabled, trace the chunk after which \(\phi_c\) changes.

Read the ordinary dense path first. Then treat TTT as a change to one state and gradient protocol. This makes it easier to tell whether a new module changes representations, context, state, or the optimization objective.

## Summary

This chapter established the smallest mathematical bridge for the book: shapes describe data flow, matrix multiplication transforms representations, softmax and cross-entropy define the prediction target, gradients and the chain rule explain updates, masks define supervision, and dtype defines numerical boundaries.

The next chapter turns these tools into an experiment contract: state what is being compared, which variables must stay fixed, and what evidence can support the claim.
