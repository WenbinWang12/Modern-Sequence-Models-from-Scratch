## 2 Method

### 2.1 Notation

A token sequence is \(x_1,\ldots,x_{L+1}\). At position \(t\) the model reads \(x_{1:t}\) and predicts \(x_{t+1}\); \(L\) is the number of predicted positions in one training window. The model has \(n_\ell\) residual blocks, hidden width \(d\), MLP inner width \(d_{\mathrm{ff}}\) and vocabulary size \(V\). Each sliding-window attention layer sees the most recent \(w\) positions.

Parameters are split in two. **Slow weights** \(\theta\) are fixed once training ends and shared by every sequence. **Fast weights** \(\phi\) belong to the sequence currently being read and change as it is read. The sequence is cut into update chunks of \(C\) tokens; chunk \(c\) covers positions \(\mathcal T_c=\{(c-1)C+1,\ldots,cC\}\). \(\phi_{c-1}\) denotes the fast weights used to predict chunk \(c\), and \(\phi_0\) is their initial value.

### 2.2 Architecture

The backbone is a pre-norm, decoder-only Transformer. Each block does two things in order:

$$
u=h+\operatorname{Attn}\bigl(\operatorname{RMSNorm}(h)\bigr),\qquad
h'=u+\operatorname{MLP}\bigl(\operatorname{RMSNorm}(u)\bigr).
$$

Attention uses rotary position embeddings (RoPE, base \(5\times10^5\)) and QK normalisation: queries and keys are RMS-normalised per head before the dot product, and scores are divided by \(\sqrt{d_h}\). The MLP is a bias-free SwiGLU, \(\operatorname{MLP}(v)=W_{\mathrm{down}}\bigl(\operatorname{SiLU}(W_{\mathrm{gate}}v)\odot W_{\mathrm{up}}v\bigr)\). Input embeddings and the output projection share one matrix.

TTT-E2E changes only the last \(n_s\) blocks (the **suffix**; the other \(n_\ell-n_s\) blocks are the **prefix**). Each suffix block gains a second MLP of the same shape, MLP′, between attention and the original MLP. Its three matrices are the fast weights:

$$
u=h+\operatorname{Attn}(\operatorname{RMSNorm}(h)),\qquad
v=u+\operatorname{MLP}'_{\phi}(\operatorname{RMSNorm}(u)),\qquad
h'=v+\operatorname{MLP}(\operatorname{RMSNorm}(v)).
$$

The original MLP stays a slow weight, so knowledge from pre-training keeps a home that test-time updates cannot overwrite. This choice, and updating only the last quarter of the blocks, follow the original paper[^ttte2e]. Embeddings, normalisation and attention are never updated at test time.

<figure markdown="span" class="report-figure">
  ![Forward pass and the two loops of TTT-E2E](assets/report/fig_method.light.svg#only-light){ width="100%" }
  ![Forward pass and the two loops of TTT-E2E](assets/report/fig_method.dark.svg#only-dark){ width="100%" }
  <figcaption>Figure 1. Forward pass of TTT-E2E. The prefix runs once over the whole sequence. The suffix runs chunk by chunk: each chunk is predicted with the current fast weights, and its loss then moves the fast weights one step. The outer loop sums the chunk losses and differentiates through the whole chain of updates.</figcaption>
</figure>

### 2.3 Inner loop: predict first, then update

The prefix does not depend on the fast weights, so it is computed once for the whole sequence, giving features \(g_t=f^{\mathrm{pre}}_\theta(x_{1:t})\). The suffix is computed chunk by chunk. After chunk \(c\) has been read, its mean loss is

$$
\ell_c(\theta,\phi_{c-1})=\frac1C\sum_{t\in\mathcal T_c}-\log p\bigl(x_{t+1}\mid x_{1:t};\theta,\phi_{c-1}\bigr).
$$

Chunk \(c\) is predicted with \(\phi_{c-1}\), which has only seen the targets of the first \(c-1\) chunks. Only once every target in chunk \(c\) has been observed do the fast weights take one gradient step:

$$
g_c=\nabla_{\phi}\,\ell_c(\theta,\phi)\Big|_{\phi=\phi_{c-1}},\qquad
\hat g_c=\min\!\Bigl(1,\frac{\tau}{\lVert g_c\rVert_2}\Bigr)g_c,\qquad
\phi_c=\phi_{c-1}-\eta\,\hat g_c .
$$

\(\eta\) is the inner learning rate and \(\tau\) caps the gradient norm taken over all fast weights together. The updated \(\phi_c\) is used only from chunk \(c+1\) onwards, so no token is ever scored by weights that have seen it, and evaluation has no label leakage. A trailing partial chunk is scored but triggers no update.

Suffix attention keeps a KV cache: chunk \(c\) sees the keys and values of earlier chunks inside the window. They were computed with the fast weights of their time and are not recomputed after an update.

### 2.4 Outer loop: learning to be updated

The training objective is the mean loss *after* reading and updating in this way:

$$
\mathcal L(\theta)=\frac{1}{n_c}\sum_{c=1}^{n_c}\ell_c\bigl(\theta,\phi_{c-1}(\theta)\bigr),\qquad \phi_0\subset\theta .
$$

\(\phi_0\) is part of the slow weights: the initial value of the fast weights is learned. Because \(\phi_{c-1}\) is a function of \(\theta\), the gradient of \(\mathcal L\) flows along the update chain as well as through ordinary back-propagation:

$$
\frac{d\phi_c}{d\theta}=\frac{d\phi_{c-1}}{d\theta}-\eta\,\frac{d\hat g_c}{d\theta},
$$

where \(d\hat g_c/d\theta\) contains second derivatives of \(\ell_c\). In code this is one more differentiation of the whole forward function, inner loop included (gradients of gradients). The slow weights therefore learn two things: how to predict the next token, and how to make the suffix predict better once it has been updated.

The **first-order** variant in the ablations treats \(\hat g_c\) as a constant, \(d\hat g_c/d\theta=0\). The forward pass is unchanged and training is cheaper, but the outer loop no longer knows how the slow weights shape the update.

### 2.5 Inference

Inference runs the same inner loop without the outer loop. A session's state has three parts: the current fast weights \(\phi\) (fixed size, independent of how much has been read), the KV cache of the last \(w\) positions in each layer, and the features and targets of the chunk still being filled. During generation one update runs each time \(C\) tokens have accumulated and the next token is known. No session modifies the released slow weights, and sessions do not share fast weights.

### 2.6 Implementation notes

The implementation is in JAX. The following points bear directly on how far the results can be trusted.

- **Outer-loop gradients.** The sequence loss runs the chunks in a `lax.scan`, calling `jax.value_and_grad` at each step to obtain \(g_c\); the whole function is then differentiated again. The chunk scan is two-level with checkpointing, so the backward pass keeps about \(\sqrt{n_c}\) snapshots of the fast weights and recomputes the rest.
- **Sliding-window attention.** During training the prefix applies windowed attention to the whole sequence. We cut the sequence into blocks of \(w\) positions and let each block attend densely to itself and the previous block, all blocks in parallel. In float64 it differs from the per-position masked reference by at most \(7\times10^{-7}\) (forward values and gradients with respect to queries and keys). It is a different algorithm for the same function, not an approximation.
- **Cross-entropy.** The vocabulary projection and log-softmax are computed in token blocks; the values equal the one-shot computation.
- **Precision.** Parameters are float32; matrix products use bfloat16; normalisation, softmax and the loss use float32.
- **Correctness tests.** The repository's tests cover causality (changing future tokens leaves earlier outputs unchanged), agreement of the outer-loop gradient with finite differences, agreement of streaming inference with the full forward pass, checkpoint restore, and the property that the first-order variant keeps the forward pass and changes the gradient. All pass on CPU.

### 2.7 Complexity

Let \(c_{\mathrm{lin}}\) be the cost of one token passing through the linear layers of one block. Full attention pays, per token, an attention cost proportional to the length already read, and its KV cache grows linearly. Sliding-window attention and TTT-E2E pay an attention cost that depends on \(w\) only. On top of that, TTT-E2E pays two more terms per token: the forward pass of MLP′ in the suffix, and one backward pass through the suffix per chunk (roughly twice a suffix forward pass, plus the output layer). Neither depends on the length already read, so inference cost and state size are constant in context length. The price is the second-order training pass through the inner loop. Measured numbers are in Section 4.10.
