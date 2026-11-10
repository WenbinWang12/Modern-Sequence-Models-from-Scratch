# 17. Inner and outer loops: learning how to adapt

## Two clocks behind one prediction

The previous chapter followed a token through the forward path. Now imagine a paragraph that introduces an unfamiliar term. The model has made a prediction and has just observed the correct next token. We want that observation to help with the rest of the paragraph, and we want many training texts to teach the model how to use such observations. These are different problems, requiring different updates.

We keep the tiny hidden width \(d=16\) and chunk length \(C=4\). By the end of this chapter you should be able to identify the fast state used by a chunk, the label that completes its update, and the outer-gradient path through that update. The prerequisites are prefix/suffix computation and token cross-entropy. We will still derive the chain rule rather than treating automatic differentiation as an explanation.

Let \(\theta_k\) denote static parameters at outer training step \(k\), and let \(\phi_c\) be fast weights at the beginning of chunk \(c\) in one sequence. Here \(c\) counts chunks, not tokens. At the start of a sequence the code extracts \(\phi_0\) from \(\theta_k\); it does not sample a fresh set of random matrices.

~~~text
hold theta_k fixed
  phi_0 → chunk 0 predictions and loss → phi_1 → chunk 1 → phi_2
                    differentiate sequence loss w.r.t. theta_k
                                      ↓
                                  theta_{k+1}
~~~

The inner loop changes sequence state. The outer loop changes the static parameters and learned fast-weight initialization used by later training examples. A final \(\phi\) from one independent training sample is not used directly to initialize the next sample. At inference, retaining it means continuing the same session.

## From token cross-entropy to a fast-weight update

Start with an intentionally simplified two-class language model. Its vocabulary contains \(A,B\); one scalar fast parameter \(\phi\) produces logits \(\mathbf z=(0,\phi)\). There is no attention or MLP in this example. It isolates the derivative. For target \(B\),

$$
p_\phi(B)=\frac{e^\phi}{1+e^\phi},\qquad
\ell(\phi)=-\log p_\phi(B)=\log(1+e^\phi)-\phi.
$$

Differentiating the two terms gives

$$
\frac{d\ell}{d\phi}
=\frac{e^\phi}{1+e^\phi}-1
=p_\phi(B)-1.
$$

At \(\phi_0=0\), both candidates have probability \(0.5\), the loss is \(\log2\approx0.6931\), and the gradient is \(-0.5\). With learning rate \(1\), SGD gives \(\phi_1=0-1(-0.5)=0.5\). Substituting back gives \(p_{\phi_1}(B)\approx0.6225\) and loss \(0.4741\). This is a complete hand calculation of an observation changing a subsequent prediction.

If two positions have targets \(B,A\), and this toy model uses the same \((0,\phi)\) at both positions, their mean gradient at zero is \([(-0.5)+0.5]/2=0\). No update need not mean a software bug; the model may lack a representation that distinguishes the contexts. Real logits depend on position-specific hidden states, so the gradient is

$$
\nabla_\phi\mathcal L_{\mathrm{inner}}
=\frac{1}{\sum_t m_t}
\sum_t m_t
\left(\frac{\partial\mathbf z_t}{\partial\phi}\right)^{\mathsf T}
(\mathbf p_t-\operatorname{onehot}(y_t)),
\qquad \sum_t m_t>0.
$$

First compute each vocabulary logit's error, then propagate it through the Jacobian from fast parameters to logits. In the scalar example, the first logit has derivative \(0\) and the second has derivative \(1\), leaving \(p(B)-1\). The implementation differentiates a whole pytree of matrices; matrix entries do not directly regress onto the numerical values of token IDs.

A Jacobian is a table of derivatives: entry \(v,i\) describes how logit \(v\) changes when parameter \(i\) changes slightly. Flattening fast matrices into \(n_\phi\) parameters gives a \([V,n_\phi]\) table. Its transpose multiplies the \([V]\) error vector to produce a \([n_\phi]\) parameter gradient. Automatic differentiation usually computes this product directly without storing the entire table.

## Clipping changes the step, not the label

As a separate arithmetic illustration, consider \(\ell(\phi)=\tfrac12(\phi-3)^2\). At \(\phi_0=1\), the gradient is \(-2\), and SGD with learning rate \(0.5\) reaches \(2\). This squared-error example explains update arithmetic; it is not the actual inner objective.

For gradient leaves \(g_j\), the code computes the global norm and update

$$
\widetilde{\|g\|}_2=\sqrt{\sum_j\sum_i g_{j,i}^2+10^{-12}},
\quad
a=\min\left(1,\frac{\kappa}{\widetilde{\|g\|}_2}\right),
\quad
\phi'=\phi-\eta_{\mathrm{inner}}ag.
$$

The small value inside the square root matches code and avoids numerical problems in higher derivatives at zero gradient. With \(\kappa=0.5\), the scalar gradient is multiplied by approximately \(0.25\), giving approximately \(1.25\). This constrains the length of the entire gradient tree, not each entry separately or the parameter norm. Record learning rate, clipping threshold, gradient norm, and update norm separately; a small state change cannot identify its cause.

## Four input positions require five tokens

A block of \(C\) predictions needs \(C\) labels. Inputs \(x_1,x_2,x_3,x_4\) require targets \(x_2,x_3,x_4,x_5\). An online reader at \(x_4\) does not yet know the final label \(x_5\); that observation must arrive before the update completes.

| Stage | Mathematical input positions | State used for prediction | Labels needed for update | Resulting state |
|---|---|---|---|---|
| chunk 0 | \(1,\ldots,4\) | \(\phi_0\) | \(x_2,\ldots,x_5\) | \(\phi_1\) |
| chunk 1 | \(5,\ldots,8\) | \(\phi_1\) | \(x_6,\ldots,x_9\) | \(\phi_2\) |
| fewer than 4 remaining prediction positions | remaining positions | current \(\phi\) | available real labels | score, but retain old \(\phi\) |

Training already has the entire sequence in memory but must preserve the same causal order: predict this block first, and let later blocks use its update. <code>sequence_loss</code> pads the tail for fixed shapes, scores valid labels, and conditionally retains old fast weights for an incomplete chunk. A nonzero tail mask alone is not permission to update.

## Mapping equations to functions and arrays

In <code>ttt/inner_loop.py</code>, <code>inner_step</code> runs the suffix, computes cross-entropy, differentiates fast weights, and calls <code>sgd_update</code>. Its output cache comes from pre-update prediction; historical KV is not re-encoded after the update. Optional <code>track_post_loss</code> evaluates the same block again. It is an adaptation diagnostic, not new held-out evidence.

| Code object | Single-sequence shape or type | Meaning |
|---|---|---|
| <code>hidden</code> | \([C,d]\) | prefix output |
| <code>targets</code> | \([C]\) | next-token integer IDs |
| <code>mask</code> | \([C]\) | target weights |
| <code>fast_weights</code> | pytree | arrays changed by the inner loop |
| <code>cache</code> | tuple of layer states | history read by the suffix |
| scan carry | \((\phi_c,\text{cache}_c)\) | state passed to the next chunk |

<code>sequence_loss</code> uses two levels of <code>lax.scan</code> and <code>jax.checkpoint</code> to recompute intermediate values during backward. This checkpoint is computational rematerialization, not a checkpoint file on disk. It reduces saved activations without itself detaching gradients.

## Combining examples in the outer objective

If sample \(b\) has \(n_b\) valid targets and mean loss \(\mathcal L_b\), the batch's token mean is

$$
\mathcal L_{\mathrm{batch}}
=\frac{\sum_b n_b\mathcal L_b}{\max(\sum_b n_b,1)}.
$$

Combining 100 targets at mean loss 2 with 2 targets at mean loss 10 gives \(220/102\approx2.157\), not \((2+10)/2=6\). The denominator guard prevents division by zero; it does not make an all-zero mask useful training data.

<code>ttt/training.py</code> differentiates the objective with respect to static parameters and passes gradients to the outer Optax optimizer. MoE regularization enters the training objective while <code>nll</code> remains separate. The inner learning rate has its own warmup, moving linearly from <code>inner_lr_start</code> to its target. The outer schedule uses warmup and cosine decay. Fix and record both schedules in a comparison.

## Why the outer derivative contains second derivatives

Let the initialization be \(\phi_0(\theta)\). First expand one SGD update when clipping is inactive:

$$
\phi_1=\phi_0(\theta)-\eta g(\theta,\phi_0(\theta)),
\qquad g=\nabla_\phi\mathcal L_{\mathrm{inner}}.
$$

Set \(J_0=d\phi_0/d\theta\), and assume \(\eta\) does not depend on \(\theta\) for this step. Then

$$
\frac{d\phi_1}{d\theta}
=J_0-\eta\left(
\frac{\partial g}{\partial\theta}
+\frac{\partial g}{\partial\phi}J_0
\right),
$$

$$
\frac{d\mathcal L_{\mathrm{outer}}}{d\theta}
=\frac{\partial\mathcal L_{\mathrm{outer}}}{\partial\theta}
+\frac{\partial\mathcal L_{\mathrm{outer}}}{\partial\phi_1}
\frac{d\phi_1}{d\theta}.
$$

Because \(g\) is already a first derivative, differentiating it introduces second-order terms. A full meta-gradient retains them. A common first-order approximation ignores derivatives of \(g\) while retaining \(J_0\); applying <code>stop_gradient</code> to all of \(\phi_1\) also removes the initialization path. These are different approximations and cannot both be described simply as “first order.” See [MAML](https://arxiv.org/abs/1703.03400v3) for the original background on learning an adaptable initialization.

The earlier binary model makes the three outcomes calculable by hand. Set \(\phi_0=\theta=0\), inner target \(B\), and learning rate 1, giving \(\phi_1=0.5\). Since \(g(\phi)=p_\phi(B)-1\), we have \(dg/d\phi=p_\phi(B)(1-p_\phi(B))=0.25\) at zero. Consequently, \(d\phi_1/d\theta=1-0.25=0.75\).

To isolate the future-prediction derivative, temporarily choose the next position's target \(A\) as the outer objective: \(\mathcal L_{\rm future}=\log(1+e^{\phi_1})\). Its derivative with respect to \(\phi_1\) is \(0.6225\). The full chain rule gives approximately \(0.6225\times0.75=0.4669\); ignoring the inner gradient's derivative gives \(0.6225\); detaching all of \(\phi_1\) gives \(0\). These are different computations, not three ways to write the same gradient. The real sequence objective includes losses at other positions; this isolated future term makes the path through adaptation visible.

When clipping activates, replace \(g\) with \(a(g)g\); its derivative includes the clipping scale's dependence. The unclipped derivation is therefore not the whole implementation. The current inner loop has no configurable FOMAML switch. Its outer loss differentiates through the implemented update. Adding a first-order comparison requires specifying which derivative is removed and checking that computation; an experiment report cannot claim an existing option that the code does not expose.

## Minimal remote validation and report reading

On a remote GPU node with dependencies installed, run from the repository root:

~~~bash
bash scripts/validate_gpu.sh outputs/ch17/validation.json
~~~

This runs the built-in tiny cases in <code>ttt.validate</code>. It does not load your formal training corpus or start a 125M run. The report has <code>passed</code>, <code>environment</code>, and <code>cases</code>. Actual case fields include <code>loss</code>, <code>finite_gradients</code>, <code>logits_max_error</code>, <code>compilation_seconds</code>, <code>training_seconds</code>, <code>state_bytes</code>, and <code>inner_updates</code>.

Success requires finite loss and gradients, streaming error within the script's \(10^{-3}\) limit, and the internal causality assertion after modifying future tokens. Causality has no separate JSON boolean: <code>passed=true</code> is written only after the assertion succeeds. If no GPU is found, inspect the backend first. For excessive error, inspect chunk labels and caches; for nonfinite gradients, inspect learning rate, clipping, and precision. Failure reports contain <code>passed=false</code>, <code>error_type</code>, and <code>error</code>.

## Summary: an update is not a quality improvement

We have connected token loss, fast-weight gradients, later chunk state, and the outer derivative. The functional report establishes the listed checks for small configurations. It does not establish generalization on a real corpus, cover every length, or test session snapshot recovery.

A <code>C2/F2</code> (or <code>C4/F4</code>) comparison fixes architecture, initialization, data, budget, and outer schedules, changing only whether the inner update runs. The number names the recipe's full-attention layer count. Removing MLP-prime is a separate structural ablation. The next chapter carries the same label-availability constraint into token-by-token inference, where caches, incomplete chunks, and fast weights must advance together.
