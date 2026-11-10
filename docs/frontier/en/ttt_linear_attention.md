# TTT and linear attention: where similarity begins and equivalence ends

**Source snapshot: 2026-11-11. Code review baseline: <code>0614bd5</code>. Status: an executable local algebra example; independent model-equivalence and quality experiments are not implemented.**

## From “remember this sentence” to a precise question

Suppose a model reads “the door number is 42” and must answer the number later. Attention can retain keys and values for earlier positions. Another approach updates a small model to map a key to its value. Both use history, but that intuition does not establish equivalence: read/write order, loss, normalization, and state parameterization change the computation.

We first study a linear model small enough to calculate by hand. By the end, readers should identify the step that produces a delta rule and explain why it does not prove that this repository's TTT-E2E is ordinary linear attention. The prerequisites are matrix multiplication and squared-loss gradients; see [inner and outer loops](../../book/en/05_inner_outer_loop.md) for the difference from next-token cross-entropy.

## Sources and the claim under investigation

[Learning to (Learn at Test Time), v4, 2025-08-31](https://arxiv.org/abs/2407.04620v4) uses learnable model parameters as sequence state and studies TTT-Linear and TTT-MLP. [Test-Time Training with KV Binding Is Secretly Linear Attention, v4, 2026-05-12](https://arxiv.org/abs/2602.21204v4) analyzes a class of KV-binding TTT mechanisms in relation to learned linear attention.

Our check concerns an update under explicit assumptions. The title cannot be expanded into “every gradient update equals standard linear attention,” nor can we skip the mapping from a paper's model to the local next-token MLP-prime.

## Deriving a linear-model update

Let column vectors \(k_t\in\mathbb R^{d_k}\) and \(v_t\in\mathbb R^{d_v}\) be the key and value, and let \(W_t\in\mathbb R^{d_v\times d_k}\) be the fast matrix. We use lowercase \(v_t\) for values to avoid confusing them with vocabulary size \(V\). Define the squared reconstruction loss:

$$
\ell_t(W)=\frac12\|Wk_t-v_t\|_2^2.
$$

Write the residual as \(e_t=Wk_t-v_t\). Element \(W_{ij}\) affects only residual component \(i\), with derivative \(k_{t,j}\). Thus
\(\partial\ell_t/\partial W_{ij}=e_{t,i}k_{t,j}\). Reassembling the elements gives

$$
\nabla_W\ell_t=(Wk_t-v_t)k_t^\mathsf T.
$$

One SGD step with learning rate \(\eta\) is

$$
W_t=W_{t-1}-\eta(W_{t-1}k_t-v_t)k_t^\mathsf T
=W_{t-1}(I-\eta k_tk_t^\mathsf T)+\eta v_tk_t^\mathsf T.
$$

The first term corrects existing state; the second writes a new association. This is a delta update for the stated linear model and loss. By comparison, a simple additive linear-attention state is \(S_t=S_{t-1}+v_tk_t^\mathsf T\). It lacks the prediction-error correction, so it generally differs from \(W_t\). Decay, normalization denominators, and learned feature maps each need separate definitions.

For \(W_0=0\), \(k_1=(1,0)^\mathsf T\), \(v_1=(2,-1)^\mathsf T\), and \(\eta=0.5\),

$$
W_1=\begin{pmatrix}1&0\\-0.5&0\end{pmatrix}.
$$

Using the same key again with \(v_2=(0,1)^\mathsf T\) gives
\(W_2=\bigl(\begin{smallmatrix}0.5&0\\0.25&0\end{smallmatrix}\bigr)\).
Simply adding \(\eta v_2k_2^\mathsf T\) to \(W_1\) instead produces first column \((1,0)^\mathsf T\). The second step exposes the difference; checking only the first step from zero state misses it.

For a query column vector \(q_t\in\mathbb R^{d_k}\), reading the updated state gives \(W_tq_t\in\mathbb R^{d_v}\). This defines a write-then-read convention for the example. The repository's next-token TTT-E2E predicts a chunk first, then updates with observed labels for subsequent chunks. An algebraic identity does not imply an identical causal protocol.

## Mapping to the current code

| Capability | Status | Entry point and limit |
|---|---|---|
| Gated delta recurrence | Implemented | <code>ttt/models/ttt_e2e/gated_delta.py</code>; also includes convolution, key/query normalization, decay, and output gating |
| Next-token fast-weight update | Implemented | <code>ttt/inner_loop.py</code>; MLP-prime uses token cross-entropy, not the KV squared reconstruction above |
| Two-step linear identity below | Teaching experiment | Self-contained NumPy command, independent of the trainer |
| Equal-state-size TTT-Linear/linear-attention training comparison | Planned | No independent configuration, data protocol, or result report |

## Minimal run and expected artifact

After logging into the remote CPU machine, run this complete snippet in an environment with NumPy. It checks that two forms agree and that the additive update differs. It was not executed during this editing pass.

~~~bash
mkdir -p outputs/frontier-linear
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np

w = np.zeros((2, 2))
additive = w.copy()
k = np.array([1.0, 0.0])
eta = 0.5
errors = []
for v in (np.array([2.0, -1.0]), np.array([0.0, 1.0])):
    gradient_form = w - eta * np.outer(w @ k - v, k)
    recurrence_form = w @ (np.eye(2) - eta * np.outer(k, k)) + eta * np.outer(v, k)
    errors.append(float(np.max(np.abs(gradient_form - recurrence_form))))
    np.testing.assert_allclose(gradient_form, recurrence_form, atol=1e-12, rtol=0)
    w = gradient_form
    additive += eta * np.outer(v, k)
np.testing.assert_allclose(w, [[0.5, 0.0], [0.25, 0.0]])
assert not np.allclose(w, additive)
report = {"scope": "two-step-linear-squared-loss",
          "max_errors": errors, "delta_state": w.tolist(),
          "additive_state": additive.tolist(), "passed": True}
Path("outputs/frontier-linear/algebra.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
PY
~~~

Both <code>max_errors</code> should be near zero and <code>delta_state</code> should match the calculation. On failure, first check column-vector conventions, outer-product orientation, and update timing. This is not a language-model evaluation or a verification of all parallel forms in the papers.

## Controls, measurements, and decisions

The complete experiment must first fix \(d_k,d_v\), initial state, keys/values/queries, learning rate, loss, and read/write order. Only then should it compare feature maps, gates, or training quality. Replacing several components together prevents attribution.

| Question | Criterion | Current record |
|---|---|---|
| Does two-step SGD equal the expanded recurrence? | Maximum float64 error at most \(10^{-12}\) | Command supplied; not executed |
| Does local Gated Delta equal the simplified update? | Compare state/output after explicitly removing extra transforms | Comparison not established |
| Is quality comparable at matched parameter/compute budgets? | Held-out loss, valid tokens, FLOPs, throughput, memory | Not run |

Different initial states, masks, or update schedules indicate a protocol mismatch, not a refuted method. The next run saves the algebra report, then implements explicit TTT-Linear and recurrence paths. Until then, this topic supports a conditional derivation, not general equivalence or performance claims.
