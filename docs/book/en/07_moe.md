# 19. MoE: routing turns a block into a dynamic graph

## Why copying an FFN is not MoE

If we copy one feed-forward network eight times and execute all eight for every token, we have made a larger dense model. A Mixture-of-Experts (MoE) layer instead lets a router choose a small number of experts for each token. Total parameter capacity can grow while the active compute per token stays closer to the selected experts.

The path for one hidden vector is:

~~~text
h
→ router logits
→ top-k expert IDs and gate weights
→ dispatch tokens by expert
→ expert FFN or MLP-prime
→ restore original token order
→ weighted sum + residual
~~~

The shape can remain correct even when the order is wrong. That is why routing needs semantic checks, not only a final <code>shape == expected</code> assertion.

<figure class="book-figure">
  <img src="../../../assets/figures/moe-routing.svg" alt="A router selects a small number of experts, applies capacity assignment, and combines their outputs">
  <figcaption><strong>Figure 5. MoE routing.</strong> The router, capacity rule, and grouped expert computation define the dynamic path for each token; reports must include routing statistics.</figcaption>
</figure>

## A four-token example

Assume four tokens, two experts, and `top_k=1`. The router may assign `[A,B,C,D]` to experts `[0,0,1,0]`. Expert 0 receives three rows and expert 1 receives one. After each expert runs its FFN, the dispatch code must put the outputs back into positions A, B, C, D before adding the residual. With `top_k=2`, each token is sent to two experts and the normalized gate weights combine their outputs:

$$
y_t=\sum_{e\in S_t}g_{t,e}f_e(h_t),\qquad
\sum_{e\in S_t}g_{t,e}=1.
$$

The router produces \(r_t\in\mathbb R^E\) for \(E\) experts. Top-k chooses \(S_t\) and the corresponding gate weights \(g_{t,e}\). A balanced router might send two of the four tokens to each expert, but the model does not guarantee that balance by itself.

For a two-expert top-1 router, one possible probability table is:

| token | \(p(E_1)\) | \(p(E_2)\) | selected expert |
|---|---:|---:|---|
| \(t_1\) | 0.80 | 0.20 | \(E_1\) |
| \(t_2\) | 0.70 | 0.30 | \(E_1\) |
| \(t_3\) | 0.40 | 0.60 | \(E_2\) |
| \(t_4\) | 0.30 | 0.70 | \(E_2\) |

The dispatch groups are therefore \(E_1:[t_1,t_2]\) and \(E_2:[t_3,t_4]\). Expert outputs must be restored to the original order \([t_1,t_2,t_3,t_4]\) before the residual addition. Swapping the second and third rows leaves the shape unchanged while giving two positions the wrong computation.

## Load balancing by hand

Let \(f_e\) be the fraction of tokens actually selected by expert \(e\), and let \(p_e\) be the mean router probability for that expert. A common auxiliary term is

$$
\mathcal L_{\mathrm{aux}}
=E\sum_{e=1}^{E}f_ep_e.
$$

For the table above, \(f=(0.5,0.5)\) and the mean probabilities are \(p=(0.55,0.45)\), so

$$
\mathcal L_{\mathrm{aux}}
=2(0.5\cdot0.55+0.5\cdot0.45)=1.
$$

If all four tokens go to \(E_1\), then \(f=(1,0)\). With mean probabilities \((0.7,0.3)\), the same term is \(2(1\cdot0.7)=1.4\), indicating a more concentrated route. This auxiliary value is not language-model NLL; publish both values.

## Capacity, drop, and auxiliary objectives

An expert's capacity is derived from the number of tokens in the step, `top_k`, and a capacity factor. If the assigned rows exceed capacity, the implementation must choose a policy: drop tokens, pad and mask them, or allow a larger allocation that may cause an out-of-memory error. Token drop saves memory but means those tokens did not receive a full expert computation.

The repository reports language-model NLL separately from routing auxiliary loss and router z loss. NLL asks whether the next token was predicted; auxiliary loss encourages balanced usage; z loss controls the scale of router logits. A lower combined value can come from an auxiliary term even when token prediction is unchanged, so publish each component.

If load bias is updated from expert counts, that update is another algorithmic state transition. It is not an ordinary AdamW parameter step; record its rate and whether it is included in the checkpoint.

With \(T\) tokens, \(E\) experts, top-k \(k\), and capacity factor \(\alpha\), a common per-expert capacity is

$$
C=\left\lceil\frac{Tk}{E}\alpha\right\rceil.
$$

For \(T=4\), \(E=2\), \(k=1\), and \(\alpha=1\), \(C=2\). If all four tokens select \(E_1\), only two fit. The remaining tokens must be dropped, padded and masked, redirected, or admitted through a larger allocation. The current reference <code>moe_forward</code> groups tokens by top-k but does not expose an independent capacity-factor/drop switch, so this calculation describes an implementation boundary rather than an existing command-line option.

## Code path and the three implementation levels

`ttt/models/ttt_e2e/moe.py` owns router scores, dispatch, expert computation, and order restoration. `grouped_gemm.py` provides grouped matrix multiplication paths. `ttt/distributed.py` defines data and expert mesh axes and sharding.

Keep three levels distinct:

1. **Mathematical path:** select experts, compute expert functions, combine outputs.
2. **Reference path:** often loops over experts and is easier to inspect.
3. **Optimized path:** groups rows into fewer kernels and may require stricter shapes or padding.

Correctness of the reference path does not prove that a grouped kernel is equivalent. Compare expert counts, order restoration, and logits before comparing timing.

## Expert parallelism

With expert parallelism, an expert's parameters live on one or more devices. Tokens may cross devices through all-to-all communication. `expert_parallel_size` therefore changes the compute graph, memory placement, and failure modes. A rank's local expert shape is only a shard of the global shape; report both.

Before making a quality claim, fix expert count, top-k, shared experts, capacity, drop policy, router losses, mesh, precision, and training tokens. `route_mlp`, `route_delta`, and `route_e2e` change the routing location and should not silently change width or budget at the same time.

## Small remote gate

The mesh configuration and validator provide a reproducible systems check:

~~~bash
bash scripts/run_gpu_train.sh \
  --config configs/runs/mesh_tiny.json \
  --data outputs/sample-tokens.npy \
  --output outputs/mesh-tiny

bash scripts/validate_mesh.sh \
  outputs/mesh-validation.json \
  outputs/mesh-checkpoint
~~~

The report should include global and local expert shapes, route counts, two-step loss, restore consistency, communication time, and memory. Start with a no-drop tiny configuration. This proves the path; it does not prove that a capacity policy or optimized kernel scales to a large model.

The next chapter defines an ablation protocol so changes in quality can be separated from changes in active compute and communication.
