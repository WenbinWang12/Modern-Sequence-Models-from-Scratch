# 14. Dense causal-language-model baseline

[中文版](../20_baseline_model.md)

The dense baseline is the control for TTT, Loop, and MoE. It fixes tokenizer, manifests, sequence length, effective token budget, initialization, optimizer, schedule, seed, precision, hardware, and checkpoint format before adding dynamic state or routing.

For logits z at position t and target y_t:

$$
\ell_t=-\log\frac{\exp z_{t,y_t}}
{\sum_v \exp z_{t,v}}.
$$

The repository can compute cross-entropy in token and vocabulary blocks, preserving the same objective while reducing peak memory. Masked loss uses the sum of valid-token losses divided by the valid-token count.

The baseline disables experts, fast MLP, inner updates, and loop mode. It can still choose sliding or full attention, but attention type must be recorded as a separate variable.

Each training step samples a window, runs sequence_loss, computes outer gradients, applies Optax, records loss and state statistics, and periodically checkpoints. A finite decreasing loss is a health signal, not evidence of broad capability; tiny data can simply be memorized.

For each later comparison, report data and tokens, total and active parameters, optimizer and inner-loop settings, hardware and precision, compile and steady cost, memory, communication, training and held-out loss, and restore behavior. The next part follows a token through the TTT forward path.
