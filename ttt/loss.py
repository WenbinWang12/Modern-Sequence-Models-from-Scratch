"""不构造完整词表 logits 的交叉熵。"""

import jax
import jax.numpy as jnp


def chunked_cross_entropy(hidden, weight, targets, mask, *, bias=None, token_block=64, vocab_block=4096):
    length, dimension = hidden.shape
    vocabulary = weight.shape[0]
    token_block = min(token_block, length)
    vocab_block = min(vocab_block, vocabulary)
    token_pad, vocab_pad = (-length) % token_block, (-vocabulary) % vocab_block
    features = jnp.pad(hidden, ((0, token_pad), (0, 0))).reshape(-1, token_block, dimension)
    labels = jnp.pad(targets, (0, token_pad)).reshape(-1, token_block)
    weights = jnp.pad(weight, ((0, vocab_pad), (0, 0))).reshape(-1, vocab_block, dimension)
    if bias is None:
        bias = jnp.zeros(vocabulary, dtype=hidden.dtype)
    biases = jnp.pad(bias, (0, vocab_pad)).reshape(-1, vocab_block)
    minimum = jnp.finfo(jnp.float32).min

    def token_step(unused, batch):
        current_hidden, current_targets = batch
        initial = (jnp.full((token_block,), minimum), jnp.zeros(token_block), jnp.zeros(token_block))

        def vocab_step(carry, vocab_batch):
            maximum, denominator, target_score = carry
            current_weight, current_bias, index = vocab_batch
            scores = (current_hidden @ current_weight.T).astype(jnp.float32) + current_bias.astype(jnp.float32)
            ids = index * vocab_block + jnp.arange(vocab_block)
            scores = jnp.where(ids[None] < vocabulary, scores, minimum)
            next_maximum = jnp.maximum(maximum, jnp.max(scores, axis=-1))
            denominator = jnp.exp(maximum - next_maximum) * denominator + jnp.sum(
                jnp.exp(scores - next_maximum[:, None]),
                axis=-1,
            )
            target_score += jnp.sum(jnp.where(current_targets[:, None] == ids[None], scores, 0), axis=-1)
            return (next_maximum, denominator, target_score), None

        (maximum, denominator, target_score), _ = jax.lax.scan(
            jax.checkpoint(vocab_step),
            initial,
            (weights, biases, jnp.arange(weights.shape[0], dtype=jnp.int32)),
        )
        return unused, maximum + jnp.log(denominator) - target_score

    _, losses = jax.lax.scan(jax.checkpoint(token_step), None, (features, labels))
    losses = losses.reshape(-1)[:length]
    return jnp.where(mask > 0, losses * mask, 0.0)
