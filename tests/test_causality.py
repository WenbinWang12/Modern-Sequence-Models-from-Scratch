"""未来 token 不得影响此前预测。"""

import jax.numpy as jnp
import numpy as np

from ttt.inner_loop import InnerLoopConfig, sequence_loss


def test_meta_sequence_has_no_future_leak(model_and_params):
    model, params = model_and_params
    tokens = jnp.asarray([1, 2, 3, 4, 5, 6, 7])
    changed = tokens.at[4:].set(jnp.asarray([11, 12, 13]))
    inner = InnerLoopConfig(learning_rate=0.1)
    before = sequence_loss(model, params, tokens, inner, return_logits=True)
    after = sequence_loss(model, params, changed, inner, return_logits=True)
    np.testing.assert_allclose(before.logits[:4], after.logits[:4], atol=1e-6)


def test_last_target_does_not_change_its_prediction(model_and_params):
    model, params = model_and_params
    tokens = jnp.asarray([1, 2, 3, 4, 5])
    inner = InnerLoopConfig(learning_rate=0.1)
    before = sequence_loss(model, params, tokens, inner, return_logits=True)
    after = sequence_loss(model, params, tokens.at[-1].set(12), inner, return_logits=True)
    np.testing.assert_allclose(before.logits, after.logits, atol=1e-6)
