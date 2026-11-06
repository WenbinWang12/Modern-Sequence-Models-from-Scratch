"""测试用小模型。"""

import jax
import pytest

from ttt.models.ttt_e2e import TTTE2EConfig, TTTE2EForCausalLM


@pytest.fixture(scope="session")
def model_and_params():
    config = TTTE2EConfig(
        vocab_size=19,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        sliding_window=4,
        max_position_embeddings=32,
        ttt_chunk_size=2,
        compute_dtype="float32",
        post_norm=False,
        layer_types=("full_attention", "sliding_attention"),
    )
    model = TTTE2EForCausalLM(config)
    return model, model.init_parameters(jax.random.PRNGKey(7))
