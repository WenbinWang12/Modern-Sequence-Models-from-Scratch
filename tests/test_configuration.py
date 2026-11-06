"""配置校验与实验组定义。"""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from ttt.checkpoint import save_checkpoint
from ttt.config import ExperimentConfig, load_experiment
from ttt.inner_loop import InnerLoopConfig
from ttt.models import AutoConfig, AutoModelForCausalLM
from ttt.models.ttt_e2e.configuration_ttt_e2e import TTTE2EConfig
from ttt.training import TrainConfig, TrainState


def test_defaults_and_roundtrip():
    config = TTTE2EConfig()
    assert config.fast_intermediate_size == config.intermediate_size == 2048
    assert config.ttt_layer_indices == (9, 10, 11)
    assert config.head_dim == 64
    assert TTTE2EConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    with pytest.raises(FrozenInstanceError):
        config.hidden_size = 1


@pytest.mark.parametrize(
    "changes",
    [
        {"hidden_size": 0},
        {"vocab_size": True},
        {"num_key_value_heads": 5},
        {"head_dim": 3},
        {"layer_types": ["full_attention"]},
        {"ttt_suffix_length": 13},
        {"use_fast_mlp": False, "ttt_suffix_length": 3},
        {"ttt_chunk_size": 9000},
        {"rope_theta": float("nan")},
        {"bos_token_id": -1},
    ],
)
def test_invalid_configuration(changes):
    with pytest.raises(ValueError):
        TTTE2EConfig(**changes)


def test_paired_controls():
    directory = Path(__file__).parents[1] / "configs" / "125m"
    sk, ck, fk = (load_experiment(directory / f"{name}.json") for name in ("sk", "ck", "fk"))
    assert ck.model == fk.model
    assert not sk.model.use_fast_mlp
    assert not ck.inner.enabled and fk.inner.enabled
    assert sk.model.layer_types == ck.model.layer_types
    assert ck.model.intermediate_size == sk.model.intermediate_size


def test_pretrained_config_and_auto_registry(tmp_path):
    config = TTTE2EConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        sliding_window=4,
        ttt_chunk_size=2,
        compute_dtype="float32",
        post_norm=False,
    )
    config.save_pretrained(tmp_path)
    loaded = AutoConfig.from_pretrained(tmp_path)
    assert loaded == config
    assert (
        AutoConfig.from_pretrained(Path(__file__).parents[1] / "configs" / "125m" / "ck.json").model_type == "ttt_e2e"
    )
    model = AutoModelForCausalLM.from_config(loaded)
    assert model.config == config
    assert AutoModelForCausalLM.from_config(loaded, mode="train").gradient_checkpointing
    assert not AutoModelForCausalLM.from_config(loaded, mode="infer").gradient_checkpointing
    with pytest.raises(ValueError, match="mode"):
        AutoModelForCausalLM.from_config(loaded, mode="unknown")

    experiment = ExperimentConfig(
        loaded, InnerLoopConfig(), TrainConfig(steps=1, sequence_length=8), tokenizer="custom"
    )
    params = model.init_parameters(jax.random.PRNGKey(0))
    checkpoint_path = tmp_path / "step_00000001.npz"
    save_checkpoint(checkpoint_path, TrainState(params, None, jnp.asarray(1)), experiment, {"cursor": 1})
    restored = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    assert restored.params is not None
    assert restored.forward(input_ids=jnp.asarray([1, 2], dtype=jnp.int32)).logits.shape == (2, 32)
