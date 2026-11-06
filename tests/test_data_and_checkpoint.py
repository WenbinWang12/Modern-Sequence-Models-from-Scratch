"""数据采样与断点恢复。"""

import jax
import jax.numpy as jnp
import numpy as np

from ttt.checkpoint import load_checkpoint, save_checkpoint
from ttt.config import ExperimentConfig
from ttt.data import BatchStream, ByteTokenizer, prepare_data
from ttt.inner_loop import InnerLoopConfig
from ttt.training import TrainConfig, TrainState, make_optimizer, make_train_step


def test_byte_tokenizer_and_data(tmp_path):
    tokenizer = ByteTokenizer()
    text = "配置 TTT test"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    source = tmp_path / "text.txt"
    source.write_text(text * 5, encoding="utf-8")
    output = tmp_path / "tokens.npy"
    count = prepare_data(source, output, tokenizer)
    assert np.load(output).size == count
    stream = BatchStream(output, sequence_length=4, batch_size=2, vocab_size=259, seed=1)
    state = stream.state_dict()
    first = stream.next_batch()
    stream.load_state_dict(state)
    np.testing.assert_array_equal(stream.next_batch(), first)


def test_checkpoint_restores_next_optimizer_step(model_and_params, tmp_path):
    model, params = model_and_params
    training = TrainConfig(steps=3, sequence_length=4, warmup_steps=0, inner_warmup_steps=0)
    inner = InnerLoopConfig(learning_rate=0.1)
    experiment = ExperimentConfig(model.config, inner, training, tokenizer="custom")
    optimizer = make_optimizer(training)
    state = TrainState(params, optimizer.init(params), jnp.asarray(0))
    step = make_train_step(model, inner, training, optimizer)
    tokens = jnp.asarray([[1, 2, 3, 4, 5]])
    masks = jnp.ones((1, 4))
    state, _ = step(state, tokens, masks)
    path = tmp_path / "checkpoint.npz"
    save_checkpoint(path, state, experiment, {"test": 1})
    loaded = load_checkpoint(path, load_optimizer=True)
    assert loaded.data_state == {"test": 1}
    expected, _ = step(state, tokens, masks)
    actual, metrics = step(loaded.state, tokens, masks)
    assert np.isfinite(float(metrics["loss"]))
    for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        np.testing.assert_array_equal(left, right)
