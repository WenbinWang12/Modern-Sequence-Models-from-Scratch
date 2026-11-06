"""验证设备分片、二阶训练和分片检查点恢复。"""

import argparse
import tempfile
import time
from pathlib import Path

import jax
import numpy as np

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ExperimentConfig
from .distributed import create_mesh, initialize_distributed, initialize_mesh_state, make_mesh_train_step
from .inner_loop import InnerLoopConfig
from .lifecycle import atomic_json
from .models.ttt_e2e import TTTE2EConfig, TTTE2EForCausalLM
from .provenance import environment_report
from .training import TrainConfig, make_optimizer


def validate_mesh(*, scale="tiny", expert_parallel_size=2, output_dir=None, require_gpu=False):
    initialize_distributed()
    if require_gpu and any(device.platform != "gpu" for device in jax.local_devices()):
        raise RuntimeError("本次验证要求 GPU。")
    large = scale == "1b"
    config = TTTE2EConfig(
        vocab_size=128256 if large else 32,
        hidden_size=2048 if large else 16,
        intermediate_size=5376 if large else 32,
        num_hidden_layers=24 if large else 2,
        num_attention_heads=32 if large else 2,
        num_key_value_heads=8 if large else 1,
        num_experts=4,
        num_experts_per_token=2,
        moe_intermediate_size=1024 if large else 16,
        first_k_dense_layers=2 if large else 0,
        ttt_suffix_length=6 if large else 1,
        max_position_embeddings=16,
        sliding_window=4,
        ttt_chunk_size=2,
        compute_dtype="float16" if large else "float32",
        post_norm=large,
        attention_query_block_size=4,
        attention_key_block_size=4,
        loss_token_block_size=2,
        loss_vocab_block_size=4096 if large else 32,
    )
    training = TrainConfig(
        steps=2,
        batch_size=2,
        sequence_length=4,
        warmup_steps=0,
        inner_warmup_steps=0,
        gradient_checkpointing=True,
        parallelism="mesh",
        expert_parallel_size=expert_parallel_size,
    )
    inner = InnerLoopConfig(learning_rate=0.1)
    experiment = ExperimentConfig(config, inner, training, tokenizer="custom")
    model, mesh = TTTE2EForCausalLM(config, True), create_mesh(expert_parallel_size)
    optimizer = make_optimizer(training)
    start = time.perf_counter()
    state, shardings = initialize_mesh_state(model, optimizer, jax.random.PRNGKey(41), mesh)
    jax.block_until_ready(state)
    init_seconds = time.perf_counter() - start
    parameter_count = sum(value.size for value in jax.tree.leaves(state.params))
    batch_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None))
    batch = np.asarray([[1, 2, 3, 4, 5], [3, 4, 5, 6, 7]], dtype=np.int32)
    ids = jax.make_array_from_callback(batch.shape, batch_sharding, lambda indices: batch[indices])
    masks = jax.make_array_from_callback(
        (2, 4), batch_sharding, lambda indices: np.ones((2, 4), dtype=np.float32)[indices]
    )
    step = make_mesh_train_step(model, inner, training, optimizer, mesh, shardings)
    timings, history = [], []
    for _ in range(2):
        start = time.perf_counter()
        state, metrics = step(state, ids, masks)
        metrics = {name: float(value) for name, value in metrics.items()}
        if not all(np.isfinite(value) for value in metrics.values()):
            raise AssertionError("分片训练出现非有限指标。")
        timings.append(time.perf_counter() - start)
        history.append(metrics)
    leaves = jax.tree.leaves(state.params["layers"][-1]["moe"]["experts"])
    expert_shard_shapes = [
        {"global": list(value.shape), "local": list(value.addressable_shards[0].data.shape)} for value in leaves
    ]
    checkpoint_roundtrip = False
    # 大规模测试避免额外写入十余 GB；小规模完整检查优化器和下一步结果。
    if not large:
        if jax.process_count() > 1 and output_dir is None:
            raise ValueError("多机验证需要共享 output_dir。")
        root = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="ttt-mesh-"))
        checkpoint = root / "mesh-step-2.orbax"
        save_checkpoint(checkpoint, state, experiment, {"test": "mesh"})
        restored = load_checkpoint(checkpoint, load_optimizer=True, target_shardings=shardings).state
        for left, right in zip(jax.tree.leaves(state), jax.tree.leaves(restored)):
            for a, b in zip(left.addressable_shards, right.addressable_shards):
                np.testing.assert_array_equal(a.data, b.data)
        expected, expected_metrics = step(state, ids, masks)
        actual, actual_metrics = step(restored, ids, masks)
        for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
            for x, y in zip(a.addressable_shards, b.addressable_shards):
                np.testing.assert_array_equal(x.data, y.data)
        assert float(expected_metrics["loss"]) == float(actual_metrics["loss"])
        checkpoint_roundtrip = True
    memory = [device.memory_stats() for device in jax.local_devices()]
    return {
        "passed": True,
        "scale": scale,
        "parameters": parameter_count,
        "mesh": dict(mesh.shape),
        "init_seconds": init_seconds,
        "step_seconds": timings,
        "metrics": history,
        "expert_shards": expert_shard_shapes,
        "checkpoint_roundtrip": checkpoint_roundtrip,
        "device_memory": memory,
        "experiment": experiment.to_dict(),
        "environment": environment_report(),
    }


def main():
    parser = argparse.ArgumentParser(description="验证多设备 TTT 训练")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", choices=("tiny", "1b"), default="tiny")
    parser.add_argument("--expert-parallel-size", type=int, default=2)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    try:
        report = validate_mesh(
            scale=args.scale,
            expert_parallel_size=args.expert_parallel_size,
            output_dir=args.checkpoint_dir,
            require_gpu=args.require_gpu,
        )
    except BaseException as error:
        if jax.process_index() == 0:
            atomic_json(
                args.output,
                {
                    "passed": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "environment": environment_report(),
                },
            )
        raise
    if jax.process_index() == 0:
        atomic_json(args.output, report)


if __name__ == "__main__":
    main()
