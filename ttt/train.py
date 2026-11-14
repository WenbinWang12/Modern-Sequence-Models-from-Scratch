"""训练入口。"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .checkpoint import load_checkpoint, save_checkpoint
from .config import load_experiment
from .data import BatchStream
from .distributed import (
    create_mesh,
    global_batch,
    initialize_distributed,
    initialize_mesh_state,
    make_mesh_train_step,
    mesh_state_shapes,
)
from .lifecycle import StopSignal, atomic_json, latest_checkpoint
from .models.ttt_e2e.modeling_ttt_e2e import TTTE2EForCausalLM
from .provenance import environment_report, run_metadata
from .training import TrainState, make_optimizer, make_train_step


def train(experiment, data_path, output_dir, *, resume=None, initialize_from=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "run_metadata.json", run_metadata(experiment, data_path, output_dir))
    atomic_json(output_dir / "status.json", {"status": "initializing", "stage": "startup"})
    initialize_distributed()
    training = experiment.training
    model = TTTE2EForCausalLM(experiment.model, training.gradient_checkpointing)
    optimizer = make_optimizer(training)
    devices = jax.local_devices()
    mesh_mode = training.parallelism == "mesh" or training.expert_parallel_size > 1 or jax.process_count() > 1
    distributed = not mesh_mode and training.data_parallel and len(devices) > 1
    if training.parallelism == "pmap" and jax.process_count() > 1:
        raise ValueError("多机运行使用 mesh，pmap 仅用于单机。")
    if distributed and training.batch_size % len(devices):
        raise ValueError("数据并行时 batch_size 必须能被本机设备数整除。")
    if jax.process_count() > 1 and training.checkpoint_format != "orbax":
        raise ValueError("多机训练必须使用 Orbax 检查点。")
    stream = BatchStream(
        data_path,
        sequence_length=training.sequence_length,
        batch_size=training.batch_size,
        vocab_size=experiment.model.vocab_size,
        seed=training.seed,
    )
    mesh = create_mesh(training.expert_parallel_size) if mesh_mode else None
    state_shardings = None
    if mesh_mode and (resume is not None or initialize_from is not None):
        state, state_shardings = mesh_state_shapes(
            model, optimizer, jax.random.PRNGKey(training.seed), mesh, shard_parameters=training.shard_parameters
        )
    elif mesh_mode:
        state, state_shardings = initialize_mesh_state(
            model,
            optimizer,
            jax.random.PRNGKey(training.seed),
            mesh,
            shard_parameters=training.shard_parameters,
        )
    else:
        params = model.init_parameters(jax.random.PRNGKey(training.seed))
        state = TrainState(params, optimizer.init(params), jnp.asarray(0, dtype=jnp.int32))
    if resume is not None:
        checkpoint = load_checkpoint(resume, load_optimizer=True, target_shardings=state_shardings)
        if checkpoint.experiment != experiment:
            raise ValueError("恢复训练时必须使用与 checkpoint 相同的实验配置。")
        state = checkpoint.state
        if checkpoint.data_state is None:
            raise ValueError("checkpoint 缺少数据采样状态。")
        stream.load_state_dict(checkpoint.data_state)
    elif initialize_from is not None:
        checkpoint = load_checkpoint(initialize_from, target_shardings=state_shardings)
        expected = jax.eval_shape(model.init_parameters, jax.random.PRNGKey(0))
        if jax.tree.structure(expected) != jax.tree.structure(checkpoint.state.params):
            raise ValueError("初始化检查点的参数结构不匹配。")
        for shape, value in zip(jax.tree.leaves(expected), jax.tree.leaves(checkpoint.state.params)):
            if shape.shape != value.shape or shape.dtype != value.dtype:
                raise ValueError("初始化检查点的参数形状或精度不匹配。")
        params = checkpoint.state.params
        opt_state = (
            jax.jit(optimizer.init, out_shardings=state_shardings.opt_state)(params)
            if mesh_mode
            else optimizer.init(params)
        )
        state = TrainState(
            params,
            opt_state,
            jax.device_put(jnp.asarray(0, dtype=jnp.int32), state_shardings.step) if mesh_mode else state.step,
        )
    if mesh_mode:
        step_fn = make_mesh_train_step(model, experiment.inner, training, optimizer, mesh, state_shardings)
    else:
        step_fn = make_train_step(model, experiment.inner, training, optimizer, distributed=distributed)
    completed = int(state.step)
    if distributed:
        state = jax.device_put_replicated(state, devices)

    def host_state():
        return jax.tree.map(lambda value: value[0], state) if distributed else state

    primary = jax.process_index() == 0
    if primary:
        config_path = output_dir / "experiment.json"
        if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != experiment.to_dict():
            raise ValueError("输出目录已属于另一份实验配置。")
        atomic_json(config_path, experiment.to_dict())
        atomic_json(output_dir / "environment.json", environment_report())
        atomic_json(output_dir / "status.json", {"status": "running", "step": completed})
    extension = ".npz" if training.checkpoint_format == "npz" else ".orbax"
    saved = None
    log = (output_dir / "metrics.jsonl").open("a", encoding="utf-8") if primary else None
    try:
        with StopSignal() as stopping:
            while completed < training.steps:
                started = time.perf_counter()
                if mesh_mode:
                    batch, masks = global_batch(stream, completed, mesh, training.batch_size)
                else:
                    batch, masks = stream.next_batch_with_masks()
                    batch, masks = jnp.asarray(batch), jnp.asarray(masks)
                    if distributed:
                        batch = batch.reshape(len(devices), -1, training.sequence_length + 1)
                        masks = masks.reshape(len(devices), -1, training.sequence_length)
                state, device_metrics = step_fn(state, batch, masks)
                metrics = {name: float(value[0] if distributed else value) for name, value in device_metrics.items()}
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("训练出现非有限损失或梯度。")
                completed += 1
                seconds = time.perf_counter() - started
                metrics.update(
                    step=completed,
                    seconds=seconds,
                    tokens_seen=completed * training.batch_size * training.sequence_length,
                    tokens_per_second=training.batch_size * training.sequence_length / seconds,
                )
                if primary and completed % training.log_every == 0:
                    memory = devices[0].memory_stats()
                    if memory:
                        metrics["device_peak_bytes"] = memory.get("peak_bytes_in_use", 0)
                    message = json.dumps(metrics, ensure_ascii=False)
                    print(message, flush=True)
                    log.write(message + "\n")
                    log.flush()
                stop = stopping.requested
                if jax.process_count() > 1:
                    from jax.experimental import multihost_utils

                    stop = bool(np.any(multihost_utils.process_allgather(np.asarray(stop))))
                if completed % training.checkpoint_every == 0 or completed == training.steps or stop:
                    saved = output_dir / f"step_{completed:08d}{extension}"
                    save_checkpoint(saved, host_state(), experiment, stream.state_dict())
                    if primary:
                        atomic_json(output_dir / "latest.json", {"step": completed, "path": saved.name})
                if stop:
                    break
            if saved is None:
                saved = output_dir / f"step_{completed:08d}{extension}"
                if not saved.exists():
                    save_checkpoint(saved, host_state(), experiment, stream.state_dict())
            if primary:
                atomic_json(
                    output_dir / "status.json",
                    {
                        "status": "complete" if completed >= training.steps else "stopped",
                        "step": completed,
                        "checkpoint": str(saved),
                    },
                )
    except BaseException as error:
        if primary:
            atomic_json(
                output_dir / "status.json", {"status": "failed", "step": completed, "error_type": type(error).__name__}
            )
        raise
    finally:
        if log is not None:
            log.close()
    return saved


def main():
    parser = argparse.ArgumentParser(description="训练 TTT-E2E")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume")
    group.add_argument("--auto-resume", action="store_true")
    group.add_argument("--initialize-from")
    args = parser.parse_args()
    resume = latest_checkpoint(args.output) if args.auto_resume else args.resume
    checkpoint = train(
        load_experiment(args.config), args.data, args.output, resume=resume, initialize_from=args.initialize_from
    )
    if jax.process_index() == 0:
        print(json.dumps({"checkpoint": str(checkpoint)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
