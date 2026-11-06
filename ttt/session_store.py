"""保存与恢复会话快权重、缓存和更新边界。"""

import json
import os
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .lifecycle import atomic_json


def save_session(session, path, model_revision):
    import orbax.checkpoint as ocp

    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"会话快照已存在：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    state = session.state
    arrays = {}
    counts = {}
    for prefix, tree in (("fast", state.fast_weights), ("cache", state.cache), ("chunk", state.chunk_cache)):
        leaves = jax.tree.leaves(tree)
        counts[prefix] = len(leaves)
        arrays.update({f"{prefix}_{index}": value for index, value in enumerate(leaves)})
    arrays["features"] = state.pending_features
    arrays["rng"] = state.rng_key
    if state.last_logits is not None:
        arrays["logits"] = state.last_logits
    metadata = {
        "format_version": 2,
        "model_revision": model_revision,
        "config": session.model.config.to_dict(),
        "inner": asdict(session.inner),
        "counts": counts,
        "position": state.position,
        "pending_count": state.pending_count,
        "targets": list(state.pending_targets),
        "chunk_start": state.chunk_start,
        "updates": state.updates,
    }
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(temporary / "arrays", arrays)
        checkpointer.wait_until_finished()
    atomic_json(temporary / "metadata.json", metadata)
    os.replace(temporary, path)


def load_session(session, path, model_revision):
    import orbax.checkpoint as ocp

    path = Path(path).resolve()
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata["format_version"] != 2 or metadata["model_revision"] != model_revision:
        raise ValueError("会话格式或模型版本不匹配。")
    if metadata["config"] != session.model.config.to_dict() or metadata["inner"] != asdict(session.inner):
        raise ValueError("会话配置或更新策略不匹配。")
    position, pending, start = metadata["position"], metadata["pending_count"], metadata["chunk_start"]
    if not 0 <= position <= session.model.config.max_position_embeddings:
        raise ValueError("会话位置越界。")
    chunk_size = session.model.config.ttt_chunk_size
    if (
        not 0 <= pending <= chunk_size
        or start < 0
        or start % chunk_size
        or start + pending != position
        or (position > 0 and pending == 0)
    ):
        raise ValueError("会话更新边界不一致。")
    if len(metadata["targets"]) != max(pending - 1, 0):
        raise ValueError("会话待更新标签数量不一致。")
    if any(
        isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < session.model.config.vocab_size
        for token in metadata["targets"]
    ):
        raise ValueError("会话标签超出词表。")
    expected_updates = start // chunk_size if session.inner.enabled and session.state.fast_weights else 0
    if metadata["updates"] != expected_updates:
        raise ValueError("会话更新次数与位置不一致。")
    trees = {
        "fast": session.model.initial_fast_weights(session.params),
        "cache": session.model.init_cache(range(session.model.config.num_hidden_layers), 1),
        "chunk": session.model.init_cache(
            range(session.model.config.prefix_length, session.model.config.num_hidden_layers), 1
        ),
    }
    sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
    with ocp.StandardCheckpointer() as checkpointer:
        description = checkpointer.metadata(path / "arrays")
        target = {}
        for prefix, tree in trees.items():
            values = jax.tree.flatten_with_path(tree)[0]
            if len(values) != metadata["counts"][prefix]:
                raise ValueError("会话参数结构不一致。")
            for index, (key_path, template) in enumerate(values):
                name = f"{prefix}_{index}"
                item = description[name]
                dynamic = prefix in ("cache", "chunk") and getattr(key_path[-1], "name", None) in (
                    "key",
                    "value",
                    "positions",
                )
                if dynamic:
                    valid = (
                        item.shape[1:] == template.shape[1:]
                        and 1 <= item.shape[0] <= session.model.config.max_position_embeddings
                    )
                else:
                    valid = item.shape == template.shape
                if not valid or item.dtype != template.dtype:
                    raise ValueError(f"会话数组 {name} 的尺寸或精度不匹配。")
                target[name] = jax.ShapeDtypeStruct(item.shape, item.dtype, sharding=sharding)
        for name, shape, dtype in (
            ("features", session.state.pending_features.shape, session.state.pending_features.dtype),
            ("rng", (2,), jnp.dtype("uint32")),
        ):
            if description[name].shape != shape or description[name].dtype != dtype:
                raise ValueError(f"会话数组 {name} 不匹配。")
            target[name] = jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
        if "logits" in description:
            if (
                description["logits"].shape != (session.model.config.vocab_size,)
                or description["logits"].dtype != jnp.float32
            ):
                raise ValueError("会话输出头维度或精度不匹配。")
            target["logits"] = jax.ShapeDtypeStruct((session.model.config.vocab_size,), jnp.float32, sharding=sharding)
        if ("logits" in description) != (position > 0):
            raise ValueError("会话位置与输出状态不一致。")
        arrays = checkpointer.restore(path / "arrays", target=target)
    state = replace(session.state)
    for prefix, tree in trees.items():
        restored = jax.tree.unflatten(
            jax.tree.structure(tree), [arrays[f"{prefix}_{index}"] for index in range(metadata["counts"][prefix])]
        )
        if prefix in ("cache", "chunk"):
            upper = position if prefix == "cache" else start
            for key_path, value in jax.tree.flatten_with_path(restored)[0]:
                if getattr(key_path[-1], "name", None) == "positions":
                    positions = np.asarray(value)
                    if np.any(positions < -1) or np.any(positions >= upper):
                        raise ValueError("会话缓存包含未来位置。")
        setattr(
            state, "fast_weights" if prefix == "fast" else "chunk_cache" if prefix == "chunk" else "cache", restored
        )
    state.position, state.pending_count, state.chunk_start = position, pending, start
    state.pending_targets = tuple(metadata["targets"])
    state.updates = metadata["updates"]
    state.pending_features, state.rng_key = arrays["features"], arrays["rng"]
    state.last_logits = arrays.get("logits")
    session.state = state
