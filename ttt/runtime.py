"""单会话推理与测试时更新。"""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .inner_loop import InnerLoopConfig
from .runtime_kernels import runtime_kernels


@dataclass
class SessionState:
    fast_weights: tuple
    cache: tuple
    position: int
    pending_features: jax.Array
    pending_targets: tuple
    chunk_cache: tuple
    chunk_start: int
    pending_count: int = 0
    last_logits: jax.Array | None = None
    updates: int = 0
    rng_key: jax.Array | None = None


class TTTSession:
    def __init__(self, model, params, inner=None, *, jit=True):
        self.model = model
        self.params = params
        self.inner = inner if inner is not None else InnerLoopConfig()
        kernels = runtime_kernels(model.config, self.inner, jit, model.gradient_checkpointing)
        self._prefix, self._suffix, self._suffix_hidden, self._project_logits, self._update = kernels
        self.reset()

    def reset(self):
        config = self.model.config
        capacity = min(config.ttt_chunk_size, config.max_position_embeddings)
        cache = self.model.init_cache(range(config.num_hidden_layers), capacity)
        self.state = SessionState(
            fast_weights=self.model.initial_fast_weights(self.params),
            cache=cache,
            position=0,
            pending_features=jnp.zeros((config.ttt_chunk_size, config.hidden_size), dtype=config.compute_dtype),
            pending_targets=(),
            chunk_cache=cache[config.prefix_length :],
            chunk_start=0,
            rng_key=jax.random.PRNGKey(0),
        )

    def _start_chunk(self):
        state, config = self.state, self.model.config
        required = min(state.position + config.ttt_chunk_size, config.max_position_embeddings)
        capacity = min(1 << max(0, required - 1).bit_length(), config.max_position_embeddings)
        state.cache = tuple(
            self.model.grow_layer_cache(index, cache, capacity) for index, cache in enumerate(state.cache)
        )
        state.pending_count = 0
        state.pending_targets = ()
        state.chunk_start = state.position
        state.chunk_cache = state.cache[config.prefix_length :]

    def _finish_chunk(self, observed_next_token):
        state = self.state
        if self.inner.enabled and state.fast_weights:
            targets = jnp.asarray((*state.pending_targets, observed_next_token), dtype=jnp.int32)
            state.fast_weights = self._update(
                self.params,
                state.fast_weights,
                state.pending_features,
                targets,
                state.chunk_cache,
                jnp.asarray(state.chunk_start, dtype=jnp.int32),
            )
            state.updates += 1
        # 沿用此前前向得到的 KV，不用新快权重重算历史。
        self._start_chunk()

    def observe(self, token_ids, *, return_logits=True):
        ids = np.asarray(token_ids)
        if ids.ndim != 1 or (ids.size and not np.issubdtype(ids.dtype, np.integer)):
            raise ValueError("token_ids 必须是一维整数序列。")
        config = self.model.config
        if ids.size and (ids.min() < 0 or ids.max() >= config.vocab_size):
            raise ValueError("token ID 超出词表范围。")
        if self.state.position + ids.size > config.max_position_embeddings:
            raise ValueError("会话超过 max_position_embeddings。")
        if not ids.size:
            return jnp.empty((0, config.vocab_size), dtype=jnp.float32)
        outputs = []
        offset = 0
        while offset < ids.size:
            state = self.state
            pending = state.pending_count
            if pending == config.ttt_chunk_size:
                # 只有下一 token 已经被观测到，才能补齐最后一个训练标签。
                self._finish_chunk(int(ids[offset]))
                pending = 0
            count = min(config.ttt_chunk_size - pending, ids.size - offset)
            current = ids[offset : offset + count].astype(np.int32)
            split = config.prefix_length
            hidden, prefix_cache = self._prefix(
                self.params,
                jnp.asarray(current),
                state.cache[:split],
                jnp.asarray(state.position),
            )
            if return_logits:
                output = self._suffix(
                    self.params,
                    hidden,
                    state.fast_weights,
                    state.cache[split:],
                    jnp.asarray(state.position),
                )
                logits, suffix_cache = output.logits, output.cache
            else:
                features, suffix_cache = self._suffix_hidden(
                    self.params,
                    hidden,
                    state.fast_weights,
                    state.cache[split:],
                    jnp.asarray(state.position),
                )
                logits = self._project_logits(self.params, features[-1:])
            state.cache = prefix_cache + suffix_cache
            state.pending_features = jax.lax.dynamic_update_slice(
                state.pending_features,
                hidden,
                (jnp.asarray(pending, dtype=jnp.int32), 0),
            )
            state.pending_count += count
            labels = current if pending else current[1:]
            state.pending_targets += tuple(int(token) for token in labels)
            state.position += count
            state.last_logits = logits[-1]
            if return_logits:
                outputs.append(logits)
            offset += count
        return jnp.concatenate(outputs, axis=0) if return_logits else self.state.last_logits[None]

    def prefill(self, token_ids, *, return_logits=False):
        return self.observe(token_ids, return_logits=return_logits)

    def decode(self, token_id):
        return self.observe(np.asarray([token_id], dtype=np.int32))[0]

    def generate(self, token_ids, *, max_new_tokens=32, temperature=0.0, top_k=0, seed=None, eos_token_id=None):
        if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens 必须是非负整数。")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature 必须是有限的非负数。")
        if not isinstance(top_k, int) or not 0 <= top_k <= self.model.config.vocab_size:
            raise ValueError("top_k 必须位于 [0, vocab_size]。")
        if self.state.position + len(token_ids) + max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError("提示词与生成长度超过 max_position_embeddings。")
        if len(token_ids):
            self.observe(token_ids, return_logits=False)
        if self.state.last_logits is None:
            raise ValueError("首次生成必须提供至少一个提示 token。")
        key = jax.random.PRNGKey(seed) if seed is not None else self.state.rng_key
        generated = []
        for _ in range(max_new_tokens):
            logits = self.state.last_logits
            if temperature == 0:
                token = int(jnp.argmax(logits))
            else:
                if top_k:
                    threshold = jax.lax.top_k(logits, top_k)[0][-1]
                    logits = jnp.where(logits >= threshold, logits, -jnp.inf)
                key, sample_key = jax.random.split(key)
                token = int(jax.random.categorical(sample_key, logits / temperature))
            generated.append(token)
            self.state.rng_key = key
            self.decode(token)
            if eos_token_id is not None and token == eos_token_id:
                break
        return generated

    def state_bytes(self):
        seen = set()

        def size(tree):
            total = 0
            for array in jax.tree.leaves(tree):
                if id(array) not in seen:
                    total += array.size * array.dtype.itemsize
                    seen.add(id(array))
            return total

        return {
            "fast_weights": size(self.state.fast_weights),
            "kv_cache": size(self.state.cache),
            "chunk_cache": size(self.state.chunk_cache),
            "pending_chunk": size(self.state.pending_features) + 4 * len(self.state.pending_targets),
            "last_logits": size(self.state.last_logits),
        }
