"""带容量限制和会话隔离的推理管理器。"""

import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

import jax

from .runtime import TTTSession


class SessionCapacityError(RuntimeError):
    pass


class SessionBusyError(RuntimeError):
    pass


@dataclass
class SessionEntry:
    session: TTTSession
    touched: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionManager:
    def __init__(
        self,
        model,
        params,
        inner,
        *,
        model_revision,
        max_sessions=8,
        max_state_bytes=8 * 1024**3,
        ttl_seconds=3600,
        jit=True,
    ):
        if max_sessions <= 0 or max_state_bytes <= 0 or ttl_seconds <= 0:
            raise ValueError("会话数、状态容量和超时必须为正。")
        self.model, self.params, self.inner = model, params, inner
        self.model_revision = model_revision
        self.max_sessions, self.max_state_bytes, self.ttl_seconds = max_sessions, max_state_bytes, ttl_seconds
        self.jit = jit
        self.entries = {}
        self.lock = threading.RLock()
        # 预留最大上下文的持久状态；训练梯度和编译临时内存另外计量。
        cache = jax.eval_shape(
            lambda: model.init_cache(range(model.config.num_hidden_layers), model.config.max_position_embeddings)
        )
        fast = model.initial_fast_weights(params)
        self.reservation_bytes = (
            sum(value.size * value.dtype.itemsize for value in jax.tree.leaves(cache)) * 2
            + sum(value.size * value.dtype.itemsize for value in jax.tree.leaves(fast)) * 2
            + model.config.ttt_chunk_size * model.config.hidden_size * 4
            + model.config.vocab_size * 4
            + model.config.ttt_chunk_size * 4
            + 8
        )

    def _expire(self):
        now = time.monotonic()
        for identity, entry in list(self.entries.items()):
            if now - entry.touched > self.ttl_seconds and entry.lock.acquire(blocking=False):
                del self.entries[identity]
                entry.lock.release()

    def create(self):
        with self.lock:
            self._expire()
            if len(self.entries) >= self.max_sessions:
                raise SessionCapacityError("已达到最大会话数。")
            if (len(self.entries) + 1) * self.reservation_bytes > self.max_state_bytes:
                raise SessionCapacityError("会话状态预算不足。")
            session = TTTSession(self.model, self.params, self.inner, jit=self.jit)
            identity = uuid.uuid4().hex
            self.entries[identity] = SessionEntry(session)
            return identity

    @contextmanager
    def acquire(self, identity):
        with self.lock:
            self._expire()
            entry = self.entries[identity]
            if not entry.lock.acquire(blocking=False):
                raise SessionBusyError("该会话正在处理另一个请求。")
            original = replace(entry.session.state)
        try:
            yield entry.session
            used = sum(entry.session.state_bytes().values())
            if used > self.reservation_bytes:
                raise SessionCapacityError("请求超出会话状态预算。")
        except BaseException:
            entry.session.state = original
            raise
        finally:
            entry.touched = time.monotonic()
            entry.lock.release()

    def delete(self, identity):
        with self.lock:
            entry = self.entries[identity]
            if not entry.lock.acquire(blocking=False):
                raise SessionBusyError("不能删除正在处理请求的会话。")
            del self.entries[identity]
            entry.lock.release()

    def statistics(self):
        with self.lock:
            return {
                "sessions": len(self.entries),
                "state_bytes": sum(sum(entry.session.state_bytes().values()) for entry in self.entries.values()),
                "reserved_bytes": len(self.entries) * self.reservation_bytes,
                "max_sessions": self.max_sessions,
                "max_state_bytes": self.max_state_bytes,
                "model_revision": self.model_revision,
            }
