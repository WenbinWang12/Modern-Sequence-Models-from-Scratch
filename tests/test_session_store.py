"""会话恢复必须保留快权重、缓存、标签边界和随机流。"""

import jax
import numpy as np
import pytest

from ttt.inner_loop import InnerLoopConfig
from ttt.runtime import TTTSession
from ttt.serving import SessionBusyError, SessionCapacityError, SessionManager
from ttt.session_store import load_session, save_session


@pytest.mark.parametrize("position", [0, 2, 5, 9])
def test_session_roundtrip(model_and_params, tmp_path, position):
    pytest.importorskip("orbax.checkpoint")
    model, params = model_and_params
    original = TTTSession(model, params, InnerLoopConfig(learning_rate=0.1), jit=False)
    original.observe(np.arange(1, position + 1, dtype=np.int32))
    original.state.rng_key = jax.random.PRNGKey(71)
    save_session(original, tmp_path / "session", "weights-sha")
    restored = TTTSession(model, params, original.inner, jit=False)
    load_session(restored, tmp_path / "session", "weights-sha")
    assert restored.state.position == position
    assert restored.state.updates == original.state.updates
    assert restored.state.pending_targets == original.state.pending_targets
    for a, b in zip(jax.tree.leaves(restored.state.cache), jax.tree.leaves(original.state.cache)):
        np.testing.assert_array_equal(a, b)
    assert restored.generate([10], max_new_tokens=3, temperature=0.7) == original.generate(
        [10], max_new_tokens=3, temperature=0.7
    )
    with pytest.raises(ValueError, match="版本"):
        load_session(restored, tmp_path / "session", "other-weights")


def test_session_admission_lock_and_rollback(model_and_params):
    model, params = model_and_params
    manager = SessionManager(model, params, InnerLoopConfig(), model_revision="v1", max_sessions=1, jit=False)
    identity = manager.create()
    with pytest.raises(SessionCapacityError):
        manager.create()
    with manager.acquire(identity) as session:
        with pytest.raises(SessionBusyError):
            with manager.acquire(identity):
                pass
        session.decode(1)
    with pytest.raises(RuntimeError):
        with manager.acquire(identity) as session:
            session.decode(2)
            raise RuntimeError("模拟请求失败")
    with manager.acquire(identity) as session:
        assert session.state.position == 1
        np.testing.assert_array_equal(session.state.rng_key, jax.random.PRNGKey(0))
    manager.delete(identity)
    tiny = SessionManager(model, params, InnerLoopConfig(), model_revision="v1", max_state_bytes=1, jit=False)
    with pytest.raises(SessionCapacityError):
        tiny.create()


def test_http_session_flow(model_and_params, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    pytest.importorskip("orbax.checkpoint")
    from fastapi.testclient import TestClient

    from ttt.serve import create_app

    model, params = model_and_params
    manager = SessionManager(model, params, InnerLoopConfig(), model_revision="v1", jit=False)
    with TestClient(create_app(manager, api_key="test-key", snapshot_dir=tmp_path)) as client:
        assert client.get("/health").status_code == 401
        client.headers["Authorization"] = "Bearer test-key"
        session_id = client.post("/sessions").json()["session_id"]
        prefix = f"/sessions/{session_id}"
        assert client.post(prefix + "/generate", json={"token_ids": [1, 2], "max_new_tokens": 2}).status_code == 200
        snapshot = client.post(prefix + "/snapshot").json()["snapshot_id"]
        assert client.post(prefix + "/generate", json={"token_ids": [-1]}).status_code == 400
        assert client.post(prefix + "/generate", json={"token_ids": [True]}).status_code == 422
        assert client.post(prefix + "/restore", json={"snapshot_id": snapshot}).json()["position"] == 4
        assert client.delete(prefix).status_code == 204
        assert client.post(prefix + "/generate", json={}).status_code == 404
