"""提供按会话隔离的 token 推理接口。"""

import argparse
import hashlib
import hmac
import os
import re
import threading
import uuid
from pathlib import Path

import jax
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from .checkpoint import load_checkpoint
from .models.ttt_e2e import TTTE2EForCausalLM
from .serving import SessionBusyError, SessionCapacityError, SessionManager
from .session_store import load_session, save_session


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    token_ids: list[StrictInt] = Field(default_factory=list, max_length=131072)
    max_new_tokens: int = Field(default=32, ge=0, le=4096, strict=True)
    temperature: float = Field(default=0.0, ge=0)
    top_k: int = Field(default=0, ge=0, strict=True)
    seed: StrictInt | None = None


class RestoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot_id: str = Field(pattern=r"^[0-9a-f]{32}$")


def parameter_revision(params):
    digest = hashlib.sha256()
    for array in jax.tree.leaves(params):
        value = np.asarray(array)
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def create_app(manager, *, api_key=None, snapshot_dir=None, max_concurrent_requests=1):
    if max_concurrent_requests <= 0:
        raise ValueError("并发请求数必须为正。")
    capacity = threading.BoundedSemaphore(max_concurrent_requests)
    storage = Path(snapshot_dir).resolve() if snapshot_dir else None
    if storage:
        storage.mkdir(parents=True, exist_ok=True)

    def authenticate(authorization: str | None = Header(default=None)):
        if api_key and not hmac.compare_digest(authorization or "", f"Bearer {api_key}"):
            raise HTTPException(401, "认证失败。")

    app = FastAPI(title="TTT-E2E", dependencies=[Depends(authenticate)])

    def identity(value):
        if not re.fullmatch(r"[0-9a-f]{32}", value):
            raise HTTPException(404, "会话不存在。")
        return value

    def error_response(error):
        if isinstance(error, KeyError):
            return HTTPException(404, "会话或快照不存在。")
        if isinstance(error, SessionBusyError):
            return HTTPException(409, str(error))
        if isinstance(error, SessionCapacityError):
            return HTTPException(429, str(error))
        return HTTPException(400, str(error))

    @app.get("/health")
    def health():
        return {"status": "ready", **manager.statistics()}

    @app.post("/sessions", status_code=201)
    def create():
        try:
            return {"session_id": manager.create(), "model_revision": manager.model_revision}
        except SessionCapacityError as error:
            raise error_response(error) from error

    @app.delete("/sessions/{session_id}", status_code=204)
    def delete(session_id: str):
        try:
            manager.delete(identity(session_id))
        except (KeyError, SessionBusyError) as error:
            raise error_response(error) from error

    @app.post("/sessions/{session_id}/generate")
    def generate(session_id: str, request: GenerateRequest):
        if not capacity.acquire(blocking=False):
            raise HTTPException(429, "服务器正在处理其他请求。")
        try:
            with manager.acquire(identity(session_id)) as session:
                tokens = session.generate(**request.model_dump(), eos_token_id=manager.model.config.eos_token_id)
                # 等待设备完成，保证响应和失败回滚都发生在会话锁内。
                jax.block_until_ready((session.state.fast_weights, session.state.cache, session.state.last_logits))
                return {
                    "token_ids": tokens,
                    "position": session.state.position,
                    "updates": session.state.updates,
                    "state_bytes": session.state_bytes(),
                }
        except (KeyError, SessionBusyError, SessionCapacityError, ValueError) as error:
            raise error_response(error) from error
        finally:
            capacity.release()

    @app.post("/sessions/{session_id}/snapshot")
    def snapshot(session_id: str):
        if storage is None:
            raise HTTPException(404, "未启用会话快照。")
        try:
            with manager.acquire(identity(session_id)) as session:
                snapshot_id = uuid.uuid4().hex
                save_session(session, storage / snapshot_id, manager.model_revision)
                return {"snapshot_id": snapshot_id}
        except (KeyError, SessionBusyError) as error:
            raise error_response(error) from error

    @app.post("/sessions/{session_id}/restore")
    def restore(session_id: str, request: RestoreRequest):
        if storage is None:
            raise HTTPException(404, "未启用会话快照。")
        path = storage / request.snapshot_id
        if not path.is_dir():
            raise HTTPException(404, "快照不存在。")
        try:
            with manager.acquire(identity(session_id)) as session:
                load_session(session, path, manager.model_revision)
                return {"position": session.state.position, "updates": session.state.updates}
        except (KeyError, SessionBusyError, SessionCapacityError, ValueError) as error:
            raise error_response(error) from error

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="启动 TTT 会话服务")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-sessions", type=int, default=8)
    parser.add_argument("--state-budget-gib", type=float, default=8)
    parser.add_argument("--snapshot-dir")
    args = parser.parse_args()
    key = os.environ.get("TTT_API_KEY")
    if args.host not in ("127.0.0.1", "::1", "localhost") and not key:
        parser.error("非本机地址需要设置 TTT_API_KEY。")
    checkpoint = load_checkpoint(args.checkpoint)
    model = TTTE2EForCausalLM(checkpoint.experiment.model)
    manager = SessionManager(
        model,
        checkpoint.state.params,
        checkpoint.experiment.inner,
        model_revision=parameter_revision(checkpoint.state.params),
        max_sessions=args.max_sessions,
        max_state_bytes=int(args.state_budget_gib * 1024**3),
    )
    uvicorn.run(
        create_app(manager, api_key=key, snapshot_dir=args.snapshot_dir),
        host=args.host,
        port=args.port,
        workers=1,
        access_log=False,
    )


if __name__ == "__main__":
    main()
