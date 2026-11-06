"""跨进程广播的 dtype 提升不能污染检查点路径。"""

import jax
import numpy as np
from jax.experimental import multihost_utils

from ttt.checkpoint import _checkpoint_identity


def test_broadcast_identity_survives_integer_promotion(monkeypatch):
    expected = "0123456789abcdef0123456789abcdef"
    monkeypatch.setattr(jax, "process_count", lambda: 2)
    monkeypatch.setattr(
        multihost_utils,
        "broadcast_one_to_all",
        lambda value: np.frombuffer(expected.encode("ascii"), dtype=np.uint8).astype(np.int32),
    )
    assert _checkpoint_identity() == expected
