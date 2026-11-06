"""训练中断与运行状态。"""

import json
import os
import signal
import tempfile
from pathlib import Path


class StopSignal:
    def __init__(self):
        self.requested = False
        self.previous = {}

    def __enter__(self):
        def request(signum, frame):
            self.requested = True

        for signum in (signal.SIGINT, signal.SIGTERM):
            self.previous[signum] = signal.signal(signum, request)
        return self

    def __exit__(self, *args):
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def latest_checkpoint(directory):
    candidates = []
    for path in Path(directory).glob("step_*"):
        if path.suffix == ".npz" or (path.is_dir() and (path / "metadata.json").exists()):
            try:
                step = int(path.stem.split("_")[-1])
            except ValueError:
                continue
            candidates.append((step, path))
    return max(candidates, default=(None, None))[1]
