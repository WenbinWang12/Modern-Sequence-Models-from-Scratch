"""记录代码版本和运行环境。"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
from pathlib import Path

import jax


def source_digest(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    if not (root / "ttt").is_dir():
        raise ValueError("代码目录中缺少 ttt 包。")
    digest = hashlib.sha256()
    paths = sorted((root / "ttt").rglob("*.py"))
    # wheel 安装没有项目根文件，仍可根据包内代码记录版本。
    paths += [root / name for name in ("pyproject.toml", "uv.lock") if (root / name).is_file()]
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment_report():
    versions = {}
    for name in ("jax", "jaxlib", "optax", "numpy", "orbax-checkpoint", "fastapi", "tensorstore"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        driver = None
    return {
        "python": platform.python_version(),
        "system": platform.platform(),
        "hostname": socket.gethostname(),
        "packages": versions,
        "process_count": jax.process_count(),
        "device_count": jax.device_count(),
        "devices": [str(device) for device in jax.local_devices()],
        "nvidia_smi": driver,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "source_sha256": source_digest(),
    }


def main():
    """把当前运行环境写入 JSON 文件。"""
    parser = argparse.ArgumentParser(description="记录 TTT 运行环境")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(environment_report(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "source_sha256": source_digest()}))


if __name__ == "__main__":
    main()
