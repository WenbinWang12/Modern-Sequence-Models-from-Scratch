"""在单个 PBS 节点验证两个 JAX 进程。"""

import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    platform = os.environ.get("TTT_PROCESS_PLATFORM", "cuda")
    if platform not in ("cuda", "cpu"):
        raise ValueError("平台必须为 cuda 或 cpu。")
    identity = os.environ["PBS_JOBID"]
    output = Path("results") / ("process-" + identity)
    output.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "JAX_PLATFORMS": platform,
        "JAX_CPU_COLLECTIVES_IMPLEMENTATION": "gloo",
        "JAX_COORDINATOR_ADDRESS": f"127.0.0.1:{20000 + int(identity.split('.')[0]) % 20000}",
        "JAX_PROCESS_COUNT": "2",
        "OMP_NUM_THREADS": "2",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        environment.pop(name, None)
        environment.pop(name.lower(), None)
    processes, logs = [], []
    try:
        for rank in range(2):
            handle = (output / f"rank-{rank}.log").open("w")
            logs.append(handle)
            command = [
                sys.executable,
                "-m",
                "ttt.validate_mesh",
                "--output",
                f"results/process-{platform}-validation.json",
                "--checkpoint-dir",
                str(output),
            ]
            if platform == "cuda":
                command.append("--require-gpu")
            process_environment = {**environment, "JAX_PROCESS_ID": str(rank)}
            if platform == "cuda":
                process_environment["JAX_LOCAL_DEVICE_IDS"] = str(rank)
            processes.append(
                subprocess.Popen(command, env=process_environment, stdout=handle, stderr=subprocess.STDOUT)
            )
        while any(process.poll() is None for process in processes):
            failed = next((process.returncode for process in processes if process.returncode), None)
            if failed:
                raise SystemExit(failed)
            time.sleep(0.2)
        if any(process.returncode for process in processes):
            raise SystemExit(next(process.returncode for process in processes if process.returncode))
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        for handle in logs:
            handle.close()


if __name__ == "__main__":
    main()
