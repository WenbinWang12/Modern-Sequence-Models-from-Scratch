"""在 PBS 分配的节点上启动 JAX 多机训练。"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--module", choices=("ttt.train", "ttt.validate_mesh"), default="ttt.train")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    node_file = os.environ.get("PBS_NODEFILE")
    if not node_file:
        raise RuntimeError("该启动器必须在 PBS 作业中运行。")
    slots = Path(node_file).read_text().splitlines()
    hosts = list(dict.fromkeys(slots))
    if not hosts:
        raise RuntimeError("PBS 节点清单为空。")
    port = args.port or 20000 + int(os.environ["PBS_JOBID"].split(".")[0]) % 20000
    coordinator = f"{hosts[0]}:{port}"
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    processes = []
    for rank, host in enumerate(hosts):
        command = [
            "env",
            f"JAX_COORDINATOR_ADDRESS={coordinator}",
            f"JAX_PROCESS_COUNT={len(hosts)}",
            f"JAX_PROCESS_ID={rank}",
            sys.executable,
            "-m",
            args.module,
            *arguments,
        ]
        script = f"cd -- {shlex.quote(str(Path.cwd()))} && exec {shlex.join(command)}"
        processes.append(subprocess.Popen(["/opt/pbs/bin/pbsdsh", "-n", str(slots.index(host)), "bash", "-c", script]))
    try:
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


if __name__ == "__main__":
    main()
