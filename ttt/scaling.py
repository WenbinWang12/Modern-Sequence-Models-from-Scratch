"""拟合参数量和训练量的 scaling law，并检验留出规模。"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def fit_scaling(rows, *, bootstrap=100, seed=0):
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 0:
        raise ValueError("bootstrap 必须为非负整数。")
    if any(
        not np.isfinite([row["parameters"], row["tokens"], row["loss"]]).all()
        or min(row["parameters"], row["tokens"], row["loss"]) <= 0
        for row in rows
    ):
        raise ValueError("参数、tokens 和 loss 必须为有限正数。")
    training = [row for row in rows if not row.get("held_out", False)]
    held_out = [row for row in rows if row.get("held_out", False)]
    if (
        len(training) < 8
        or len({row["parameters"] for row in training}) < 3
        or len({row["tokens"] for row in training}) < 3
    ):
        raise ValueError("拟合至少需要八条训练结果，并交叉覆盖三个模型规模与三个训练量。")
    data = np.asarray([[row["parameters"], row["tokens"], row["loss"]] for row in training], dtype=np.float64)
    if not np.isfinite(data).all() or (data <= 0).any():
        raise ValueError("参数、tokens 和 loss 必须为有限正数。")
    x, y = data[:, :2] / np.asarray([1e6, 1e9]), data[:, 2]

    def predict(parameters, values):
        floor, log_a, log_b, alpha, beta = parameters
        return floor + np.exp(log_a) * values[:, 0] ** (-alpha) + np.exp(log_b) * values[:, 1] ** (-beta)

    def fit(indices):
        result = least_squares(
            lambda p: predict(p, x[indices]) - y[indices],
            [min(y) * 0.5, 0, 0, 0.3, 0.3],
            bounds=([0, -20, -20, 0.01, 0.01], [min(y), 20, 20, 1.0, 1.0]),
            max_nfev=5000,
        )
        if not result.success:
            raise RuntimeError("scaling 拟合未收敛。")
        return result.x

    parameters = fit(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    samples = [fit(rng.integers(0, len(y), len(y))) for _ in range(bootstrap)]
    checks = []
    for row in held_out:
        values = np.asarray([[row["parameters"] / 1e6, row["tokens"] / 1e9]])
        predictions = np.asarray([predict(p, values)[0] for p in samples])
        interval = np.quantile(predictions, [0.025, 0.975]).tolist() if samples else None
        estimate = float(predict(parameters, values)[0])
        checks.append(
            {
                **row,
                "predicted_loss": estimate,
                "interval_95": interval,
                "inside_interval": None if interval is None else interval[0] <= row["loss"] <= interval[1],
            }
        )
    return {
        "formula": "E + A*(N/1e6)^(-alpha) + B*(D/1e9)^(-beta)",
        "E": float(parameters[0]),
        "A": float(np.exp(parameters[1])),
        "B": float(np.exp(parameters[2])),
        "alpha": float(parameters[3]),
        "beta": float(parameters[4]),
        "training_rmse": float(np.sqrt(np.mean((predict(parameters, x) - y) ** 2))),
        "held_out": checks,
        "bootstrap_samples": bootstrap,
    }


def main():
    parser = argparse.ArgumentParser(description="拟合 scaling law")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=100)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    result = fit_scaling(rows, bootstrap=args.bootstrap)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
