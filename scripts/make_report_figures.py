#!/usr/bin/env python3
"""从原始训练指标和评测结果生成技术报告的图表与汇总数字。

输入全部是训练/评测脚本写出的原始文件（``metrics.jsonl``、``outputs/evals/*.json|npz``、
``outputs/bench/*.json``），不手工抄写任何数字。输出：

- ``<output>/fig_*.{light,dark}.svg`` 和 ``fig_*.png``：图表（浅色/深色两套）；
- ``<output>/results.json``：报告正文表格使用的全部数字，含按文档重采样的 95% 置信区间。

颜色按“实体”固定分配（同一个模型在所有图里颜色相同），取自一组通过色觉障碍可分性检查的
分类色；两条及以上曲线必有图例，线尾另有直接标注，文字一律用墨色而不用数据色。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
    },
}
# 实体 -> 固定色位（顺序即分类色的固定顺序，不随图变化）。
ENTITIES = ["full", "swa", "ttt", "ck", "ttt_noadapt"]
LABELS = {
    "full": "Full attention",
    "swa": "Sliding window",
    "ttt": "TTT-E2E",
    "ck": "No-update control",
    "ttt_noadapt": "TTT-E2E, updates off",
    "full_sw4096": "Full attention, 4K window",
}


def color(theme, entity):
    return theme["series"][ENTITIES.index("full" if entity == "full_sw4096" else entity)]


def new_figure(theme, columns, *, width, height):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.fonttype": "path",
            "axes.unicode_minus": False,
        }
    )
    figure, axes = plt.subplots(1, columns, figsize=(width, height), squeeze=False)
    figure.patch.set_facecolor(theme["surface"])
    for axis in axes[0]:
        axis.set_facecolor(theme["surface"])
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(theme["axis"])
            axis.spines[side].set_linewidth(0.8)
        axis.tick_params(colors=theme["axis"], labelcolor=theme["ink2"], length=3, width=0.8, labelsize=8.5)
        axis.grid(True, axis="y", color=theme["grid"], linewidth=0.8, linestyle="-")
        axis.set_axisbelow(True)
    return figure, list(axes[0])


def titled(axis, theme, title, xlabel, ylabel):
    axis.set_title(title, loc="left", color=theme["ink"], fontsize=10, fontweight="semibold", pad=8)
    axis.set_xlabel(xlabel, color=theme["ink2"], fontsize=9)
    axis.set_ylabel(ylabel, color=theme["ink2"], fontsize=9)


def figure_legend(figure, theme, entities, *, left=0.06, top=0.985):
    handles = [Line2D([0], [0], color=color(theme, entity), linewidth=2) for entity in entities]
    figure.legend(
        handles,
        [LABELS[entity] for entity in entities],
        loc="upper left",
        bbox_to_anchor=(left, top),
        ncol=len(entities),
        frameon=False,
        fontsize=8.5,
        handlelength=1.6,
        columnspacing=1.6,
        labelcolor=theme["ink2"],
    )


def label_line_ends(axis, theme, entries, *, gap_points=11.5):
    """在线尾放直接标注；标注互相挤压时沿纵向推开，并用细引线连回线尾。"""
    figure = axis.figure
    figure.canvas.draw()
    scale = figure.dpi / 72
    rows = sorted((axis.transData.transform((x, y))[1], x, y, text, tint) for x, y, text, tint in entries)
    placed = [row[0] for row in rows]
    for index in range(1, len(placed)):
        placed[index] = max(placed[index], placed[index - 1] + gap_points * scale)
    shift = sum(row[0] - value for row, value in zip(rows, placed)) / len(rows)
    for (pixel, x, y, text, tint), value in zip(rows, placed):
        offset = (value + shift - pixel) / scale
        axis.plot(
            [x],
            [y],
            marker="o",
            markersize=6.5,
            color=tint,
            markeredgecolor=theme["surface"],
            markeredgewidth=1.5,
            zorder=6,
            clip_on=False,
        )
        axis.annotate(
            text,
            xy=(x, y),
            xytext=(9, offset),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8.5,
            color=theme["ink2"],
            annotation_clip=False,
            arrowprops={"arrowstyle": "-", "color": theme["axis"], "linewidth": 0.8, "shrinkA": 0, "shrinkB": 4}
            if abs(offset) > 2
            else None,
        )


def save(figure, output, name, mode):
    figure.savefig(output / f"{name}.{mode}.svg", facecolor=figure.get_facecolor())
    if mode == "light":
        figure.savefig(output / f"{name}.png", dpi=200, facecolor=figure.get_facecolor())
    plt.close(figure)


def context_ticks(axis, lengths):
    axis.set_xscale("log", base=2)
    axis.set_xticks(lengths)
    axis.set_xticklabels([f"{length // 1024}K" if length >= 1024 else str(length) for length in lengths])
    axis.minorticks_off()


# --------------------------------------------------------------------------- 读取数据


def read_metrics(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # 续训会重复写入同一步，保留最后一次。
    by_step = {row["step"]: row for row in rows}
    return [by_step[step] for step in sorted(by_step)]


def ema(values, alpha=0.1):
    result, state = [], None
    for value in values:
        state = value if state is None else (1 - alpha) * state + alpha * value
        result.append(state)
    return np.asarray(result)


def load_eval(directory, name):
    path = directory / f"{name}.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    arrays = np.load(path.with_suffix(".npz"))
    return {
        "summary": summary,
        "loss": arrays["loss_sums"].astype(np.float64),
        "count": arrays["counts"].astype(np.float64),
        "document": arrays["document"],
        "position": arrays["position"],
    }


def cluster_sums(record):
    """把窗口级的 loss 和计数合并到文档（单文档的连续流则每个窗口自成一簇）。"""
    documents = record["document"]
    if len(np.unique(documents)) == 1:
        return record["loss"], record["count"]
    order = np.unique(documents)
    loss = np.stack([record["loss"][documents == document].sum(axis=0) for document in order])
    count = np.stack([record["count"][documents == document].sum(axis=0) for document in order])
    return loss, count


def bootstrap_difference(first, second, *, bins=None, repetitions=2000, seed=0):
    """first - second 的平均 NLL 差及 95% 区间；两者必须评测同一批窗口。"""
    if first["loss"].shape != second["loss"].shape or not np.array_equal(first["position"], second["position"]):
        raise ValueError("配对比较要求两次评测使用完全相同的窗口。")
    loss_a, count = cluster_sums(first)
    loss_b, _ = cluster_sums(second)
    if bins is not None:
        loss_a, loss_b, count = loss_a[:, bins], loss_b[:, bins], count[:, bins]
    difference, weight = (loss_a - loss_b).sum(axis=1), count.sum(axis=1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(weight), size=(repetitions, len(weight)))
    samples = difference[draws].sum(axis=1) / weight[draws].sum(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "mean": float(difference.sum() / weight.sum()),
        "ci95": [float(low), float(high)],
        "clusters": int(len(weight)),
    }


def bootstrap_curve(first, second, *, repetitions=1000, seed=0):
    loss_a, count = cluster_sums(first)
    loss_b, _ = cluster_sums(second)
    difference = loss_a - loss_b
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(count), size=(repetitions, len(count)))
    samples = difference[draws].sum(axis=1) / np.maximum(count[draws].sum(axis=1), 1)
    mean = difference.sum(axis=0) / np.maximum(count.sum(axis=0), 1)
    low, high = np.quantile(samples, [0.025, 0.975], axis=0)
    return mean, low, high


def by_position(record):
    return record["loss"].sum(axis=0) / np.maximum(record["count"].sum(axis=0), 1)


def scalar(record):
    summary = record["summary"]
    keys = (
        "nll",
        "perplexity",
        "bits_per_byte",
        "windows",
        "documents",
        "tokens",
        "checkpoint_step",
        "eval_tokens_per_second",
    )
    return {key: summary.get(key) for key in keys}


# --------------------------------------------------------------------------- 各图


def figure_training(theme, mode, runs, evals, output, results):
    metrics = {
        arm: read_metrics(runs / f"main-{arm}" / "metrics.jsonl")
        for arm in ("full", "swa", "ck", "ttt")
        if (runs / f"main-{arm}" / "metrics.jsonl").exists()
    }
    if not metrics:
        return
    order = [arm for arm in ENTITIES if arm in metrics]
    curves = {}
    for arm in ("full", "swa", "ck", "ttt"):
        points = []
        for path in sorted(evals.glob(f"curve_{arm}_s*__pg19val.json")):
            summary = json.loads(path.read_text(encoding="utf-8"))
            points.append((summary["checkpoint_step"], summary["nll"]))
        if points:
            curves[arm] = sorted(points)
    reference = "full" if "full" in metrics else "swa"
    figure, axes = new_figure(theme, 3 if curves else 2, width=11.2 if curves else 8.0, height=3.5)
    tokens_per_step = metrics[order[0]][0]["tokens_seen"] / metrics[order[0]][0]["step"]
    smooth = {
        arm: (np.asarray([row["step"] for row in rows]), ema([row["nll"] for row in rows]))
        for arm, rows in metrics.items()
    }
    ends = []
    for arm in order:
        steps, values = smooth[arm]
        axes[0].plot(steps * tokens_per_step / 1e6, values, color=color(theme, arm), linewidth=1.8)
    floor = min(values[-1] for _, values in smooth.values())
    axes[0].set_ylim(floor - 0.08, floor + 1.0)
    titled(axes[0], theme, "Training loss (smoothed)", "Training tokens (millions)", "Next-token loss (nats)")
    for arm in order:
        if arm == reference:
            continue
        common = np.intersect1d(smooth[arm][0], smooth[reference][0])
        if not len(common):
            continue
        current = smooth[arm][1][np.isin(smooth[arm][0], common)]
        base = smooth[reference][1][np.isin(smooth[reference][0], common)]
        axes[1].plot(common * tokens_per_step / 1e6, current - base, color=color(theme, arm), linewidth=1.8)
        ends.append((common[-1] * tokens_per_step / 1e6, float(current[-1] - base[-1]), LABELS[arm], color(theme, arm)))
    axes[1].axhline(0, color=theme["axis"], linewidth=0.8)
    axes[1].set_ylim(-0.12, 0.2)
    titled(
        axes[1],
        theme,
        f"Gap to {LABELS[reference].lower()} on identical batches",
        "Training tokens (millions)",
        "Loss difference (nats)",
    )
    if ends:
        label_line_ends(axes[1], theme, ends)
    if curves:
        ends = []
        for arm in [arm for arm in ENTITIES if arm in curves]:
            steps = np.asarray([step for step, _ in curves[arm]])
            values = np.asarray([value for _, value in curves[arm]])
            axes[2].plot(
                steps * tokens_per_step / 1e6,
                values,
                color=color(theme, arm),
                linewidth=1.8,
                marker="o",
                markersize=4.5,
                markeredgecolor=theme["surface"],
                markeredgewidth=1.0,
            )
            ends.append((steps[-1] * tokens_per_step / 1e6, float(values[-1]), LABELS[arm], color(theme, arm)))
        lowest = min(value for points in curves.values() for _, value in points)
        axes[2].set_ylim(lowest - 0.05, lowest + 0.6)
        titled(
            axes[2], theme, "Held-out loss, PG-19 validation", "Training tokens (millions)", "Next-token loss (nats)"
        )
        label_line_ends(axes[2], theme, ends)
        results["validation_curves"] = {arm: curves[arm] for arm in curves}
    figure_legend(figure, theme, order)
    figure.subplots_adjust(
        left=0.06, right=0.9 if curves else 0.86, top=0.8, bottom=0.14, wspace=0.42 if curves else 0.3
    )
    save(figure, output, "fig_training", mode)
    results["training"] = {
        arm: {
            "steps": rows[-1]["step"],
            "tokens_seen": rows[-1]["tokens_seen"],
            "final_train_nll_ema": float(smooth[arm][1][-1]),
            "median_tokens_per_second": float(np.median([row["tokens_per_second"] for row in rows[len(rows) // 10 :]])),
            "peak_device_bytes": max(row.get("device_peak_bytes", 0) for row in rows),
        }
        for arm, rows in metrics.items()
    }


def figure_positions(theme, mode, evals, output, results, *, dataset, window, name, title, minimum=64):
    records = {}
    for arm in ("full", "swa", "ck", "ttt", "ttt_noadapt", "full_sw4096"):
        record = load_eval(evals, f"main_{arm}__{dataset}_w{window}")
        if record is not None:
            records[arm] = record
    if len(records) < 2:
        return
    reference = next(arm for arm in ("full", "full_sw4096", "swa") if arm in records)
    order = [arm for arm in ("full", "full_sw4096", "swa", "ttt", "ck", "ttt_noadapt") if arm in records]
    bin_size = records[reference]["summary"]["bin"]
    centers = (np.arange(window // bin_size) + 0.5) * bin_size
    keep = centers >= minimum / 2
    figure, axes = new_figure(theme, 2, width=10.4, height=3.7)
    ends = []
    for arm in order:
        values = by_position(records[arm])
        axes[0].plot(centers[keep], values[keep], color=color(theme, arm), linewidth=1.8)
        ends.append((centers[-1], float(values[-1]), LABELS[arm], color(theme, arm)))
    ticks = [
        length for length in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768) if minimum <= length <= window
    ]
    context_ticks(axes[0], ticks)
    titled(axes[0], theme, title, "Tokens of context", "Next-token loss (nats)")
    label_line_ends(axes[0], theme, ends)
    ends = []
    for arm in order:
        if arm == reference:
            continue
        mean, low, high = bootstrap_curve(records[arm], records[reference])
        axes[1].fill_between(centers[keep], low[keep], high[keep], color=color(theme, arm), alpha=0.12, linewidth=0)
        axes[1].plot(centers[keep], mean[keep], color=color(theme, arm), linewidth=1.8)
        ends.append((centers[-1], float(mean[-1]), LABELS[arm], color(theme, arm)))
    axes[1].axhline(0, color=theme["axis"], linewidth=0.8)
    context_ticks(axes[1], ticks)
    titled(
        axes[1],
        theme,
        f"Gap to {LABELS[reference].lower()} (95% CI over books)",
        "Tokens of context",
        "Loss difference (nats)",
    )
    label_line_ends(axes[1], theme, ends)
    figure_legend(figure, theme, [arm for arm in order])
    figure.subplots_adjust(left=0.065, right=0.86, top=0.8, bottom=0.14, wspace=0.62)
    save(figure, output, name, mode)
    table = {
        arm: {**scalar(record), "nll_by_position": by_position(record).tolist()} for arm, record in records.items()
    }
    comparisons = {}
    late = np.arange(window // bin_size) >= (window // bin_size) * 3 // 4
    for arm in order:
        for other in order:
            if arm == other:
                continue
            comparisons[f"{arm}_minus_{other}"] = {
                "all_positions": bootstrap_difference(records[arm], records[other]),
                "last_quarter": bootstrap_difference(records[arm], records[other], bins=late),
            }
    results[name] = {"dataset": dataset, "window": window, "bin": bin_size, "arms": table, "paired": comparisons}


def figure_ablations(theme, mode, evals, output, results):
    def nll(name, suffix="pg19val_w2048"):
        path = evals / f"abl_{name}__{suffix}.json"
        return json.loads(path.read_text(encoding="utf-8"))["nll"] if path.exists() else None

    groups = [
        ("Inner learning rate", [("0.1", "ttt_ilr0p1"), ("0.3", "ttt_ilr0p3"), ("1", "ttt"), ("3", "ttt_ilr3")]),
        ("Update chunk b (tokens)", [("64", "ttt_chunk64"), ("128", "ttt"), ("256", "ttt_chunk256")]),
        ("Layers with fast weights", [("1", "ttt_suffix1"), ("2", "ttt"), ("4", "ttt_suffix4")]),
        ("Attention window k (tokens)", [("128", "ttt_window128"), ("256", "ttt"), ("512", "ttt_window512")]),
    ]
    available = [(title, [(label, nll(name)) for label, name in items]) for title, items in groups]
    available = [(title, [(label, value) for label, value in items if value is not None]) for title, items in available]
    available = [group for group in available if len(group[1]) >= 2]
    table = {}
    for path in sorted(evals.glob("abl_*__pg19val_w2048.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        table[path.name.split("__")[0][4:]] = {
            "nll": summary["nll"],
            "perplexity": summary["perplexity"],
            "step": summary["checkpoint_step"],
        }
    for path in sorted(evals.glob("abl_*__fineweb_w2048.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        table.setdefault(path.name.split("__")[0][4:], {})["fineweb_nll"] = summary["nll"]
    if table:
        results["ablations"] = table
    if not available:
        return
    figure, axes = new_figure(theme, len(available), width=2.75 * len(available) + 0.6, height=3.3)
    baselines = {arm: nll(arm) for arm in ("swa", "full")}
    values = [value for _, items in available for _, value in items] + [value for value in baselines.values() if value]
    low, high = min(values), max(values)
    for axis, (title, items) in zip(axes, available):
        positions = np.arange(len(items))
        axis.plot(
            positions,
            [value for _, value in items],
            color=color(theme, "ttt"),
            linewidth=1.8,
            marker="o",
            markersize=6.5,
            markeredgecolor=theme["surface"],
            markeredgewidth=1.5,
        )
        for arm, value in baselines.items():
            if value is not None:
                axis.axhline(value, color=color(theme, arm), linewidth=1.4)
        axis.set_xticks(positions)
        axis.set_xticklabels([label for label, _ in items])
        axis.set_xlim(-0.4, len(items) - 0.6)
        axis.set_ylim(low - 0.015, high + 0.015)
        titled(axis, theme, title, "", "PG-19 validation loss (nats)" if axis is axes[0] else "")
    figure_legend(figure, theme, [arm for arm in ("full", "swa", "ttt") if arm == "ttt" or baselines.get(arm)])
    figure.subplots_adjust(left=0.07, right=0.985, top=0.78, bottom=0.1, wspace=0.34)
    save(figure, output, "fig_ablations", mode)


def figure_cost(theme, mode, runs, evals, output, results):
    bench = {}
    for arm, name in (("full", "full_b16_t4096"), ("swa", "swa_b16_nogc"), ("ck", "ck_b16"), ("ttt", "ttt_b16_nogc")):
        path = runs / "bench" / f"{name}.json"
        if path.exists():
            bench[arm] = json.loads(path.read_text(encoding="utf-8"))
    latency = {}
    for arm in ("full", "swa", "ttt"):
        points = []
        for path in sorted(evals.glob(f"latency_{arm}__w*.json")):
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("eval_tokens_per_second"):
                points.append((summary["window"], summary["eval_tokens_per_second"]))
        if points:
            latency[arm] = sorted(points)
    if not bench and not latency:
        return
    figure, axes = new_figure(theme, 2, width=10.4, height=3.3)
    if bench:
        order = [arm for arm in ("swa", "ck", "full", "ttt") if arm in bench]
        positions = np.arange(len(order))[::-1]
        values = [bench[arm]["tokens_per_second"] / 1000 for arm in order]
        axes[0].barh(positions, values, height=0.42, color=[color(theme, arm) for arm in order], linewidth=0)
        axes[0].set_yticks(positions)
        axes[0].set_yticklabels([LABELS[arm] for arm in order])
        axes[0].grid(True, axis="x", color=theme["grid"], linewidth=0.8)
        axes[0].grid(False, axis="y")
        for position, value in zip(positions, values):
            axes[0].text(
                value + max(values) * 0.015,
                position,
                f"{value:.0f}K",
                va="center",
                ha="left",
                fontsize=8.5,
                color=theme["ink2"],
            )
        axes[0].set_xlim(0, max(values) * 1.14)
        titled(axes[0], theme, "Training throughput, one H200, 4K context", "Thousand tokens per second", "")
        results["training_cost"] = {
            arm: {
                key: bench[arm][key]
                for key in (
                    "parameters",
                    "fast_parameters",
                    "tokens_per_second",
                    "median_step_seconds",
                    "peak_bytes_in_use",
                    "batch_size",
                    "sequence_length",
                )
            }
            for arm in bench
        }
    if latency:
        ends = []
        for arm in [arm for arm in ENTITIES if arm in latency]:
            lengths = np.asarray([length for length, _ in latency[arm]])
            values = np.asarray([value for _, value in latency[arm]]) / 1000
            axes[1].plot(
                lengths,
                values,
                color=color(theme, arm),
                linewidth=1.8,
                marker="o",
                markersize=6.5,
                markeredgecolor=theme["surface"],
                markeredgewidth=1.5,
            )
            ends.append((lengths[-1], float(values[-1]), LABELS[arm], color(theme, arm)))
        context_ticks(axes[1], sorted({length for points in latency.values() for length, _ in points}))
        axes[1].set_yscale("log")
        axes[1].grid(True, axis="y", which="both", color=theme["grid"], linewidth=0.8)
        titled(
            axes[1],
            theme,
            "Inference throughput vs context length",
            "Context length (tokens)",
            "Thousand tokens per second",
        )
        label_line_ends(axes[1], theme, ends)
        results["inference_throughput"] = latency
    figure_legend(figure, theme, [arm for arm in ENTITIES if arm in latency or arm in bench])
    figure.subplots_adjust(left=0.13, right=0.88, top=0.78, bottom=0.15, wspace=0.3)
    save(figure, output, "fig_cost", mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成技术报告图表")
    parser.add_argument("--runs", type=Path, default=Path("outputs"))
    parser.add_argument("--evals", type=Path, default=Path("outputs/evals"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets/report"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for mode, theme in THEMES.items():
        scratch = results if mode == "light" else {}
        figure_training(theme, mode, args.runs, args.evals, args.output, scratch)
        figure_positions(
            theme,
            mode,
            args.evals,
            args.output,
            scratch,
            dataset="pg19test",
            window=4096,
            name="fig_context_4k",
            title="PG-19 test loss by context length",
        )
        figure_positions(
            theme,
            mode,
            args.evals,
            args.output,
            scratch,
            dataset="fineweb",
            window=4096,
            name="fig_context_4k_fineweb",
            title="FineWeb-Edu held-out loss by context length",
        )
        figure_positions(
            theme,
            mode,
            args.evals,
            args.output,
            scratch,
            dataset="pg19test",
            window=32768,
            name="fig_context_32k",
            title="PG-19 test loss beyond the training length",
            minimum=256,
        )
        figure_ablations(theme, mode, args.evals, args.output, scratch)
        figure_cost(theme, mode, args.runs, args.evals, args.output, scratch)
    (args.output / "results.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"figures": sorted(path.name for path in args.output.glob("fig_*")), "sections": sorted(results)}))


if __name__ == "__main__":
    main()
