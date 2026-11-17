#!/usr/bin/env python3
"""从原始训练指标和评测结果生成技术报告的图表与汇总数字。

输入全部是训练/评测脚本写出的原始文件（``metrics.jsonl``、``outputs/evals/*.json|npz``、
``outputs/bench/*.json``、``outputs/benchmarks-v2/*.json``、``outputs/audits/*.json``），不手工
抄写任何数字。输出：

- ``<output>/fig_*.{light,dark}.svg`` 和 ``fig_*.png``：图表（浅色/深色两套）；
- ``<output>/results.json``：报告正文和表格使用的全部数字，差值附按文档重采样的 95% 置信区间。

颜色按“实体”固定分配（同一个模型在所有图里颜色相同），取自一组通过色觉障碍可分性检查的
分类色；两条及以上曲线必有图例，线尾另有直接标注，文字一律用墨色而不用数据色。
缺少某项评测时对应的图和字段会被跳过，因此可以在实验进行中反复运行。
"""

from __future__ import annotations

import argparse
import json
import shutil
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
ALIASES = {"full_sw4096": "full", "full_extrapolated": "full"}
LABELS = {
    "full": "Full attention",
    "swa": "Sliding window",
    "ttt": "TTT-E2E",
    "ck": "No-update control",
    "ttt_noadapt": "TTT-E2E, updates off",
    "full_sw4096": "Full attention, last 4K only",
    "full_extrapolated": "Full attention, untruncated",
}
MAIN_ARMS = ("full", "swa", "ck", "ttt")
RANGES_4K = ((0, 512), (512, 1024), (1024, 2048), (2048, 4096))
RANGES_32K = ((0, 4096), (4096, 8192), (8192, 16384), (16384, 32768))
BENCHMARK_METRICS = {
    "hellaswag": "acc_norm",
    "piqa": "acc_norm",
    "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm",
    "winogrande": "acc",
    "sciq": "acc",
    "openbookqa": "acc_norm",
    "blimp": "acc",
    "mmlu": "acc",
    "lambada_openai": "acc",
}


def color(theme, entity):
    return theme["series"][ENTITIES.index(ALIASES.get(entity, entity))]


def new_figure(theme, columns, *, width, height):
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "path", "axes.unicode_minus": False}
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
    axis.set_title(title, loc="left", color=theme["ink"], fontsize=10, fontweight="bold", pad=8)
    axis.set_xlabel(xlabel, color=theme["ink2"], fontsize=9)
    axis.set_ylabel(ylabel, color=theme["ink2"], fontsize=9)


def figure_legend(figure, theme, entities, *, left=0.05, top=0.985, styles=None):
    styles = styles or {}
    handles = [
        Line2D([0], [0], color=color(theme, entity), linewidth=2, linestyle=styles.get(entity, "-"))
        for entity in entities
    ]
    figure.legend(
        handles,
        [LABELS[entity] for entity in entities],
        loc="upper left",
        bbox_to_anchor=(left, top),
        ncol=len(entities),
        frameon=False,
        fontsize=8.5,
        handlelength=1.8,
        columnspacing=1.5,
        labelcolor=theme["ink2"],
    )


def end_marker(axis, theme, x, y, tint):
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
        end_marker(axis, theme, x, y, tint)
        leader = {"arrowstyle": "-", "color": theme["axis"], "linewidth": 0.8, "shrinkA": 0, "shrinkB": 4}
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
            arrowprops=leader if abs(offset) > 2 else None,
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


def padded_limits(axis, values, *, pad=0.08, minimum_span=0.02):
    low, high = float(np.min(values)), float(np.max(values))
    span = max(high - low, minimum_span)
    axis.set_ylim(low - pad * span, high + pad * span)


# --------------------------------------------------------------------------- 读取数据


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_metrics(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    by_step = {row["step"]: row for row in rows}  # 续训会重复写入同一步，保留最后一次
    return [by_step[step] for step in sorted(by_step)]


def ema(values, alpha=0.1):
    result, state = [], None
    for value in values:
        state = value if state is None else (1 - alpha) * state + alpha * value
        result.append(state)
    return np.asarray(result)


def load_eval(directory, name):
    path = directory / f"{name}.json"
    if not path.exists() or not path.with_suffix(".npz").exists():
        return None
    arrays = np.load(path.with_suffix(".npz"))
    return {
        "summary": read_json(path),
        "loss": arrays["loss_sums"].astype(np.float64),
        "count": arrays["counts"].astype(np.float64),
        "document": arrays["document"],
        "position": arrays["position"],
    }


def cluster_sums(record):
    """把窗口级的 loss 和计数合并到文档；连续流评测只有一个“文档”，此时每个窗口自成一簇。"""
    documents = record["document"]
    if len(np.unique(documents)) == 1:
        return record["loss"], record["count"]
    order = np.unique(documents)
    loss = np.stack([record["loss"][documents == document].sum(axis=0) for document in order])
    count = np.stack([record["count"][documents == document].sum(axis=0) for document in order])
    return loss, count


def same_windows(first, second):
    return first["loss"].shape == second["loss"].shape and np.array_equal(first["position"], second["position"])


def bootstrap_mean(record, *, bins=None, repetitions=2000, seed=0):
    loss, count = cluster_sums(record)
    if bins is not None:
        loss, count = loss[:, bins], count[:, bins]
    loss, count = loss.sum(axis=1), count.sum(axis=1)
    draws = np.random.default_rng(seed).integers(0, len(count), size=(repetitions, len(count)))
    samples = loss[draws].sum(axis=1) / count[draws].sum(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"mean": float(loss.sum() / count.sum()), "ci95": [float(low), float(high)], "clusters": int(len(count))}


def bootstrap_difference(first, second, *, bins=None, repetitions=2000, seed=0):
    """first − second 的平均 NLL 差及 95% 区间；两者必须评测同一批窗口（配对）。"""
    if not same_windows(first, second):
        raise ValueError("配对比较要求两次评测使用完全相同的窗口。")
    loss_a, count = cluster_sums(first)
    loss_b, _ = cluster_sums(second)
    if bins is not None:
        loss_a, loss_b, count = loss_a[:, bins], loss_b[:, bins], count[:, bins]
    difference, weight = (loss_a - loss_b).sum(axis=1), count.sum(axis=1)
    draws = np.random.default_rng(seed).integers(0, len(weight), size=(repetitions, len(weight)))
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
    draws = np.random.default_rng(seed).integers(0, len(count), size=(repetitions, len(count)))
    samples = difference[draws].sum(axis=1) / np.maximum(count[draws].sum(axis=1), 1)
    mean = difference.sum(axis=0) / np.maximum(count.sum(axis=0), 1)
    low, high = np.quantile(samples, [0.025, 0.975], axis=0)
    return mean, low, high


def by_position(record):
    return record["loss"].sum(axis=0) / np.maximum(record["count"].sum(axis=0), 1)


def range_bins(record, begin, end):
    size = record["summary"]["bin"]
    return np.arange(begin // size, end // size)


def describe(record, ranges):
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
    result = {key: summary.get(key) for key in keys}
    result["nll_ci95"] = bootstrap_mean(record)["ci95"]
    result["by_range"] = {
        f"{begin}-{end}": bootstrap_mean(record, bins=range_bins(record, begin, end))
        for begin, end in ranges
        if end <= summary["window"]
    }
    return result


def paired_table(records, ranges):
    table = {}
    for first in records:
        for second in records:
            if first == second or not same_windows(records[first], records[second]):
                continue
            entry = {"all": bootstrap_difference(records[first], records[second])}
            for begin, end in ranges:
                if end <= records[first]["summary"]["window"]:
                    bins = range_bins(records[first], begin, end)
                    entry[f"{begin}-{end}"] = bootstrap_difference(records[first], records[second], bins=bins)
            table[f"{first}_minus_{second}"] = entry
    return table


# --------------------------------------------------------------------------- 各图


def figure_training(theme, mode, runs, evals, output, results):
    metrics = {
        arm: read_metrics(runs / f"main-{arm}" / "metrics.jsonl")
        for arm in MAIN_ARMS
        if (runs / f"main-{arm}" / "metrics.jsonl").exists()
    }
    if not metrics:
        return
    order = [arm for arm in ENTITIES if arm in metrics]
    curves = {}
    for arm in MAIN_ARMS:
        points = {}
        for suffix in ("pg19val", "fineweb"):
            for path in sorted(evals.glob(f"curve_{arm}_s*__{suffix}.json")):
                summary = read_json(path)
                points.setdefault(summary["checkpoint_step"], {})[suffix] = summary["nll"]
        if points:
            curves[arm] = {step: points[step] for step in sorted(points)}
    first = metrics[order[0]][0]
    tokens_per_step = first["tokens_seen"] / first["step"]
    figure, axes = new_figure(theme, 3, width=11.4, height=3.5)
    smooth = {}
    for arm in order:
        steps = np.asarray([row["step"] for row in metrics[arm]])
        smooth[arm] = (steps, ema([row["nll"] for row in metrics[arm]]))
        axes[0].plot(steps * tokens_per_step / 1e6, smooth[arm][1], color=color(theme, arm), linewidth=1.6)
    floor = min(values[-1] for _, values in smooth.values())
    axes[0].set_ylim(floor - 0.06, floor + 0.9)
    titled(axes[0], theme, "Training loss (running mean)", "Training tokens (millions)", "Next-token loss (nats)")
    panels = (("pg19val", "PG-19 validation loss"), ("fineweb", "FineWeb-Edu held-out loss"))
    for axis, (suffix, title) in zip(axes[1:], panels):
        ends, values_seen = [], []
        for arm in [arm for arm in ENTITIES if arm in curves]:
            points = [(step, value[suffix]) for step, value in curves[arm].items() if suffix in value]
            if not points:
                continue
            steps = np.asarray([step for step, _ in points]) * tokens_per_step / 1e6
            values = np.asarray([value for _, value in points])
            axis.plot(steps, values, color=color(theme, arm), linewidth=1.6)
            ends.append((steps[-1], float(values[-1]), LABELS[arm], color(theme, arm)))
            values_seen.extend(values[len(values) // 4 :])
        titled(axis, theme, title, "Training tokens (millions)", "Next-token loss (nats)")
        if ends:
            padded_limits(axis, values_seen, pad=0.12)
            label_line_ends(axis, theme, ends)
    figure_legend(figure, theme, order)
    figure.subplots_adjust(left=0.05, right=0.895, top=0.8, bottom=0.14, wspace=0.62)
    save(figure, output, "fig_training", mode)
    results["validation_curves"] = {
        arm: {str(step): value for step, value in points.items()} for arm, points in curves.items()
    }
    results["training"] = {}
    for arm, rows in metrics.items():
        seconds = [row["seconds"] for row in rows]
        results["training"][arm] = {
            "steps": rows[-1]["step"],
            "tokens_seen": rows[-1]["tokens_seen"],
            "final_train_nll": float(smooth[arm][1][-1]),
            "logged_steps": len(rows),
            "median_step_seconds": float(np.median(seconds[len(seconds) // 10 :])),
            "peak_device_bytes": max(row.get("device_peak_bytes", 0) for row in rows),
            "max_grad_norm": max(row["grad_norm"] for row in rows[len(rows) // 10 :]),
            # 预热结束之后，外循环梯度范数超过裁剪阈值 1.0 的记录步所占比例。
            "clipped_fraction": float(np.mean([row["grad_norm"] > 1.0 for row in rows[len(rows) // 10 :]])),
            "wall_clock_hours": float(sum(seconds) * (rows[-1]["step"] / len(rows)) / 3600),
        }
        if "inner_grad_norm" in rows[-1]:
            results["training"][arm].update(
                final_fast_weight_norm=rows[-1]["fast_weight_norm"],
                initial_fast_weight_norm=rows[0]["fast_weight_norm"],
                final_inner_grad_norm=rows[-1]["inner_grad_norm"],
                final_inner_update_norm=rows[-1]["inner_update_norm"],
            )


def load_arms(evals, pattern, arms):
    records = {}
    for arm in arms:
        record = load_eval(evals, pattern.format(arm=arm))
        if record is not None:
            records[arm] = record
    return records


def figure_positions(theme, mode, records, output, results, *, name, title, ranges, minimum=64, reference=None):
    if len(records) < 2:
        return
    reference = reference or next(arm for arm in ("full", "full_sw4096", "swa") if arm in records)
    order = [arm for arm in ("full", "full_sw4096", "swa", "ttt", "ck", "ttt_noadapt") if arm in records]
    window, bin_size = records[reference]["summary"]["window"], records[reference]["summary"]["bin"]
    centers = (np.arange(window // bin_size) + 0.5) * bin_size
    keep = centers >= minimum / 2
    figure, axes = new_figure(theme, 2, width=10.6, height=3.7)
    ends, shown = [], []
    for arm in order:
        values = by_position(records[arm])
        axes[0].plot(centers[keep], values[keep], color=color(theme, arm), linewidth=1.6)
        ends.append((centers[-1], float(values[-1]), LABELS[arm], color(theme, arm)))
        shown.extend(values[keep])
    ticks = [
        length for length in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768) if minimum <= length <= window
    ]
    context_ticks(axes[0], ticks)
    padded_limits(axes[0], shown, pad=0.05)
    titled(axes[0], theme, title, "Tokens of preceding context", "Next-token loss (nats)")
    label_line_ends(axes[0], theme, ends)
    ends, shown = [], []
    for arm in order:
        if arm == reference or not same_windows(records[arm], records[reference]):
            continue
        mean, low, high = bootstrap_curve(records[arm], records[reference])
        axes[1].fill_between(centers[keep], low[keep], high[keep], color=color(theme, arm), alpha=0.14, linewidth=0)
        axes[1].plot(centers[keep], mean[keep], color=color(theme, arm), linewidth=1.6)
        ends.append((centers[-1], float(mean[-1]), LABELS[arm], color(theme, arm)))
        shown.extend([*low[keep], *high[keep]])
    axes[1].axhline(0, color=theme["axis"], linewidth=0.8)
    context_ticks(axes[1], ticks)
    padded_limits(axes[1], [*shown, 0.0], pad=0.06)
    titled(
        axes[1],
        theme,
        f"Difference from {LABELS[reference].lower()} (band: 95% CI)",
        "Tokens of preceding context",
        "Loss difference (nats)",
    )
    label_line_ends(axes[1], theme, ends)
    figure_legend(figure, theme, order)
    figure.subplots_adjust(left=0.06, right=0.845, top=0.8, bottom=0.14, wspace=0.66)
    save(figure, output, name, mode)
    results[name] = {
        "window": window,
        "bin": bin_size,
        "reference": reference,
        "arms": {
            arm: {**describe(record, ranges), "nll_by_position": by_position(record).tolist()}
            for arm, record in records.items()
        },
        "paired": paired_table(records, ranges),
    }


def collect_long_context(evals, results):
    """超出训练长度的评测：各臂在 32K 窗口上的结果，以及全注意力直接外推的结果。"""
    records = load_arms(evals, "main_{arm}__pg19test_w32768", ("swa", "ttt", "ck", "ttt_noadapt"))
    extrapolated = {}
    for window in (8192, 16384):
        record = load_eval(evals, f"main_full__pg19test_w{window}")
        if record is not None:
            extrapolated[window] = record
            results.setdefault("full_extrapolation", {})[str(window)] = {
                **describe(record, ((0, 4096), (4096, 8192), (8192, 16384))),
                "nll_by_position": by_position(record).tolist(),
            }
    if "ttt" in records:
        summary = records["ttt"]["summary"]
        results["fast_weights_32k"] = {
            "chunk": summary["ttt_chunk_size"],
            "norm_by_chunk": summary["fast_weight_norm_by_chunk"],
            "update_norm_by_chunk": summary["inner_update_norm_by_chunk"],
            "grad_norm_by_chunk": summary["inner_grad_norm_by_chunk"],
        }
    return records, extrapolated


def figure_long_context(theme, mode, records, extrapolated, output, results):
    if len(records) < 2:
        return
    reference = "swa" if "swa" in records else next(iter(records))
    order = [arm for arm in ("full_sw4096", "swa", "ttt", "ck", "ttt_noadapt") if arm in records]
    window, bin_size = records[reference]["summary"]["window"], records[reference]["summary"]["bin"]
    centers = (np.arange(window // bin_size) + 0.5) * bin_size
    keep = centers >= 256
    figure, axes = new_figure(theme, 2, width=10.6, height=3.7)
    ends, shown = [], []
    for arm in order:
        values = by_position(records[arm])
        axes[0].plot(centers[keep], values[keep], color=color(theme, arm), linewidth=1.6)
        ends.append((centers[-1], float(values[-1]), LABELS[arm], color(theme, arm)))
        shown.extend(values[keep])
    styles = {}
    if extrapolated:
        longest = extrapolated[max(extrapolated)]
        size = longest["summary"]["bin"]
        positions = (np.arange(longest["summary"]["window"] // size) + 0.5) * size
        values = by_position(longest)
        visible = positions >= 256
        axes[0].plot(
            positions[visible], values[visible], color=color(theme, "full"), linewidth=1.6, linestyle=(0, (4, 2))
        )
        ends.append((positions[-1], float(values[-1]), LABELS["full_extrapolated"], color(theme, "full")))
        styles["full_extrapolated"] = (0, (4, 2))
    ticks = [256, 1024, 4096, 16384, 32768]
    context_ticks(axes[0], ticks)
    padded_limits(axes[0], shown, pad=0.08)
    axes[0].axvline(4096, color=theme["axis"], linewidth=0.8)
    axes[0].annotate(
        "training length",
        xy=(4096, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4, -3),
        textcoords="offset points",
        va="top",
        fontsize=8,
        color=theme["muted"],
    )
    titled(
        axes[0],
        theme,
        "PG-19 test loss up to 32K tokens of context",
        "Tokens of preceding context",
        "Next-token loss (nats)",
    )
    label_line_ends(axes[0], theme, ends)
    ends, shown = [], []
    for arm in order:
        if arm == reference or not same_windows(records[arm], records[reference]):
            continue
        mean, low, high = bootstrap_curve(records[arm], records[reference])
        axes[1].fill_between(centers[keep], low[keep], high[keep], color=color(theme, arm), alpha=0.14, linewidth=0)
        axes[1].plot(centers[keep], mean[keep], color=color(theme, arm), linewidth=1.6)
        ends.append((centers[-1], float(mean[-1]), LABELS[arm], color(theme, arm)))
        shown.extend([*low[keep], *high[keep]])
    axes[1].axhline(0, color=theme["axis"], linewidth=0.8)
    axes[1].axvline(4096, color=theme["axis"], linewidth=0.8)
    context_ticks(axes[1], ticks)
    padded_limits(axes[1], [*shown, 0.0], pad=0.06)
    titled(
        axes[1],
        theme,
        f"Difference from {LABELS[reference].lower()} (band: 95% CI)",
        "Tokens of preceding context",
        "Loss difference (nats)",
    )
    label_line_ends(axes[1], theme, ends)
    legend = order + (["full_extrapolated"] if extrapolated else [])
    figure_legend(figure, theme, legend, styles=styles)
    figure.subplots_adjust(left=0.06, right=0.82, top=0.8, bottom=0.14, wspace=0.78)
    save(figure, output, "fig_context_32k", mode)
    results["fig_context_32k"] = {
        "window": window,
        "bin": bin_size,
        "reference": reference,
        "arms": {
            arm: {**describe(record, RANGES_32K), "nll_by_position": by_position(record).tolist()}
            for arm, record in records.items()
        },
        "paired": paired_table(records, RANGES_32K),
    }


def figure_recall(theme, mode, evals, output, results):
    arms = [arm for arm in ("full", "swa", "ttt", "ck", "ttt_noadapt") if (evals / f"recall_{arm}__base.json").exists()]
    table = {}
    for arm in arms:
        base = load_eval(evals, f"recall_{arm}__base")
        size = base["summary"]["bin"]
        for path in sorted(evals.glob(f"recall_{arm}__d*.json")):
            record = load_eval(evals, path.stem)
            source, span, target = record["summary"]["repeat_span"]
            # 只看重复段中间的 token：跳过开头（模型还无法确认这是重复）和结尾（紧接着的原文不连续）。
            bins = np.arange(target // size + 1, (target + span) // size - 1)
            gain = bootstrap_difference(base, record, bins=bins)
            table.setdefault(arm, {})[str(target - source)] = {
                "repeat_nll": bootstrap_mean(record, bins=bins)["mean"],
                "unrepeated_nll": bootstrap_mean(base, bins=bins)["mean"],
                "gain": gain["mean"],
                "gain_ci95": gain["ci95"],
                "clusters": gain["clusters"],
            }
    if not table:
        return
    results["recall"] = table
    figure, axes = new_figure(theme, 1, width=6.4, height=3.6)
    axis = axes[0]
    ends, distances = [], []
    for arm in arms:
        points = sorted((int(distance), value) for distance, value in table[arm].items())
        distances = [distance for distance, _ in points]
        gains = [value["gain"] for _, value in points]
        axis.plot(
            distances,
            gains,
            color=color(theme, arm),
            linewidth=1.6,
            marker="o",
            markersize=5,
            markeredgecolor=theme["surface"],
            markeredgewidth=1.2,
        )
        ends.append((distances[-1], gains[-1], LABELS[arm], color(theme, arm)))
    axis.axhline(0, color=theme["axis"], linewidth=0.8)
    axis.set_xscale("log", base=2)
    axis.set_xticks(distances)
    axis.set_xticklabels([f"{distance:,}" for distance in distances])
    axis.minorticks_off()
    titled(
        axis,
        theme,
        "Verbatim recall of a repeated 64-token span",
        "Distance to the first occurrence (tokens)",
        "Loss reduction on the repeat (nats)",
    )
    label_line_ends(axis, theme, ends)
    figure_legend(figure, theme, arms, left=0.08)
    figure.subplots_adjust(left=0.1, right=0.74, top=0.8, bottom=0.14)
    save(figure, output, "fig_recall", mode)


def collect_ablations(evals, runs, results):
    table = {}
    for path in sorted([*evals.glob("abl_*__pg19val_w2048.json"), *evals.glob("scale_*__pg19val_w2048.json")]):
        name = path.name.split("__")[0]
        if name.endswith("_noadapt"):
            continue
        summary = read_json(path)
        entry = {"nll": summary["nll"], "perplexity": summary["perplexity"], "step": summary["checkpoint_step"]}
        by_bin = np.asarray(summary["nll_by_position_bin"])
        size = summary["bin"]
        entry["nll_first_window"] = float(by_bin[: 256 // size].mean())
        entry["nll_last_quarter"] = float(by_bin[-(len(by_bin) // 4) :].mean())
        for suffix, key in (("fineweb_w2048", "fineweb_nll"), ("pg19val_w2048", None)):
            other = evals / f"{name}__{suffix}.json"
            if key and other.exists():
                entry[key] = read_json(other)["nll"]
        off = evals / f"{name}_noadapt__pg19val_w2048.json"
        if off.exists():
            entry["noadapt_nll"] = read_json(off)["nll"]
        run = runs / name.replace("_", "-", 1)
        if (run / "metrics.jsonl").exists():
            rows = read_metrics(run / "metrics.jsonl")
            seconds = [row["seconds"] for row in rows]
            entry["median_step_seconds"] = float(np.median(seconds[len(seconds) // 5 :]))
            entry["tokens_seen"] = rows[-1]["tokens_seen"]
        if (run / "experiment.json").exists():
            experiment = read_json(run / "experiment.json")
            entry["steps"] = experiment["training"]["steps"]
        table[name] = entry
    if not table:
        return table
    results["ablations"] = {name: value for name, value in table.items() if name.startswith("abl_")}
    seeds = {}
    for arm in ("full", "swa", "ttt"):
        values = [
            table[name]["nll"] for name in (f"abl_{arm}", f"abl_{arm}_seed1", f"abl_{arm}_seed2") if name in table
        ]
        if len(values) >= 2:
            seeds[arm] = {"values": values, "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}
    paired = {}
    for first, second in (("ttt", "swa"), ("ttt", "full"), ("swa", "full")):
        differences = []
        for suffix in ("", "_seed1", "_seed2"):
            a, b = f"abl_{first}{suffix}", f"abl_{second}{suffix}"
            if a in table and b in table:
                differences.append(table[a]["nll"] - table[b]["nll"])
        if len(differences) >= 2:
            paired[f"{first}_minus_{second}"] = {
                "values": differences,
                "mean": float(np.mean(differences)),
                "sd": float(np.std(differences, ddof=1)),
            }
    if seeds:
        results["seeds"] = {"arms": seeds, "paired": paired}
    scaling = {}
    sizes = {"xs": "scale_xs_{arm}", "s": "abl_{arm}", "m": "scale_m_{arm}"}
    for size, pattern in sizes.items():
        for arm in ("full", "swa", "ttt"):
            name = pattern.format(arm=arm)
            if name in table:
                scaling.setdefault(size, {})[arm] = table[name]
    if len(scaling) >= 2:
        results["scaling"] = scaling
    return table


def figure_ablations(theme, mode, table, output):
    groups = [
        (
            "Inner learning rate η",
            [("0.1", "abl_ttt_ilr0p1"), ("0.3", "abl_ttt_ilr0p3"), ("1", "abl_ttt"), ("3", "abl_ttt_ilr3")],
        ),
        ("Update chunk C (tokens)", [("64", "abl_ttt_chunk64"), ("128", "abl_ttt"), ("256", "abl_ttt_chunk256")]),
        ("Blocks with fast weights", [("1", "abl_ttt_suffix1"), ("2", "abl_ttt"), ("4", "abl_ttt_suffix4")]),
        (
            "Attention window k (tokens)",
            [("128", "abl_ttt_window128"), ("256", "abl_ttt"), ("512", "abl_ttt_window512")],
        ),
    ]
    available = []
    for title, items in groups:
        points = [(label, table[name]["nll"]) for label, name in items if name in table]
        if len(points) >= 2:
            available.append((title, points))
    if not available:
        return
    figure, axes = new_figure(theme, len(available), width=2.7 * len(available) + 1.5, height=3.3)
    window_baseline = {"128": "abl_swa_window128", "256": "abl_swa", "512": "abl_swa_window512"}
    references = {arm: table[f"abl_{arm}"]["nll"] for arm in ("swa", "full") if f"abl_{arm}" in table}
    values = [value for _, points in available for _, value in points] + list(references.values())
    values += [table[name]["nll"] for name in window_baseline.values() if name in table]
    for axis, (title, points) in zip(axes, available):
        positions = np.arange(len(points))
        axis.plot(
            positions,
            [value for _, value in points],
            color=color(theme, "ttt"),
            linewidth=1.6,
            marker="o",
            markersize=6.5,
            markeredgecolor=theme["surface"],
            markeredgewidth=1.5,
        )
        if title.startswith("Attention window"):
            pairs = [
                (index, table[window_baseline[label]]["nll"])
                for index, (label, _) in enumerate(points)
                if window_baseline[label] in table
            ]
            if pairs:
                axis.plot(
                    [index for index, _ in pairs],
                    [value for _, value in pairs],
                    color=color(theme, "swa"),
                    linewidth=1.6,
                    marker="o",
                    markersize=6.5,
                    markeredgecolor=theme["surface"],
                    markeredgewidth=1.5,
                )
            if "full" in references:
                axis.axhline(references["full"], color=color(theme, "full"), linewidth=1.4)
        else:
            for arm, value in references.items():
                axis.axhline(value, color=color(theme, arm), linewidth=1.4)
        axis.set_xticks(positions)
        axis.set_xticklabels([label for label, _ in points])
        axis.set_xlim(-0.4, len(points) - 0.6)
        padded_limits(axis, values, pad=0.06)
        titled(axis, theme, title, "", "PG-19 validation loss (nats)" if axis is axes[0] else "")
    figure_legend(figure, theme, [arm for arm in ("full", "swa", "ttt") if arm == "ttt" or arm in references])
    figure.subplots_adjust(left=0.065, right=0.985, top=0.78, bottom=0.1, wspace=0.36)
    save(figure, output, "fig_ablations", mode)


def figure_scaling(theme, mode, results, output):
    scaling = results.get("scaling")
    curves = results.get("validation_curves", {})
    if not scaling:
        return
    parameters = {"xs": 10.6, "s": 25.7, "m": 85.0}  # 全注意力臂的非嵌入参数量（百万）
    figure, axes = new_figure(theme, 2, width=10.2, height=3.5)
    ends = []
    for arm in ("full", "swa", "ttt"):
        points = [
            (parameters[size], scaling[size][arm]["nll"]) for size in ("xs", "s", "m") if arm in scaling.get(size, {})
        ]
        if len(points) < 2:
            continue
        axes[0].plot(
            [x for x, _ in points],
            [y for _, y in points],
            color=color(theme, arm),
            linewidth=1.6,
            marker="o",
            markersize=5,
            markeredgecolor=theme["surface"],
            markeredgewidth=1.2,
        )
        ends.append((points[-1][0], points[-1][1], LABELS[arm], color(theme, arm)))
    axes[0].set_xscale("log")
    axes[0].set_xticks(list(parameters.values()))
    axes[0].set_xticklabels(["11M", "26M", "85M"])
    axes[0].minorticks_off()
    titled(
        axes[0],
        theme,
        "Model size at a fixed 164M-token budget",
        "Non-embedding parameters",
        "PG-19 validation loss (nats)",
    )
    if ends:
        label_line_ends(axes[0], theme, ends)
    ends, shown = [], []
    if "swa" in curves:
        for arm, reference in (("ttt", "swa"), ("ttt", "full"), ("ck", "swa")):
            if arm not in curves or reference not in curves:
                continue
            steps = sorted(set(curves[arm]) & set(curves[reference]), key=int)
            points = [
                (int(step) * 65536 / 1e6, curves[arm][step]["pg19val"] - curves[reference][step]["pg19val"])
                for step in steps
                if "pg19val" in curves[arm][step] and "pg19val" in curves[reference][step]
            ]
            if not points:
                continue
            style = "-" if reference == "swa" else (0, (4, 2))
            axes[1].plot(
                [x for x, _ in points], [y for _, y in points], color=color(theme, arm), linewidth=1.6, linestyle=style
            )
            name = f"{LABELS[arm]} − {LABELS[reference].lower()}"
            ends.append((points[-1][0], points[-1][1], name, color(theme, arm)))
            shown.extend(y for _, y in points)
        axes[1].axhline(0, color=theme["axis"], linewidth=0.8)
        if shown:
            padded_limits(axes[1], [*shown, 0.0], pad=0.1)
            label_line_ends(axes[1], theme, ends)
    titled(
        axes[1],
        theme,
        "Loss gap during the 138M-parameter runs",
        "Training tokens (millions)",
        "PG-19 validation loss difference (nats)",
    )
    figure_legend(figure, theme, ["full", "swa", "ttt"])
    figure.subplots_adjust(left=0.065, right=0.8, top=0.8, bottom=0.14, wspace=0.5)
    save(figure, output, "fig_scaling", mode)


def figure_cost(theme, mode, runs, evals, output, results):
    bench = {}
    for arm, name in (("full", "full_b16_t4096"), ("swa", "swa_b16_nogc"), ("ck", "ck_b16"), ("ttt", "ttt_b16_nogc")):
        path = runs / "bench" / f"{name}.json"
        if path.exists():
            bench[arm] = read_json(path)
    latency = {}
    for arm in ("full", "swa", "ttt"):
        points = []
        for path in sorted(evals.glob(f"latency_{arm}__w*.json")):
            summary = read_json(path)
            if summary.get("eval_tokens_per_second"):
                points.append((summary["window"], summary["eval_tokens_per_second"]))
        if points:
            latency[arm] = sorted(points)
    if bench:
        keys = (
            "parameters",
            "fast_parameters",
            "tokens_per_second",
            "median_step_seconds",
            "peak_bytes_in_use",
            "batch_size",
            "sequence_length",
        )
        results["training_cost"] = {arm: {key: bench[arm][key] for key in keys} for arm in bench}
        results["training_cost_variants"] = {
            path.stem: {key: value for key, value in read_json(path).items() if key in (*keys, "overrides", "config")}
            for path in sorted((runs / "bench").glob("*.json"))
        }
    if latency:
        results["inference_throughput"] = {arm: [list(point) for point in points] for arm, points in latency.items()}
    if not bench or not latency:
        return
    figure, axes = new_figure(theme, 2, width=10.4, height=3.3)
    order = [arm for arm in ("swa", "ck", "full", "ttt") if arm in bench]
    positions = np.arange(len(order))[::-1]
    values = [bench[arm]["tokens_per_second"] / 1000 for arm in order]
    axes[0].barh(positions, values, height=0.3, color=[color(theme, arm) for arm in order], linewidth=0)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([LABELS[arm] for arm in order])
    axes[0].tick_params(axis="y", length=0)
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
    axes[0].set_ylim(-0.6, len(order) - 0.4)
    titled(axes[0], theme, "Training throughput (4K context, one H200)", "Thousand tokens per second", "")
    ends = []
    for arm in [arm for arm in ENTITIES if arm in latency]:
        lengths = np.asarray([length for length, _ in latency[arm]])
        speeds = np.asarray([value for _, value in latency[arm]]) / 1000
        axes[1].plot(
            lengths,
            speeds,
            color=color(theme, arm),
            linewidth=1.6,
            marker="o",
            markersize=5,
            markeredgecolor=theme["surface"],
            markeredgewidth=1.2,
        )
        ends.append((lengths[-1], float(speeds[-1]), LABELS[arm], color(theme, arm)))
    context_ticks(axes[1], sorted({length for points in latency.values() for length, _ in points}))
    axes[1].set_yscale("log")
    axes[1].grid(True, axis="y", which="both", color=theme["grid"], linewidth=0.8)
    titled(
        axes[1], theme, "Forward throughput by context length", "Context length (tokens)", "Thousand tokens per second"
    )
    label_line_ends(axes[1], theme, ends)
    figure_legend(figure, theme, [arm for arm in ENTITIES if arm in latency], left=0.56)
    figure.subplots_adjust(left=0.13, right=0.88, top=0.8, bottom=0.15, wspace=0.3)
    save(figure, output, "fig_cost", mode)


# --------------------------------------------------------------------------- 只汇总、不画图的部分


def collect_dynamic(evals, results):
    table = {}
    for path in sorted(evals.glob("dynamic_*.json")):
        model, setting = path.stem.split("__")[0].split("_", 2)[1:]
        split = "test" if "pg19test" in path.stem else "validation"
        summary = read_json(path)
        table.setdefault(model, {}).setdefault(setting, {})[split] = summary["nll"]
    if table:
        results["dynamic"] = table


def collect_benchmarks(directory, results):
    table = {}
    for path in sorted([*directory.glob("main_*.json"), *(directory / "reference").glob("*.json")]):
        if path.name.endswith(".predictions.jsonl") or "limit" in path.stem or path.stem == "gpt2_pg19":
            continue
        report = read_json(path)
        if report.get("limit"):
            continue
        entry = {"model": report["model"], "tasks": {}}
        for task, values in report["results"].items():
            entry["tasks"][task] = {key: value for key, value in values.items() if key != "groups"}
        table[path.stem] = entry
    extra = directory / "reference" / "gpt2_pg19.json"
    if extra.exists() and "gpt2" in table:
        table["gpt2"]["tasks"].update(read_json(extra)["results"])
    if table:
        results["benchmarks"] = table
    crosscheck = sorted((directory / "lm-eval").rglob("results_*.json"))
    if crosscheck:
        merged = {}
        for path in crosscheck:  # 不同任务分几次运行，合并它们的结果
            for task, values in read_json(path)["results"].items():
                merged[task] = {key: value for key, value in values.items() if isinstance(value, (int, float))}
        report = read_json(crosscheck[-1])
        results["lm_eval_crosscheck"] = {
            "versions": {key: report.get(key) for key in ("lm_eval_version", "transformers_version", "date")},
            "results": merged,
        }


def collect_extended_benchmarks(directory, results):
    """汇总 ARC、PIQA、GPQA、GSM8K、HumanEval、BBH 和长上下文前置诊断。"""
    if not directory.exists():
        return
    table = {}
    for path in sorted(directory.glob("*.json")):
        report = read_json(path)
        task = report.get("task")
        if task is None:
            for candidate in ("gsm8k", "humaneval", "bbh", "arc", "piqa", "gpqa"):
                if path.stem.endswith(f"_{candidate}"):
                    task = candidate
                    break
        adaptation = report.get("adaptation", report.get("adapt"))
        if adaptation is None:
            adaptation = "_adapt_" in f"_{path.stem}_" and "_noadapt_" not in f"_{path.stem}_"
        if "results" in report and report.get("protocol_version", "").startswith("synthetic_"):
            score = None
        else:
            score = report.get("accuracy", report.get("pass_at_1"))
        table[path.stem] = {
            "protocol_version": report.get("protocol_version"),
            "task": task or ("synthetic_long_context" if "results" in report else path.stem),
            "adaptation": adaptation,
            "correct": report.get("correct", report.get("passed")),
            "samples": report.get("samples"),
            "skipped": report.get("skipped", 0),
            "score": score,
            "results": report.get("results"),
            "checkpoint": report.get("checkpoint"),
            "data": report.get("data"),
        }
    if table:
        results["extended_benchmarks"] = table


def collect_audits(directory, data, results):
    audits = {}
    for name in ("sampler_main", "matched_batches", "duplicates", "benchmark_overlap"):
        path = directory / f"{name}.json"
        if path.exists():
            audits[name] = read_json(path)
    if "duplicates" in audits:
        audits["duplicates"]["fineweb_edu"].pop("heldout_duplicate_digests", None)
    if "benchmark_overlap" in audits:
        for task in audits["benchmark_overlap"]["tasks"].values():
            task.pop("matched_indices", None)
    corpora = {}
    for path in sorted(data.glob("*/manifest.json")):
        manifest = read_json(path)
        corpora[path.parent.name] = {
            **manifest.get("stats", {}),
            "shards": len(manifest["shards"]),
            "tokenizer": manifest.get("tokenizer"),
        }
    if corpora:
        audits["corpora"] = corpora
    if audits:
        results["audits"] = audits


def collect_provenance(runs, results):
    """每个训练运行的源码摘要、配置摘要和软件版本；摘要的算法见 ttt/provenance.py。"""
    provenance = {"runs": {}}
    for run in sorted(path for path in runs.iterdir() if (path / "run_metadata.json").exists()):
        if not run.name.startswith(("main-", "abl-", "scale-")):
            continue
        metadata = read_json(run / "run_metadata.json")
        provenance["runs"][run.name] = {
            "source_sha256": metadata["source_sha256"],
            "data_manifest_sha256": metadata["data_manifest_sha256"],
        }
        if "environment" not in provenance and (run / "environment.json").exists():
            environment = read_json(run / "environment.json")
            keys = ("python", "system", "packages")
            provenance["environment"] = {key: environment[key] for key in keys}
            provenance["environment"]["gpu"] = next(iter((environment.get("nvidia_smi") or "").splitlines()), "")
    provenance["source_digests"] = sorted({entry["source_sha256"] for entry in provenance["runs"].values()})
    if provenance["runs"]:
        results["provenance"] = provenance


def collect_interruptions(runs, results):
    """记录监督训练脚本捕获的中断和恢复次数。"""
    records = {}
    total = 0
    for run in sorted(path for path in runs.iterdir() if path.is_dir()):
        path = run / "interruptions.jsonl"
        if not path.exists():
            continue
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        records[run.name] = {
            "count": len(events),
            "resumed_from_steps": [event.get("resumed_from_step") for event in events],
        }
        total += len(events)
    if records:
        results["interruptions"] = {"total": total, "runs": records}


def collect_evaluation_failures(runs, results):
    """记录流水线评测日志中的资源失败，避免把失败误解为零分。"""
    log_path = runs.parent / "logs" / "pipeline_evals.log"
    if not log_path.exists():
        return
    resource_failures = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "RESOURCE_EXHAUSTED" in line:
            resource_failures.append(line.strip())
    if resource_failures:
        results["evaluation_failures"] = {
            "resource_exhausted_count": len(resource_failures),
            "log": "logs/pipeline_evals.log",
        }


def collect_niah(directory, results):
    """单针检索：准确率，以及“针在场 vs 不在场”的答案对数似然差。"""
    table = {}
    for path in sorted(directory.glob("*.json")):
        name = path.stem.replace("main_", "")
        report = read_json(path)
        table[name] = {"model": report["model"], "results": report["results"]}
    if table:
        results["niah"] = table


def collect_sessions(evals, results):
    table = {path.stem[len("profile_") :]: read_json(path) for path in sorted(evals.glob("profile_*.json"))}
    if table:
        results["sessions"] = table


def copy_raw(args):
    """把本脚本读取的原始评测输出复制到报告目录，供读者核对（曲线和吞吐的原始点已在 results.json 中）。"""
    groups = {
        "evals": [path for path in args.evals.glob("*.json") if not path.name.startswith(("curve_", "latency_"))],
        "bench": (args.runs / "bench").glob("*.json"),
        "benchmarks": args.benchmarks.glob("*.json"),
        "extended_benchmarks": (args.extended_benchmarks).glob("*.json"),
        "benchmarks/reference": [
            path for path in (args.benchmarks / "reference").glob("*.json") if "limit" not in path.stem
        ],
        "benchmarks/lm-eval": (args.benchmarks / "lm-eval").rglob("results_*.json"),
        "audits": args.audits.glob("*.json"),
        "niah": args.niah.glob("*.json"),
    }
    for name, paths in groups.items():
        target = args.output / "raw" / name
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(paths):
            shutil.copyfile(path, target / path.name)
    pipeline_log = args.runs.parent / "logs" / "pipeline_evals.log"
    if pipeline_log.exists():
        target = args.output / "raw" / "logs"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pipeline_log, target / pipeline_log.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成技术报告图表")
    parser.add_argument("--runs", type=Path, default=Path("outputs"))
    parser.add_argument("--evals", type=Path, default=Path("outputs/evals"))
    parser.add_argument("--benchmarks", type=Path, default=Path("outputs/benchmarks-v2"))
    parser.add_argument("--extended-benchmarks", type=Path, default=Path("outputs/benchmarks-extended"))
    parser.add_argument("--audits", type=Path, default=Path("outputs/audits"))
    parser.add_argument("--niah", type=Path, default=Path("outputs/niah"))
    parser.add_argument("--data", type=Path, default=Path("data/tokens"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets/report"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    pg19 = load_arms(args.evals, "main_{arm}__pg19test_w4096", ("full", "swa", "ck", "ttt", "ttt_noadapt"))
    fineweb = load_arms(args.evals, "main_{arm}__fineweb_w4096", ("full", "swa", "ck", "ttt", "ttt_noadapt"))
    long_records, extrapolated = collect_long_context(args.evals, results)
    table = collect_ablations(args.evals, args.runs, results)
    collect_dynamic(args.evals, results)
    collect_benchmarks(args.benchmarks, results)
    collect_extended_benchmarks(args.extended_benchmarks, results)
    collect_audits(args.audits, args.data, results)
    collect_provenance(args.runs, results)
    collect_interruptions(args.runs, results)
    collect_evaluation_failures(args.runs, results)
    collect_sessions(args.evals, results)
    collect_niah(args.niah, results)
    copy_raw(args)
    for mode, theme in THEMES.items():
        scratch = results if mode == "light" else {}
        figure_training(theme, mode, args.runs, args.evals, args.output, scratch)
        figure_positions(
            theme,
            mode,
            pg19,
            args.output,
            scratch,
            name="fig_context_4k",
            title="PG-19 test loss by amount of context",
            ranges=RANGES_4K,
        )
        figure_positions(
            theme,
            mode,
            fineweb,
            args.output,
            scratch,
            name="fig_context_4k_fineweb",
            title="FineWeb-Edu held-out loss by position",
            ranges=RANGES_4K,
        )
        figure_long_context(theme, mode, long_records, extrapolated, args.output, scratch)
        figure_recall(theme, mode, args.evals, args.output, scratch)
        figure_ablations(theme, mode, table, args.output)
        figure_scaling(theme, mode, {**results, **scratch}, args.output)
        figure_cost(theme, mode, args.runs, args.evals, args.output, scratch)
    (args.output / "results.json").write_bytes((json.dumps(results, indent=1) + "\n").encode("utf-8"))
    summary = {"figures": sorted(path.name for path in args.output.glob("fig_*.png")), "sections": sorted(results)}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
