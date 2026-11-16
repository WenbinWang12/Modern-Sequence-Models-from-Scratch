"""把训练 JSONL 指标渲染成可发布的 SVG 图。"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Iterable

__all__ = ["render_metrics", "main"]


_WIDTH = 1000
_HEIGHT = 760
_LEFT = 88
_RIGHT = 28
_TOP = 62
_BOTTOM = 46
_PANEL_GAP = 42
_PANEL_HEIGHT = 190
_COLORS = {
    "loss": "#2563eb",
    "tokens_per_second": "#d97706",
    "grad_norm": "#059669",
    "fast_weight_norm": "#9333ea",
}


def _read_metrics(path: Path) -> list[dict[str, float]]:
    """读取 JSONL，并保留拥有 step 的有效记录。"""

    records: list[dict[str, float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or "step" not in value:
            raise ValueError(f"第 {line_number} 行没有 step 字段。")
        records.append({key: float(item) for key, item in value.items() if isinstance(item, (int, float))})
    if not records:
        raise ValueError("指标文件没有可绘制的记录。")
    return records


def _scale(value: float, low: float, high: float, start: float, end: float) -> float:
    if high <= low:
        return (start + end) / 2
    return start + (value - low) / (high - low) * (end - start)


def _bounds(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    low, high = min(values), max(values)
    if low == high:
        padding = max(abs(low) * 0.05, 1.0)
        return low - padding, high + padding
    padding = (high - low) * 0.08
    return low - padding, high + padding


def _line_points(records: list[dict[str, float]], key: str, x0: float, x1: float, y0: float, y1: float) -> str:
    values = [record[key] for record in records if key in record]
    if not values:
        return ""
    steps = [record["step"] for record in records if key in record]
    x_low, x_high = min(record["step"] for record in records), max(record["step"] for record in records)
    y_low, y_high = _bounds(values)
    points = []
    for step, value in zip(steps, values):
        x = _scale(step, x_low, x_high, x0, x1)
        y = _scale(value, y_low, y_high, y1, y0)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _panel(records: list[dict[str, float]], key: str, label: str, top: float) -> str:
    x0, x1 = _LEFT, _WIDTH - _RIGHT
    y0, y1 = top + 30, top + _PANEL_HEIGHT
    values = [record[key] for record in records if key in record]
    if not values:
        return f'<text x="{_LEFT}" y="{top + 24}" class="missing">{html.escape(label)}: no data</text>'
    y_low, y_high = _bounds(values)
    x_low = min(record["step"] for record in records)
    x_high = max(record["step"] for record in records)
    path = _line_points(records, key, x0, x1, y0, y1)
    last = values[-1]
    parts = [
        f'<text x="{_LEFT}" y="{top + 20}" class="panel-title">{html.escape(label)}</text>',
        f'<text x="{x0}" y="{y1 + 25}" class="tick">{x_low:g}</text>',
        f'<text x="{x1}" y="{y1 + 25}" text-anchor="end" class="tick">{x_high:g}</text>',
        f'<text x="{x0 - 10}" y="{y0 + 4}" text-anchor="end" class="tick">{y_high:.3g}</text>',
        f'<text x="{x0 - 10}" y="{y1 + 4}" text-anchor="end" class="tick">{y_low:.3g}</text>',
        f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="axis"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="axis"/>',
        f'<polyline points="{path}" fill="none" stroke="{_COLORS[key]}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>',
        f'<circle cx="{_scale(x_high, x_low, x_high, x0, x1):.2f}" cy="{_scale(last, y_low, y_high, y1, y0):.2f}" r="4" fill="{_COLORS[key]}"/>',
        f'<text x="{x1}" y="{top + 20}" text-anchor="end" class="value">last {last:.4g}</text>',
    ]
    return "\n".join(parts)


def render_metrics(input_path: Path, output_path: Path) -> dict[str, object]:
    """读取指标并写入一个独立 SVG，返回报告摘要。"""

    records = _read_metrics(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("loss", "Loss / NLL"),
        ("tokens_per_second", "Training throughput (tokens/s)"),
        ("grad_norm", "Gradient norm"),
        ("fast_weight_norm", "Fast-weight norm"),
    ]
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" height="{_HEIGHT}" viewBox="0 0 {_WIDTH} {_HEIGHT}" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">Training metrics</title>",
        f'<desc id="desc">Training metrics from step {records[0]["step"]:g} through {records[-1]["step"]:g}.</desc>',
        "<style>text{font-family:Inter,Arial,sans-serif;fill:#334155}.panel-title{font-size:18px;font-weight:600}.tick{font-size:12px;fill:#64748b}.value{font-size:13px;fill:#334155}.missing{font-size:16px;fill:#b91c1c}.axis{stroke:#cbd5e1;stroke-width:1}</style>",
        f'<rect width="{_WIDTH}" height="{_HEIGHT}" fill="#f8fafc"/>',
        '<text x="88" y="34" class="panel-title">TTT-E2E training metrics</text>',
    ]
    for index, (key, label) in enumerate(panels):
        content.append(_panel(records, key, label, _TOP + index * (_PANEL_HEIGHT + _PANEL_GAP)))
    content.append("</svg>\n")
    output_path.write_text("\n".join(content), encoding="utf-8")
    return {
        "input": str(input_path),
        "output": str(output_path),
        "records": len(records),
        "first_step": records[0]["step"],
        "last_step": records[-1]["step"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把训练 JSONL 指标渲染成 SVG")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(render_metrics(args.input, args.output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
