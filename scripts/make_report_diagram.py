#!/usr/bin/env python3
"""绘制技术报告的方法示意图（纯标准库，输出浅色和深色两份 SVG）。

图中从左到右是同一条序列的三个更新块：前段各层对整条序列只算一次；后段各层逐块前向，
每块先用当前快权重预测并得到损失，再用这个损失对快权重做一步梯度更新，更新后的快权重
只用于后面的块。外循环对所有块的损失求和，并对整条更新链求导。
"""

from __future__ import annotations

import argparse
from pathlib import Path

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "panel": "#f3f2ee",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "line": "#c3c2b7",
        "fast": "#1baf7a",
        "fast_fill": "#e3f4ed",
        "outer": "#2a78d6",
    },
    "dark": {
        "surface": "#1a1a19",
        "panel": "#262624",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "line": "#4a4a46",
        "fast": "#199e70",
        "fast_fill": "#1d3229",
        "outer": "#3987e5",
    },
}
FONT = "system-ui, -apple-system, 'Segoe UI', 'Noto Sans', 'DejaVu Sans', sans-serif"
WIDTH, HEIGHT = 1010, 476
PILL, BOX, GAP = 54, 190, 28


def text(x, y, value, *, size=13, fill, anchor="middle", weight=400):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}" dominant-baseline="middle">{value}</text>'
    )


def rect(x, y, width, height, *, fill, stroke, radius=8, stroke_width=1.2):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def arrow(x1, y1, x2, y2, *, color, marker, width=1.6):
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
        f'stroke-width="{width}" marker-end="url(#{marker})"/>'
    )


def render(theme) -> str:
    ink, ink2, line, fast, outer = theme["ink"], theme["ink2"], theme["line"], theme["fast"], theme["outer"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" '
        f'height="{HEIGHT}" font-family="{FONT}" role="img" '
        'aria-label="TTT-E2E forward pass: prefix blocks run once, suffix blocks run chunk by chunk and '
        'update their fast weights after each chunk; the outer loop differentiates through all updates">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{theme["surface"]}"/>',
        "<defs>",
    ]
    for name, tint in (("slow", line), ("fast", fast), ("outer", outer)):
        parts.append(
            f'<marker id="{name}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0 0.8 L9 5 L0 9.2 z" fill="{tint}"/></marker>'
        )
    parts.append("</defs>")

    left = 28
    pill_x = [left + index * (PILL + GAP + BOX + GAP) for index in range(4)]
    box_x = [left + PILL + GAP + index * (BOX + GAP + PILL + GAP) for index in range(3)]
    loss_y, suffix_y, suffix_h, prefix_y, chunk_y = 92, 138, 84, 268, 364
    middle = suffix_y + suffix_h / 2
    subscript = "₀₁₂₃"

    # 外循环：所有块的损失相加，梯度沿整条链回传。
    first, last = box_x[0] + BOX / 2, box_x[2] + BOX / 2
    parts.append(f'<path d="M{first:.1f} 58 V44 H{last:.1f} V58" fill="none" stroke="{outer}" stroke-width="1.6"/>')
    parts.append(
        f'<line x1="{box_x[1] + BOX / 2:.1f}" y1="44" x2="{box_x[1] + BOX / 2:.1f}" y2="58" stroke="{outer}" stroke-width="1.6"/>'
    )
    parts.append(
        text(
            (first + last) / 2,
            24,
            "Outer loop: θ ← θ − α ∇θ (ℓ₁ + ℓ₂ + ℓ₃ + …), differentiating through every fast-weight update",
            size=13.5,
            fill=ink,
            weight=600,
        )
    )

    for index in range(4):
        x = pill_x[index]
        parts.append(rect(x, middle - 17, PILL, 34, fill=theme["fast_fill"], stroke=fast, radius=17, stroke_width=1.6))
        parts.append(text(x + PILL / 2, middle + 1, f"φ{subscript[index]}", size=15, fill=ink, weight=600))
    parts.append(text(pill_x[0] + PILL / 2, middle + 34, "learned", size=11.5, fill=ink2))
    parts.append(text(pill_x[0] + PILL / 2, middle + 48, "init ⊂ θ", size=11.5, fill=ink2))
    parts.append(text(pill_x[3] + PILL + 16, middle + 1, "⋯", size=20, fill=ink2))

    for index in range(3):
        x = box_x[index]
        centre = x + BOX / 2
        number = subscript[index + 1]
        parts.append(rect(x, suffix_y, BOX, suffix_h, fill=theme["panel"], stroke=line))
        parts.append(text(centre, suffix_y + 22, "Suffix blocks", size=13.5, fill=ink, weight=600))
        parts.append(text(centre, suffix_y + 43, "attention → fast MLP′ → MLP", size=11.5, fill=ink2))
        parts.append(
            text(centre, suffix_y + 62, f"predict chunk {index + 1} with φ{subscript[index]}", size=11.5, fill=ink2)
        )
        # 快权重进入本块；本块的损失梯度产生下一组快权重。
        parts.append(arrow(pill_x[index] + PILL + 2, middle, x - 3, middle, color=fast, marker="fast"))
        parts.append(arrow(x + BOX + 2, middle, pill_x[index + 1] - 3, middle, color=fast, marker="fast"))
        update = f"φ{subscript[index]} − η ĝ{number}"
        parts.append(text(pill_x[index + 1] + PILL / 2, middle + 34, "=", size=11.5, fill=ink2))
        parts.append(text(pill_x[index + 1] + PILL / 2, middle + 49, update, size=11.5, fill=ink2))
        # 损失节点。
        parts.append(
            f'<circle cx="{centre:.1f}" cy="{loss_y - 16}" r="16" fill="{theme["surface"]}" stroke="{outer}" stroke-width="1.6"/>'
        )
        parts.append(text(centre, loss_y - 15, f"ℓ{number}", size=14, fill=ink, weight=600))
        parts.append(arrow(centre, suffix_y - 2, centre, loss_y + 3, color=line, marker="slow"))
        # 前段特征进入后段，输入进入前段。
        parts.append(arrow(centre, prefix_y - 2, centre, suffix_y + suffix_h + 4, color=line, marker="slow"))
        parts.append(arrow(centre, chunk_y - 2, centre, prefix_y + 56, color=line, marker="slow"))
        parts.append(rect(x, chunk_y, BOX, 40, fill=theme["surface"], stroke=line))
        parts.append(text(centre, chunk_y + 21, f"chunk {index + 1}  ·  C tokens", size=12.5, fill=ink))

    prefix_x, prefix_w = box_x[0], box_x[2] + BOX - box_x[0]
    parts.append(rect(prefix_x, prefix_y, prefix_w, 52, fill=theme["panel"], stroke=line))
    parts.append(
        text(
            prefix_x + prefix_w / 2, prefix_y + 18, "Prefix blocks (slow weights only)", size=13.5, fill=ink, weight=600
        )
    )
    parts.append(
        text(
            prefix_x + prefix_w / 2,
            prefix_y + 37,
            "sliding-window attention + MLP, computed once for the whole sequence",
            size=11.5,
            fill=ink2,
        )
    )

    # 图例：色块在前，文字用墨色。
    legend_y = 440
    parts.append(
        f'<line x1="{left}" y1="{legend_y}" x2="{left + 26}" y2="{legend_y}" stroke="{fast}" stroke-width="2.4"/>'
    )
    parts.append(
        text(
            left + 34,
            legend_y + 1,
            "Inner loop (also runs at test time): after each chunk, one SGD step with ĝ = ∇φ ℓ clipped to norm τ",
            size=12,
            fill=ink2,
            anchor="start",
        )
    )
    parts.append(
        f'<line x1="{left}" y1="{legend_y + 20}" x2="{left + 26}" y2="{legend_y + 20}" stroke="{outer}" stroke-width="2.4"/>'
    )
    parts.append(
        text(
            left + 34,
            legend_y + 21,
            "Outer loop (training only): next-token loss summed over chunks, back-propagated through the inner steps",
            size=12,
            fill=ink2,
            anchor="start",
        )
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制方法示意图")
    parser.add_argument("--output", type=Path, default=Path("docs/assets/report"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for mode, theme in THEMES.items():
        (args.output / f"fig_method.{mode}.svg").write_bytes(render(theme).encode("utf-8"))
    print("wrote fig_method.light.svg and fig_method.dark.svg")


if __name__ == "__main__":
    main()
