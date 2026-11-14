"""在网站发布时，把教材 SVG 导出为可下载的 PNG 和矢量 PDF。"""

import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import cairosvg

FIGURES = ("cleaning-views", "dedup-candidates", "bpe-learning-vs-encoding")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", type=Path, required=True)
    args = parser.parse_args()
    figures_dir = args.docs_dir.resolve() / "assets" / "figures"
    output = figures_dir / "exports"
    output.mkdir(parents=True, exist_ok=True)
    fonts = {}
    for family in ("Noto Sans CJK SC", "DejaVu Sans Mono"):
        # 记录实际匹配到的字体，便于解释不同构建环境下的排版差异。
        result = subprocess.run(
            ["fc-match", "-f", "%{family}: %{style}", family],
            check=True, capture_output=True, text=True,
        )
        fonts[family] = result.stdout.strip()
    report = {
        "purpose": "publication assets; not model or data test results",
        "python": platform.python_version(),
        "libraries": {package: version(package) for package in (
            "CairoSVG", "cairocffi", "cssselect2", "tinycss2", "Pillow",
        )},
        "fonts": fonts,
        "png_scale": 2,
        "pdf_origin": "SVG vector export, not a compiled TikZ document",
        "figures": [],
    }
    for name in FIGURES:
        for suffix in ("", "-zh"):
            stem = name + suffix
            source = figures_dir / f"{stem}.svg"
            png = output / f"{stem}.png"
            pdf = output / f"{stem}.pdf"
            cairosvg.svg2png(url=str(source), write_to=str(png), scale=2)
            cairosvg.svg2pdf(url=str(source), write_to=str(pdf))
            report["figures"].append({
                "source": source.name, "source_sha256": sha256(source),
                "png": png.name, "png_sha256": sha256(png),
                "pdf": pdf.name, "pdf_sha256": sha256(pdf),
            })
            print(f"exported {stem}: PNG + PDF", flush=True)
    (output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
