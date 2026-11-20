#!/usr/bin/env python3
"""生成数据配比的候选配方（只依赖标准库）。

``configs/data/sweep/`` 下每个文件是一个候选，数值是训练窗口被抽到的概率。来源有两份语料：

- v2（``configs/data/corpus_v2.json``）：自己挑选的九个来源，配比是这里手工给出的；
- dolma3（``configs/data/corpus_dolma3.json``）：Olmo 3 公开预训练配比的均匀抽样。各来源用同一个
  抽样率，所以按各来源的 token 数加权就是原配比；token 数取自构建后的汇总文件。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COMPONENTS = ("web_edu", "web", "wiki", "books", "papers", "math_web", "math_problems", "code", "qa")
DOLMA3 = ("d3_web", "d3_pdfs", "d3_code", "d3_math", "d3_arxiv", "d3_wiki")
# 来源名 -> 所在语料（data/tokens/<语料>/<来源>-train）。
CORPUS = {**dict.fromkeys(COMPONENTS, "v2"), **dict.fromkeys(DOLMA3, "dolma3")}

# 各候选只改变来源概率。领域粗分为：网页（web_edu + web）、百科、书籍、论文、数学、代码与技术问答。
MIXTURES = {
    # 上一版主实验的配比：教育类网页和长篇书籍各半（这里的数据池换成了 v2 的）。
    "v1_books50": {"web_edu": 0.50, "books": 0.50},
    # 只用教育类网页。
    "web_only": {"web_edu": 1.00},
    # Marin 小规模 scaling suite 的比例：网页 97.2%、代码 2.3%、数学 0.5%。
    "web97": {"web_edu": 0.600, "web": 0.372, "code": 0.023, "math_web": 0.005},
    # 网页为主，少量专门领域数据。
    "v2_light": {
        "web_edu": 0.45,
        "web": 0.20,
        "wiki": 0.10,
        "books": 0.08,
        "papers": 0.04,
        "math_web": 0.03,
        "math_problems": 0.02,
        "code": 0.06,
        "qa": 0.02,
    },
    # 多领域配比：网页 42%、百科 10%、书籍 12%、论文 8%、数学 7%、代码 15%、技术问答 6%。
    # 参考了 Marin hero run 主阶段各主题的占比，但代码类低于它的约 30%（见 v2_code30）。
    "v2": {
        "web_edu": 0.26,
        "web": 0.16,
        "wiki": 0.10,
        "books": 0.12,
        "papers": 0.08,
        "math_web": 0.04,
        "math_problems": 0.03,
        "code": 0.15,
        "qa": 0.06,
    },
    # 代码与技术问答合计 30%，对应 Marin hero run 主阶段软件类主题的总占比。
    "v2_code30": {
        "web_edu": 0.20,
        "web": 0.13,
        "wiki": 0.09,
        "books": 0.12,
        "papers": 0.08,
        "math_web": 0.04,
        "math_problems": 0.04,
        "code": 0.24,
        "qa": 0.06,
    },
    # 九个来源等概率。
    "uniform": {name: 1.0 for name in COMPONENTS},
}


def recipe(weights: dict) -> dict:
    unknown = set(weights) - set(CORPUS)
    if unknown:
        raise ValueError(f"未知来源：{sorted(unknown)}")
    total = sum(weights.values())
    return {
        "format": "ttt_mixture_v1",
        "selection_unit": "window",
        "exhaustion_policy": "with_replacement",
        "sources": [
            {
                "name": name,
                "data": f"../../../data/tokens/{corpus}/{name}-train/manifest.json",
                "probability": round(weights[name] / total, 6),
            }
            for name, corpus in CORPUS.items()
            if name in weights
        ],
    }


def dolma3_mixtures(report: Path) -> dict:
    """Dolma 3 抽样语料的配方；语料还没有构建（没有汇总文件）时返回空。"""
    if not report.exists():
        return {}
    sources = json.loads(report.read_text(encoding="utf-8"))["sources"]
    natural = {name: sources[name]["train"]["tokens"] for name in DOLMA3}
    return {
        # Olmo 3 的原配比。
        "dolma3": natural,
        # 原配比里没有书籍。这一份拿出 10% 给 PG-19，用来衡量书籍评测对训练数据是否同域有多敏感。
        "dolma3_books10": {"books": sum(natural.values()) / 9, **natural},
    }


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成数据配比的候选配方")
    parser.add_argument("--output", type=Path, default=Path("configs/data/sweep"))
    parser.add_argument("--dolma3-report", type=Path, default=Path("configs/data/corpus_dolma3.report.json"))
    args = parser.parse_args()
    mixtures = {**MIXTURES, **dolma3_mixtures(args.dolma3_report)}
    for name, weights in mixtures.items():
        write(args.output / f"{name}.json", recipe(weights))
    print(json.dumps({"written": len(mixtures), "mixtures": list(mixtures)}))


if __name__ == "__main__":
    main()
