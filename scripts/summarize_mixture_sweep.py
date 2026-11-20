#!/usr/bin/env python3
"""汇总数据配比对照实验：每个配方在各领域 held-out 上的 bits-per-byte 和零样本 benchmark 成绩。

读取 ``run_mixture_sweep.sh`` 写出的 ``<配方>[-s<种子>]__domains.json``、``__dolma3.json`` 和
``__benchmarks.json``。同一配方的多个种子取平均，并给出样本标准差；另外把所有多种子配方的方差
合并成一个“种子噪声”，用来判断单种子配方之间的差别是否可信。某次运行没有的指标显示为 ``-``。
只依赖标准库。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

RUN_NAME = re.compile(r"^(?P<mix>.+?)(?:-s(?P<seed>\d+))?$")
# held-out 结果文件的后缀 -> 指标前缀和表格标题。
DOMAIN_FILES = {
    "domains": ("bpb", "held-out bits-per-byte（越低越好）"),
    "dolma3": ("d3", "Dolma 3 held-out bits-per-byte（越低越好）"),
}


def headline(result: dict) -> tuple[str, float]:
    """每个任务取一个主指标：多选题用（归一化）准确率，滚动困惑度任务用 word perplexity。"""
    for key in ("acc_norm", "acc", "word_perplexity"):
        if key in result:
            return key, result[key]
    raise ValueError("结果里没有可用的主指标。")


def load_run(results: Path, run: str) -> tuple[dict, dict]:
    """一次运行的全部指标，以及每个 benchmark 用的是哪种主指标。"""
    metrics, kinds = {}, {}
    for suffix, (prefix, _) in DOMAIN_FILES.items():
        path = results / f"{run}__{suffix}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            for name, value in payload["domains"].items():
                metrics[f"{prefix}/{name}"] = value["bits_per_byte"]
            metrics[f"{prefix}/macro"] = payload["macro_bits_per_byte"]
    path = results / f"{run}__benchmarks.json"
    if path.exists():
        for task, result in json.loads(path.read_text(encoding="utf-8"))["results"].items():
            kinds[task], metrics[f"bench/{task}"] = headline(result)
    return metrics, kinds


def table(header, rows) -> str:
    rule = "|" + "|".join("---" if column == 0 else "---:" for column in range(len(header))) + "|"
    return "\n".join(["| " + " | ".join(header) + " |", rule, *("| " + " | ".join(row) + " |" for row in rows)])


def aggregate(values: list[float]) -> dict:
    spread = statistics.stdev(values) if len(values) > 1 else None
    return {"mean": statistics.fmean(values), "sd": spread, "n": len(values), "values": values}


def pooled_sd(groups: list[list[float]]):
    """把各配方的种子方差按自由度合并。"""
    freedom = sum(len(group) - 1 for group in groups)
    if not freedom:
        return None
    return math.sqrt(sum(statistics.variance(group) * (len(group) - 1) for group in groups if len(group) > 1) / freedom)


def cell(stats: dict | None, scale: float, digits: int) -> str:
    if stats is None:
        return "-"
    text = f"{scale * stats['mean']:.{digits}f}"
    return text if stats["sd"] is None else f"{text} ±{scale * stats['sd']:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总数据配比对照实验")
    parser.add_argument("--results", type=Path, default=Path("outputs/mixture_sweep"))
    parser.add_argument("--recipes", type=Path, default=Path("configs/data/sweep"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs, kinds = {}, {}
    for path in sorted(args.results.glob("*__domains.json")):
        run = path.name.split("__")[0]
        match = RUN_NAME.match(run)
        metrics, found = load_run(args.results, run)
        kinds.update(found)
        runs.setdefault(match["mix"], {})[int(match["seed"] or 0)] = metrics
    if not runs:
        raise SystemExit("没有找到结果文件。")

    summary = {}
    for mix, seeds in runs.items():
        order = sorted(seeds)
        # 只汇总每个种子都有的指标。
        shared = [key for key in seeds[order[0]] if all(key in seeds[seed] for seed in order)]
        stats = {key: aggregate([seeds[seed][key] for seed in order]) for key in shared}
        summary[mix] = {"seeds": order, "metrics": stats}
        recipe = args.recipes / f"{mix}.json"
        if recipe.exists():
            sources = json.loads(recipe.read_text(encoding="utf-8"))["sources"]
            summary[mix]["probabilities"] = {source["name"]: source["probability"] for source in sources}
    order = sorted(summary, key=lambda mix: summary[mix]["metrics"]["bpb/macro"]["mean"])
    # 列的顺序取自指标最全的那次运行，其余运行多出来的指标排在后面。
    keys = list(max(summary.values(), key=lambda entry: len(entry["metrics"]))["metrics"])
    keys += [key for mix in order for key in summary[mix]["metrics"] if key not in keys]
    noise = {
        key: pooled_sd([summary[mix]["metrics"][key]["values"] for mix in order if key in summary[mix]["metrics"]])
        for key in keys
    }

    def scale(key: str) -> float:
        """准确率换算成百分数；bits-per-byte 和困惑度保持原值。"""
        return 100.0 if key.startswith("bench/") and not kinds[key.split("/", 1)[1]].endswith("perplexity") else 1.0

    def render(prefix: str, digits: int) -> str | None:
        columns = [key for key in keys if key.startswith(prefix)]
        rows = []
        for mix in order:
            stats = summary[mix]["metrics"]
            if any(key in stats for key in columns):
                cells = (cell(stats.get(key), scale(key), digits) for key in columns)
                rows.append([mix, str(len(summary[mix]["seeds"])), *cells])
        if not rows:
            return None
        if any(noise[key] is not None for key in columns):
            cells = ("-" if noise[key] is None else f"{scale(key) * noise[key]:.{digits}f}" for key in columns)
            rows.append(["种子噪声（合并 sd）", "", *cells])
        return table(["mixture", "seeds", *(key.split("/", 1)[1] for key in columns)], rows)

    sections = [(title, render(f"{prefix}/", 3)) for prefix, title in DOMAIN_FILES.values()]
    sections.append(("zero-shot benchmarks（准确率 %；wikitext 为 word perplexity，越低越好）", render("bench/", 1)))
    print("\n\n".join(f"{title}\n\n{body}" for title, body in sections if body))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"mixtures": summary, "pooled_seed_sd": noise}
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
