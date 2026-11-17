#!/usr/bin/env python3
"""把技术报告模板渲染成网站页面：所有数字都从 ``results.json`` 读取。

模板语法只有三种：

- ``{{ 表达式 }}``：在一个只包含结果数据和格式化函数的命名空间里求值；
- ``{% table 名称 %}``：插入由本脚本生成的 Markdown 表格；
- ``{% if 表达式 %} … {% else %} … {% endif %}``：按数据是否已经产生来选择段落（不支持嵌套）。

中英文模板共用同一份 ``results.json``，因此两种语言里的数字一定相同。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ARM_ORDER = ("full", "swa", "ck", "ttt_noadapt", "ttt")
ARM_NAMES = {
    "zh": {
        "full": "全注意力",
        "swa": "滑窗注意力",
        "ck": "无更新对照",
        "ttt_noadapt": "TTT-E2E（关闭更新）",
        "ttt": "TTT-E2E",
        "full_sw4096": "全注意力（只看最近 4K）",
    },
    "en": {
        "full": "Full attention",
        "swa": "Sliding window",
        "ck": "No-update control",
        "ttt_noadapt": "TTT-E2E (updates off)",
        "ttt": "TTT-E2E",
        "full_sw4096": "Full attention (last 4K only)",
    },
}
TASKS = (
    ("hellaswag", "HellaSwag", "acc_norm"),
    ("piqa", "PIQA", "acc_norm"),
    ("arc_easy", "ARC-E", "acc_norm"),
    ("arc_challenge", "ARC-C", "acc_norm"),
    ("winogrande", "WinoGrande", "acc"),
    ("sciq", "SciQ", "acc"),
    ("openbookqa", "OBQA", "acc_norm"),
    ("blimp", "BLiMP", "acc"),
    ("mmlu", "MMLU", "acc"),
    ("lambada_openai", "LAMBADA", "acc"),
)
REFERENCE_MODELS = (
    ("distilgpt2", "DistilGPT-2"),
    ("gpt2", "GPT-2"),
    ("gpt2-medium", "GPT-2 Medium"),
    ("EleutherAI_pythia-70m", "Pythia-70M"),
    ("EleutherAI_pythia-160m", "Pythia-160M"),
    ("EleutherAI_gpt-neo-125m", "GPT-Neo-125M"),
    ("facebook_opt-125m", "OPT-125M"),
    ("HuggingFaceTB_SmolLM-135M", "SmolLM-135M"),
    ("HuggingFaceTB_SmolLM2-135M", "SmolLM2-135M"),
)
OUR_MODELS = (
    ("main_full", "full"),
    ("main_swa", "swa"),
    ("main_ck", "ck"),
    ("main_ttt_noadapt", "ttt_noadapt"),
    ("main_ttt", "ttt"),
)


class Box:
    """让模板可以用点号访问嵌套字典；缺少的键返回空 Box，便于写 ``{% if %}``。"""

    def __init__(self, value):
        self._value = value

    def __getattr__(self, name):
        return self[name]

    def __getitem__(self, name):
        value = self._value
        if isinstance(value, dict) and str(name) in value:
            return wrap(value[str(name)])
        if isinstance(value, list) and isinstance(name, int):
            return wrap(value[name])
        return Box(None)

    def __bool__(self):
        return bool(self._value)

    def __iter__(self):
        return iter(self._value or ())

    def __len__(self):
        return len(self._value or ())

    def keys(self):
        return list(self._value) if isinstance(self._value, dict) else []

    def raw(self):
        return self._value

    def __repr__(self):
        raise ValueError("模板引用了不存在或不是标量的字段。")


def wrap(value):
    return Box(value) if isinstance(value, (dict, list)) or value is None else value


def minus(text):
    return text.replace("-", "−")


def fixed(digits):
    return lambda value: minus(f"{value:.{digits}f}")


def signed(value, digits=3):
    return minus(f"{value:+.{digits}f}")


def interval(pair, digits=3):
    low, high = (pair.raw() if isinstance(pair, Box) else pair)[:2]
    return minus(f"[{low:+.{digits}f}, {high:+.{digits}f}]")


def percent(value, digits=1):
    return f"{100 * value:.{digits}f}"


def millions(value, digits=1):
    return f"{value / 1e6:.{digits}f}M"


def billions(value, digits=2):
    return f"{value / 1e9:.{digits}f}B"


def thousands(value, digits=1):
    return f"{value / 1e3:.{digits}f}K"


def gib(value, digits=1):
    return f"{value / 2**30:.{digits}f}"


def grouped(value):
    return f"{int(round(value)):,}"


def markdown_table(header, rows, align=None):
    align = align or ["l"] + ["r"] * (len(header) - 1)
    rule = [":--" if side == "l" else "--:" for side in align]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(rule) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def bold_best(cells, values, *, lowest=True):
    """把一列里最好的数值加粗；values 中的 None 不参与比较。"""
    present = [value for value in values if value is not None]
    if not present:
        return cells
    best = min(present) if lowest else max(present)
    return [f"**{cell}**" if value is not None and value == best else cell for cell, value in zip(cells, values)]


# --------------------------------------------------------------------------- 表格


def table_main(r, lang):
    names = ARM_NAMES[lang]
    pg19, fineweb = r["fig_context_4k"], r["fig_context_4k_fineweb"]
    arms = [arm for arm in ARM_ORDER if pg19["arms"][arm]]
    reference = "full" if "full" in arms else arms[0]
    header = {
        "zh": [
            "模型",
            "PG-19 test NLL",
            "PPL",
            "bits/byte",
            f"相对{names[reference]}的差值（95% CI）",
            "FineWeb-Edu NLL",
            "PPL",
        ],
        "en": [
            "Model",
            "PG-19 test NLL",
            "PPL",
            "bits/byte",
            f"Δ vs {names[reference].lower()} (95% CI)",
            "FineWeb-Edu NLL",
            "PPL",
        ],
    }[lang]
    f3, f2 = fixed(3), fixed(2)
    columns = {
        "nll": [pg19["arms"][arm]["nll"] for arm in arms],
        "fw": [fineweb["arms"][arm]["nll"] if fineweb["arms"][arm] else None for arm in arms],
    }
    nll_cells = bold_best([f3(value) for value in columns["nll"]], columns["nll"])
    fw_cells = bold_best([f3(value) if value is not None else "–" for value in columns["fw"]], columns["fw"])
    rows = []
    for index, arm in enumerate(arms):
        entry = pg19["arms"][arm]
        if arm == reference:
            delta = "–"
        else:
            pair = pg19["paired"][f"{arm}_minus_{reference}"]["all"]
            delta = f"{signed(pair['mean'])} {interval(pair['ci95'])}"
        fw = fineweb["arms"][arm]
        rows.append(
            [
                names[arm],
                nll_cells[index],
                f2(entry["perplexity"]),
                f3(entry["bits_per_byte"]),
                delta,
                fw_cells[index],
                f2(fw["perplexity"]) if fw else "–",
            ]
        )
    return markdown_table(header, rows)


def range_label(key, lang):
    begin, end = (int(part) for part in key.split("-"))

    def short(value):
        return f"{value // 1024}K" if value >= 1024 and value % 1024 == 0 else str(value)

    return f"{short(begin)}–{short(end)}"


def table_ranges(
    r, lang, section="fig_context_4k", arms=ARM_ORDER, comparisons=(("ttt", "swa"), ("ttt", "full"), ("swa", "full"))
):
    names = ARM_NAMES[lang]
    data = r[section]
    present = [arm for arm in arms if data["arms"][arm]]
    keys = data["arms"][present[0]]["by_range"].keys()
    first = {"zh": "上下文位置（token）", "en": "Context position (tokens)"}[lang]
    header = [first] + [range_label(key, lang) for key in keys] + [{"zh": "全部", "en": "All"}[lang]]
    f3 = fixed(3)
    rows = []
    for arm in present:
        entry = data["arms"][arm]
        rows.append([names[arm]] + [f3(entry["by_range"][key]["mean"]) for key in keys] + [f3(entry["nll"])])
    for first_arm, second_arm in comparisons:
        pair = data["paired"][f"{first_arm}_minus_{second_arm}"]
        if not pair:
            continue
        label = f"Δ {names[first_arm]} − {names[second_arm]}"
        cells = [f"{signed(pair[key]['mean'])}<br>{interval(pair[key]['ci95'])}" for key in keys]
        cells.append(f"{signed(pair['all']['mean'])}<br>{interval(pair['all']['ci95'])}")
        rows.append([label] + cells)
    return markdown_table(header, rows)


def table_long(r, lang):
    arms = ("full_sw4096", "swa", "ck", "ttt_noadapt", "ttt")
    comparisons = (("ttt", "swa"), ("ttt", "full_sw4096"), ("ttt", "ttt_noadapt"))
    return table_ranges(r, lang, "fig_context_32k", arms, comparisons)


def benchmark_cell(entry, task, metric):
    values = entry["tasks"][task]
    if not values:
        return None
    return values[metric]


def table_benchmarks(r, lang, which="all"):
    names = ARM_NAMES[lang]
    bench = r["benchmarks"]
    header = [{"zh": "模型", "en": "Model"}[lang], {"zh": "参数量", "en": "Params"}[lang]] + [
        label for _, label, _ in TASKS
    ]
    header.append({"zh": "平均", "en": "Mean"}[lang])
    rows, groups = [], []
    if which in ("all", "ours"):
        groups.append([(key, names[arm], True) for key, arm in OUR_MODELS if bench[key]])
    if which in ("all", "reference"):
        groups.append([(key, label, False) for key, label in REFERENCE_MODELS if bench[key]])
    for group in groups:
        values = {key: [benchmark_cell(bench[key], task, metric) for task, _, metric in TASKS] for key, _, _ in group}
        for key, label, ours in group:
            entry = bench[key]
            if ours:
                parameters = r["training_cost"][{"ck": "ck", "ttt_noadapt": "ttt"}.get(key[5:], key[5:])]["parameters"]
            else:
                parameters = entry["model"]["parameters"]
            cells = [percent(value) if value is not None else "–" for value in values[key]]
            present = [value for value in values[key] if value is not None]
            mean = percent(sum(present) / len(present)) if len(present) == len(TASKS) else "–"
            rows.append([label, millions(parameters, 0)] + cells + [mean])
    return markdown_table(header, rows)


def table_perplexity(r, lang):
    names = ARM_NAMES[lang]
    bench = r["benchmarks"]
    header = {
        "zh": [
            "模型",
            "最大上下文",
            "WikiText-103 词级 PPL",
            "WikiText-103 bits/byte",
            "PG-19 test 词级 PPL",
            "PG-19 test bits/byte",
            "LAMBADA PPL",
        ],
        "en": [
            "Model",
            "Max context",
            "WikiText-103 word PPL",
            "WikiText-103 bits/byte",
            "PG-19 test word PPL",
            "PG-19 test bits/byte",
            "LAMBADA PPL",
        ],
    }[lang]
    f2, f3 = fixed(2), fixed(3)
    rows = []
    models = [(key, names[arm]) for key, arm in OUR_MODELS] + list(REFERENCE_MODELS)
    for key, label in models:
        entry = bench[key]
        if not entry:
            continue
        wiki, books, lambada = entry["tasks"]["wikitext"], entry["tasks"]["pg19"], entry["tasks"]["lambada_openai"]
        rows.append(
            [
                label,
                grouped(entry["model"]["max_length"]),
                f2(wiki["word_perplexity"]) if wiki else "–",
                f3(wiki["bits_per_byte"]) if wiki else "–",
                f2(books["word_perplexity"]) if books else "–",
                f3(books["bits_per_byte"]) if books else "–",
                f2(lambada["perplexity"]) if lambada else "–",
            ]
        )
    return markdown_table(header, rows)


def table_crosscheck(r, lang):
    ours = r["benchmarks"]["gpt2"]["tasks"]
    reference = r["lm_eval_crosscheck"]["results"]
    header = {
        "zh": ["任务", "指标", "lm-evaluation-harness", "本仓库评测脚本", "差值"],
        "en": ["Task", "Metric", "lm-evaluation-harness", "This repository", "Difference"],
    }[lang]
    pairs = [
        ("hellaswag", "acc", "acc,none"),
        ("hellaswag", "acc_norm", "acc_norm,none"),
        ("piqa", "acc", "acc,none"),
        ("piqa", "acc_norm", "acc_norm,none"),
        ("arc_easy", "acc", "acc,none"),
        ("arc_easy", "acc_norm", "acc_norm,none"),
        ("arc_challenge", "acc", "acc,none"),
        ("arc_challenge", "acc_norm", "acc_norm,none"),
        ("winogrande", "acc", "acc,none"),
        ("sciq", "acc", "acc,none"),
        ("sciq", "acc_norm", "acc_norm,none"),
        ("openbookqa", "acc", "acc,none"),
        ("openbookqa", "acc_norm", "acc_norm,none"),
        ("lambada_openai", "acc", "acc,none"),
        ("lambada_openai", "perplexity", "perplexity,none"),
        ("wikitext", "word_perplexity", "word_perplexity,none"),
        ("wikitext", "bits_per_byte", "bits_per_byte,none"),
        ("blimp", "macro_acc", "acc,none"),
        ("mmlu", "acc", "acc,none"),
    ]
    f4 = fixed(4)
    rows = []
    for task, metric, key in pairs:
        theirs = reference[task][key]
        mine = ours[task][metric]
        if isinstance(theirs, Box) or isinstance(mine, Box):
            continue
        rows.append(
            [task, metric, f4(theirs), f4(mine), minus(f"{mine - theirs:+.1e}") if abs(mine - theirs) > 5e-7 else "0"]
        )
    return markdown_table(header, rows, ["l", "l", "r", "r", "r"])


def table_cost(r, lang):
    names = ARM_NAMES[lang]
    cost = r["training_cost"]
    header = {
        "zh": ["模型", "总参数", "快权重参数", "训练吞吐（token/s）", "相对滑窗", "每步时间（s）", "峰值显存（GiB）"],
        "en": [
            "Model",
            "Parameters",
            "Fast-weight parameters",
            "Training throughput (tokens/s)",
            "Relative to sliding window",
            "Step time (s)",
            "Peak memory (GiB)",
        ],
    }[lang]
    base = cost["swa"]["tokens_per_second"]
    rows = []
    for arm in ("swa", "ck", "full", "ttt"):
        entry = cost[arm]
        if not entry:
            continue
        rows.append(
            [
                names[arm],
                millions(entry["parameters"]),
                millions(entry["fast_parameters"]) if entry["fast_parameters"] else "–",
                grouped(entry["tokens_per_second"]),
                f"{entry['tokens_per_second'] / base:.2f}×",
                f"{entry['median_step_seconds']:.3f}",
                gib(entry["peak_bytes_in_use"]),
            ]
        )
    return markdown_table(header, rows)


def table_latency(r, lang):
    names = ARM_NAMES[lang]
    data = r["inference_throughput"]
    arms = [arm for arm in ("full", "swa", "ttt") if data[arm]]
    lengths = sorted({point[0] for arm in arms for point in data[arm].raw()})
    header = [{"zh": "上下文长度", "en": "Context length"}[lang]] + [names[arm] for arm in arms]
    rows = []
    for length in lengths:
        cells = []
        for arm in arms:
            match = [point[1] for point in data[arm].raw() if point[0] == length]
            cells.append(grouped(match[0]) if match else "–")
        rows.append([grouped(length)] + cells)
    return markdown_table(header, rows)


def table_recall(r, lang):
    names = ARM_NAMES[lang]
    data = r["recall"]
    arms = [arm for arm in ("full", "swa", "ck", "ttt_noadapt", "ttt") if data[arm]]
    distances = sorted(int(key) for key in data[arms[0]].keys())
    header = [{"zh": "与首次出现的距离（token）", "en": "Distance to first occurrence (tokens)"}[lang]] + [
        grouped(distance) for distance in distances
    ]
    rows = []
    for arm in arms:
        cells = [
            f"{signed(data[arm][str(distance)]['gain'], 2)}<br>{interval(data[arm][str(distance)]['gain_ci95'], 2)}"
            for distance in distances
        ]
        rows.append([names[arm]] + cells)
    return markdown_table(header, rows)


ABLATION_ROWS = (
    ("abl_full", {"zh": "全注意力", "en": "Full attention"}),
    ("abl_swa", {"zh": "滑窗注意力（k=256）", "en": "Sliding window (k=256)"}),
    ("abl_swa_iso", {"zh": "滑窗注意力，训练时间与 TTT 相同", "en": "Sliding window, same training time as TTT"}),
    ("abl_ck", {"zh": "无更新对照", "en": "No-update control"}),
    (
        "abl_ttt",
        {"zh": "TTT-E2E（基准：η=1，C=128，后 2 层，k=256）", "en": "TTT-E2E (base: η=1, C=128, last 2 blocks, k=256)"},
    ),
    (
        "abl_ttt_fo",
        {"zh": "一阶近似（外循环不对内循环梯度求导）", "en": "First-order (no gradient through inner gradients)"},
    ),
    ("abl_ttt_ilr0p1", {"zh": "内循环学习率 η=0.1", "en": "Inner learning rate η=0.1"}),
    ("abl_ttt_ilr0p3", {"zh": "内循环学习率 η=0.3", "en": "Inner learning rate η=0.3"}),
    ("abl_ttt_ilr3", {"zh": "内循环学习率 η=3", "en": "Inner learning rate η=3"}),
    ("abl_ttt_chunk64", {"zh": "更新块 C=64", "en": "Update chunk C=64"}),
    ("abl_ttt_chunk256", {"zh": "更新块 C=256", "en": "Update chunk C=256"}),
    ("abl_ttt_suffix1", {"zh": "只在最后 1 层放快权重", "en": "Fast weights in the last block only"}),
    ("abl_ttt_suffix4", {"zh": "最后 4 层放快权重", "en": "Fast weights in the last 4 blocks"}),
    ("abl_swa_window128", {"zh": "滑窗注意力，k=128", "en": "Sliding window, k=128"}),
    ("abl_ttt_window128", {"zh": "TTT-E2E，k=128", "en": "TTT-E2E, k=128"}),
    ("abl_swa_window512", {"zh": "滑窗注意力，k=512", "en": "Sliding window, k=512"}),
    ("abl_ttt_window512", {"zh": "TTT-E2E，k=512", "en": "TTT-E2E, k=512"}),
)


def table_ablations(r, lang):
    data = r["ablations"]
    header = {
        "zh": [
            "设置",
            "训练 token",
            "PG-19 val NLL",
            "前 256 个 token",
            "最后 512 个 token",
            "关闭更新时",
            "FineWeb-Edu NLL",
            "每步时间（s）",
        ],
        "en": [
            "Setting",
            "Training tokens",
            "PG-19 val NLL",
            "First 256 tokens",
            "Last 512 tokens",
            "Updates off",
            "FineWeb-Edu NLL",
            "Step time (s)",
        ],
    }[lang]
    f3 = fixed(3)
    rows = []
    for key, label in ABLATION_ROWS:
        entry = data[key]
        if not entry:
            continue
        rows.append(
            [
                label[lang],
                millions(entry["tokens_seen"], 0) if entry["tokens_seen"] else "–",
                f3(entry["nll"]),
                f3(entry["nll_first_window"]),
                f3(entry["nll_last_quarter"]),
                f3(entry["noadapt_nll"]) if entry["noadapt_nll"] else "–",
                f3(entry["fineweb_nll"]) if entry["fineweb_nll"] else "–",
                f"{entry['median_step_seconds']:.3f}" if entry["median_step_seconds"] else "–",
            ]
        )
    return markdown_table(header, rows)


def table_seeds(r, lang):
    names = ARM_NAMES[lang]
    data = r["seeds"]
    header = {
        "zh": ["量", "seed 0", "seed 1", "seed 2", "均值", "标准差"],
        "en": ["Quantity", "seed 0", "seed 1", "seed 2", "Mean", "SD"],
    }[lang]
    f3 = fixed(3)
    rows = []
    for arm in ("full", "swa", "ttt"):
        entry = data["arms"][arm]
        if not entry:
            continue
        values = entry["values"].raw()
        cells = [f3(value) for value in values] + ["–"] * (3 - len(values))
        rows.append([f"{names[arm]} NLL"] + cells + [f3(entry["mean"]), f3(entry["sd"])])
    for key in ("ttt_minus_swa", "ttt_minus_full", "swa_minus_full"):
        entry = data["paired"][key]
        if not entry:
            continue
        first, second = key.split("_minus_")
        values = entry["values"].raw()
        cells = [signed(value) for value in values] + ["–"] * (3 - len(values))
        rows.append([f"Δ {names[first]} − {names[second]}"] + cells + [signed(entry["mean"]), f3(entry["sd"])])
    return markdown_table(header, rows)


def table_scaling(r, lang):
    names = ARM_NAMES[lang]
    data = r["scaling"]
    sizes = (("xs", "6 × 384"), ("s", "8 × 512"), ("m", "12 × 768"))
    header = {
        "zh": ["层数 × 宽度", f"{names['full']}", f"{names['swa']}", "TTT-E2E", "Δ TTT − 滑窗", "Δ TTT − 全注意力"],
        "en": [
            "Layers × width",
            names["full"],
            names["swa"],
            "TTT-E2E",
            "Δ TTT − sliding window",
            "Δ TTT − full attention",
        ],
    }[lang]
    f3 = fixed(3)
    rows = []
    for size, label in sizes:
        entry = data[size]
        if not entry:
            continue
        values = {arm: entry[arm]["nll"] if entry[arm] else None for arm in ("full", "swa", "ttt")}
        cells = [f3(values[arm]) if values[arm] is not None else "–" for arm in ("full", "swa", "ttt")]
        gaps = [
            signed(values["ttt"] - values[other]) if values["ttt"] is not None and values[other] is not None else "–"
            for other in ("swa", "full")
        ]
        rows.append([label] + cells + gaps)
    return markdown_table(header, rows)


def table_dynamic(r, lang):
    data = r["dynamic"]
    header = {
        "zh": ["模型", "测试时内循环学习率", "PG-19 val NLL（250 个窗口）", "PG-19 test NLL"],
        "en": ["Model", "Test-time inner learning rate", "PG-19 val NLL (250 windows)", "PG-19 test NLL"],
    }[lang]
    names = ARM_NAMES[lang]
    f3 = fixed(3)
    rows = []
    labels = {"off": {"zh": "不更新", "en": "no updates"}[lang]}
    for model, arm in (("ck", "ck"), ("ttt", "ttt")):
        entries = data[model]
        if not entries:
            continue
        order = sorted(entries.keys(), key=lambda key: -1.0 if key == "off" else float(key[2:].replace("p", ".")))
        for key in order:
            value = entries[key]
            rate = labels.get(key, key[2:].replace("p", "."))
            test = value["test"]
            rows.append(
                [
                    names[arm],
                    rate,
                    f3(value["validation"]) if not isinstance(value["validation"], Box) else "–",
                    f3(test) if not isinstance(test, Box) else "–",
                ]
            )
    return markdown_table(header, rows, ["l", "r", "r", "r"])


def table_overlap(r, lang):
    data = r["audits"]["benchmark_overlap"]["tasks"]
    header = {
        "zh": ["评测集", "检查的题目数", "题干片段在训练语料中逐字出现的题目数", "占比"],
        "en": ["Benchmark", "Items probed", "Items whose probe string occurs verbatim in the training text", "Share"],
    }[lang]
    rows = []
    for task in data.keys():
        entry = data[task]
        share = entry["items_with_exact_match"] / entry["items_probed"] if entry["items_probed"] else 0.0
        rows.append(
            [task, grouped(entry["items_probed"]), grouped(entry["items_with_exact_match"]), f"{100 * share:.2f}%"]
        )
    return markdown_table(header, rows)


def table_sampler(r, lang):
    data = r["audits"]["sampler_main"]["sources"]
    header = {
        "zh": [
            "来源",
            "语料 token 数",
            "目标比例",
            "实际抽到的窗口",
            "实际比例",
            "被覆盖的语料比例",
            "被重复抽到的比例",
        ],
        "en": [
            "Source",
            "Corpus tokens",
            "Target share",
            "Windows drawn",
            "Realised share",
            "Share of corpus seen",
            "Seen blocks drawn more than once",
        ],
    }[lang]
    targets = r["audits"]["sampler_main"]["target_probabilities"]
    rows = []
    for name in data.keys():
        entry = data[name]
        rows.append(
            [
                name,
                billions(entry["corpus_tokens"]),
                f"{100 * targets[name]:.0f}%",
                grouped(entry["windows"]),
                f"{100 * entry['window_fraction']:.2f}%",
                f"{100 * entry['corpus_fraction_seen']:.1f}%",
                f"{100 * entry['seen_blocks_sampled_more_than_once']:.1f}%",
            ]
        )
    return markdown_table(header, rows)


def table_sessions(r, lang):
    names = ARM_NAMES[lang]
    data = r["sessions"]
    arms = [arm for arm in ("full", "swa", "ttt") if data[arm]]
    header = {
        "zh": [
            "模型",
            "已读上下文（token）",
            "快权重（MiB）",
            "KV 缓存（MiB）",
            "会话状态合计（MiB）",
            "逐 token 解码（token/s）",
        ],
        "en": [
            "Model",
            "Context read (tokens)",
            "Fast weights (MiB)",
            "KV cache (MiB)",
            "Session state, total (MiB)",
            "Token-by-token decoding (tokens/s)",
        ],
    }[lang]
    rows = []
    for arm in arms:
        for row in data[arm]["rows"].raw():
            state = row["state_bytes"]
            rows.append(
                [
                    names[arm],
                    grouped(row["context_length"]),
                    f"{state['fast_weights'] / 2**20:.1f}" if state["fast_weights"] else "–",
                    f"{state['kv_cache'] / 2**20:.1f}",
                    f"{row['persistent_state_bytes'] / 2**20:.1f}",
                    grouped(row["decode_tokens_per_second"]),
                ]
            )
    return markdown_table(header, rows)


TABLES = {
    "sessions": table_sessions,
    "main": table_main,
    "ranges": table_ranges,
    "long": table_long,
    "benchmarks": table_benchmarks,
    "benchmarks_ours": lambda r, lang: table_benchmarks(r, lang, "ours"),
    "benchmarks_reference": lambda r, lang: table_benchmarks(r, lang, "reference"),
    "perplexity": table_perplexity,
    "crosscheck": table_crosscheck,
    "cost": table_cost,
    "latency": table_latency,
    "recall": table_recall,
    "ablations": table_ablations,
    "seeds": table_seeds,
    "scaling": table_scaling,
    "dynamic": table_dynamic,
    "overlap": table_overlap,
    "sampler": table_sampler,
}

INCLUDE = re.compile(r"\{% include ([\w./-]+) %\}")
CONDITIONAL = re.compile(r"\{% if (.+?) %\}\n?(.*?)(?:\{% else %\}\n?(.*?))?\{% endif %\}\n?", re.DOTALL)
TABLE = re.compile(r"\{% table (\w+) %\}")
# 表达式两侧必须有空格，避免与 LaTeX 里连续的花括号混淆。
EXPRESSION = re.compile(r"\{\{ (.+?) \}\}")


def render(template, results, lang):
    r = Box(results)
    namespace = {
        "r": r,
        "f1": fixed(1),
        "f2": fixed(2),
        "f3": fixed(3),
        "f4": fixed(4),
        "signed": signed,
        "ci": interval,
        "pct": percent,
        "M": millions,
        "B": billions,
        "K": thousands,
        "gib": gib,
        "GB": lambda value, digits=1: f"{value / 1e9:.{digits}f}",
        "n": grouped,
        "abs": abs,
        "min": min,
        "max": max,
        "round": round,
        "len": len,
        "exp": __import__("math").exp,
    }

    def evaluate(expression):
        return eval(expression.strip(), {"__builtins__": {}}, namespace)  # noqa: S307 - 模板是仓库内的可信文件

    def conditional(match):
        chosen = match.group(2) if evaluate(match.group(1)) else (match.group(3) or "")
        return chosen

    def table(match):
        return TABLES[match.group(1)](r, lang)

    def expression(match):
        value = evaluate(match.group(1))
        if isinstance(value, Box):
            raise ValueError(f"模板字段没有数据：{match.group(1).strip()}")
        return str(value)

    text = CONDITIONAL.sub(conditional, template)
    text = TABLE.sub(table, text)
    text = EXPRESSION.sub(expression, text)
    if "{%" in text or "{{ " in text:
        raise ValueError("模板中有未被处理的指令。")
    return prune_references(text)


def prune_references(text):
    """只保留正文引用过的参考文献；正文引用了没有定义的文献时报错。"""
    definition = re.compile(r"^\[\^([\w-]+)\]: ")
    body = [line for line in text.split("\n") if not definition.match(line)]
    cited = set(re.findall(r"\[\^([\w-]+)\]", "\n".join(body)))
    defined = {match.group(1) for line in text.split("\n") if (match := definition.match(line))}
    if cited - defined:
        raise ValueError(f"正文引用了未定义的参考文献：{sorted(cited - defined)}")
    kept = [line for line in text.split("\n") if not (match := definition.match(line)) or match.group(1) in cited]
    return "\n".join(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description="渲染技术报告")
    parser.add_argument("--results", type=Path, default=Path("docs/assets/report/results.json"))
    parser.add_argument("--templates", type=Path, default=Path("report"))
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))
    outputs = {"zh": args.docs / "technical_report.md", "en": args.docs / "technical_report_en.md"}
    for lang, output in outputs.items():
        template = args.templates / f"technical_report.{lang}.md"
        if not template.exists():
            continue
        source = template.read_text(encoding="utf-8")
        for _ in range(3):  # 允许被包含的文件再包含别的文件，最多三层
            source = INCLUDE.sub(lambda match: (args.templates / match.group(1)).read_text(encoding="utf-8"), source)
        text = render(source, results, lang)
        output.write_bytes(text.encode("utf-8"))
        print(f"rendered {output} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
