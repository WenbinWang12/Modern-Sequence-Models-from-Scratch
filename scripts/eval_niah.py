#!/usr/bin/env python3
"""单针检索（needle in a haystack）：按 RULER 的 ``niah`` 任务构造，用于基座语言模型。

提示词模板、针的句式、7 位数字和噪声句都取自 RULER（NVIDIA/RULER，
``scripts/data/synthetic/constants.py`` 与 ``niah.py``）。与 RULER 的差别有三点：

- 基座模型没有对话格式，这里把 RULER 的 ``answer_prefix`` 直接接在问题后面，让模型续写；
- 评分用“强制教师输入下的贪心匹配”：答案的每个 token 都是该位置概率最大的 token 才算对，
  它与贪心解码后做字符串匹配等价，但不需要逐 token 生成；同时记录答案的平均对数似然；
- 针的位置取 10 个等间隔深度（而不是随机抽样），自然文本干草堆用 PG-19 test 而不是随笔集。

每个上下文长度下，干草堆被填充到刚好不超过该长度。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import replace
from pathlib import Path

import eval_benchmarks
from eval_benchmarks import HuggingFaceScorer, JaxScorer, encode_pair, sha256_file, wilson_interval

TEMPLATE = (
    "Some special magic numbers are hidden within the following text. Make sure to memorize it. "
    "I will quiz you about the numbers afterwards.\n{context}\n"
    "What are all the special magic numbers for {query} mentioned in the provided text?"
)
ANSWER_PREFIX = " The special magic numbers for {query} mentioned in the provided text are"
NEEDLE = "One of the special magic numbers for {key} is: {value}."
NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
ADJECTIVES = (
    "quiet bright narrow heavy gentle rapid hollow frozen silver crimson distant ancient "
    "curious eager fragile humble jagged lively mellow nimble polite rustic sturdy tender"
).split()
NOUNS = (
    "harbor lantern meadow orchard quarry river saddle temple valley window anchor basket "
    "candle dragon engine feather garden hammer island jacket kettle ladder mirror needle"
).split()
DEPTHS = tuple(index / 9 for index in range(10))
SENTENCE = re.compile(r"(?<=[.!?])\s+")


def book_sentences(directory: Path, count: int) -> list[list[str]]:
    """每本书取正文中段的句子，足够填满最长的上下文。"""
    books = []
    for path in sorted(directory.glob("*.txt"))[:count]:
        text = re.sub(r"\s+", " ", path.read_text(encoding="utf-8", errors="replace"))
        middle = text[len(text) // 4 : len(text) // 4 + 400_000]
        books.append(SENTENCE.split(middle)[1:-1])
    return books


def build_example(scorer, sentences, joiner, length, depth, key, value, decoy):
    """在 token 预算内放尽可能多的句子，把针插在给定深度处。

    同时构造一个对照：同样的上下文，但针里的键和值换成无关的一对。两者的提问和答案完全相同，
    因此答案对数似然之差只反映“文中是否真的写着这个答案”。
    """
    needle = NEEDLE.format(key=key, value=value)
    control = NEEDLE.format(key=decoy[0], value=decoy[1])
    prefix = ANSWER_PREFIX.format(query=key)
    frame = TEMPLATE.format(context=needle, query=key) + prefix
    budget = length - len(scorer.encode(frame)) - len(scorer.encode(" " + value)) - 8
    chosen, used = [], 0
    for sentence in sentences:
        cost = len(scorer.encode(joiner + sentence))
        if used + cost > budget:
            break
        chosen.append(sentence)
        used += cost
    position = round(depth * len(chosen))
    head, tail = chosen[:position], chosen[position:]
    prompts = tuple(
        TEMPLATE.format(context=joiner.join(head + [item] + tail), query=key) + prefix for item in (needle, control)
    )
    return prompts, " " + value, position / max(len(chosen), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="单针检索评测")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--hf-model")
    parser.add_argument("--no-adapt", action="store_true")
    parser.add_argument("--force-window", type=int, help="把所有层改成给定窗口的滑窗注意力（参数不变）")
    parser.add_argument("--always-prefix", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lengths", nargs="+", type=int, default=[1024, 2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--haystacks", nargs="+", choices=("noise", "books"), default=["noise", "books"])
    parser.add_argument("--samples-per-depth", type=int, default=5)
    parser.add_argument("--text-dir", type=Path, default=Path("data/raw/pg19/test"))
    parser.add_argument("--tokens-per-batch", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    longest = max(args.lengths)
    eval_benchmarks.BUCKETS = tuple(sorted({*eval_benchmarks.BUCKETS, 16384, 32768}))
    if args.checkpoint:
        scorer = JaxScorer(
            args.checkpoint, adapt=not args.no_adapt, max_length=longest, tokens_per_batch=args.tokens_per_batch
        )
        if args.force_window:
            model = scorer.experiment.model
            layers = ("sliding_attention",) * model.num_hidden_layers
            changed = replace(model, layer_types=layers, sliding_window=args.force_window)
            scorer.experiment = replace(scorer.experiment, model=changed)
            scorer.description["force_window"] = args.force_window
    else:
        scorer = HuggingFaceScorer(
            args.hf_model,
            device=args.device,
            max_length=longest,
            tokens_per_batch=args.tokens_per_batch,
            always_prefix=args.always_prefix,
        )
        args.lengths = [length for length in args.lengths if length <= scorer.max_length]
    books = book_sentences(args.text_dir, 20) if "books" in args.haystacks else []
    results, records = {}, []
    for haystack in args.haystacks:
        for length in args.lengths:
            rng = random.Random(f"{args.seed}-{haystack}-{length}")
            pairs, controls, meta = [], [], []
            for depth in DEPTHS:
                for sample in range(args.samples_per_depth):
                    key = f"{rng.choice(ADJECTIVES)}-{rng.choice(NOUNS)}"
                    value = str(rng.randint(10**6, 10**7 - 1))
                    decoy = (f"{rng.choice(ADJECTIVES)}-{rng.choice(NOUNS)}", str(rng.randint(10**6, 10**7 - 1)))
                    if haystack == "noise":
                        sentences, joiner = [NOISE] * (length // 8), "\n"
                    else:
                        sentences, joiner = books[(sample + int(depth * 9) * 7) % len(books)], " "
                    prompts, target, actual = build_example(scorer, sentences, joiner, length, depth, key, value, decoy)
                    pairs.append(encode_pair(scorer, prompts[0], target))
                    controls.append(encode_pair(scorer, prompts[1], target))
                    meta.append({"depth": depth, "actual_depth": actual, "key": key, "value": value})
            started = time.perf_counter()
            scores = scorer.score(pairs, greedy=True)
            control_scores = scorer.score(controls)
            hits = 0
            by_depth = {f"{depth:.2f}": [0, 0] for depth in DEPTHS}
            for (context_ids, _), info, (loglik, exact, count), control in zip(pairs, meta, scores, control_scores):
                hits += bool(exact)
                by_depth[f"{info['depth']:.2f}"][0] += bool(exact)
                by_depth[f"{info['depth']:.2f}"][1] += 1
                records.append(
                    {
                        "haystack": haystack,
                        "length": length,
                        **info,
                        "prompt_tokens": len(context_ids),
                        "target_tokens": count,
                        "loglik": loglik,
                        "control_loglik": control[0],
                        "exact": bool(exact),
                    }
                )
            total = len(scores)
            tokens = sum(score[2] for score in scores)
            answer_nll = -sum(score[0] for score in scores) / tokens
            control_nll = -sum(score[0] for score in control_scores) / tokens
            results[f"{haystack}_{length}"] = {
                "haystack": haystack,
                "length": length,
                "samples": total,
                "accuracy": hits / total,
                "accuracy_ci95": wilson_interval(hits, total),
                "answer_nll_per_token": answer_nll,
                "control_answer_nll_per_token": control_nll,
                # 针在场时答案变便宜了多少：即使一个 token 都没答对，这一项仍能反映模型是否用到了针。
                "needle_gain_per_token": control_nll - answer_nll,
                "mean_prompt_tokens": sum(len(pair[0]) for pair in pairs) / total,
                "accuracy_by_depth": {depth: correct / count for depth, (correct, count) in by_depth.items()},
                "seconds": time.perf_counter() - started,
            }
            summary = {key: value for key, value in results[f"{haystack}_{length}"].items() if key != "accuracy_by_depth"}
            print(json.dumps(summary), flush=True)
            report = {
                "protocol_version": "niah_single_v1",
                "model": scorer.description,
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "seed": args.seed,
                "results": results,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
