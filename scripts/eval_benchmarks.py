#!/usr/bin/env python3
"""零样本公开 benchmark 评测（continuation log-likelihood 协议）。

任务模板与 lm-evaluation-harness 的同名任务保持一致，因此本仓库模型与公开参考模型
（通过 ``--hf-model`` 加载）可以在完全相同的题目、提示词和评分规则下比较：

- 多选题：对每个候选项计算 ``log p(continuation | context)``，取最大者；同时报告按候选项
  字符长度归一化的 ``acc_norm``；
- LAMBADA：最后一个词的贪心命中率和困惑度；
- WikiText：文档级滚动 log-likelihood，报告 word perplexity 和 bits per byte。

不使用 few-shot 示例，不做任何针对 benchmark 的训练或提示词搜索。每道题的预测都写入
``*.predictions.jsonl``，可用于配对检验。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

BUCKETS = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


# --------------------------------------------------------------------------- 任务定义


def _hellaswag_clean(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def build_hellaswag(rows):
    for row in rows:
        context = _hellaswag_clean(row["activity_label"] + ": " + row["ctx_a"] + " " + row["ctx_b"].capitalize())
        choices = [_hellaswag_clean(ending) for ending in row["endings"]]
        yield {
            "requests": [(context, " " + choice) for choice in choices],
            "gold": int(row["label"]),
            "choices": choices,
        }


def build_piqa(rows):
    for row in rows:
        choices = [row["sol1"], row["sol2"]]
        context = f"Question: {row['goal']}\nAnswer:"
        yield {
            "requests": [(context, " " + choice) for choice in choices],
            "gold": int(row["label"]),
            "choices": choices,
        }


def build_arc(rows):
    for row in rows:
        choices = list(row["choices"]["text"])
        gold = list(row["choices"]["label"]).index(row["answerKey"])
        context = f"Question: {row['question']}\nAnswer:"
        yield {"requests": [(context, " " + choice) for choice in choices], "gold": gold, "choices": choices}


def build_openbookqa(rows):
    for row in rows:
        choices = list(row["choices"]["text"])
        gold = list(row["choices"]["label"]).index(row["answerKey"].strip())
        yield {
            "requests": [(row["question_stem"], " " + choice) for choice in choices],
            "gold": gold,
            "choices": choices,
        }


def build_sciq(rows):
    for row in rows:
        choices = [row["distractor1"], row["distractor2"], row["distractor3"], row["correct_answer"]]
        context = f"{row['support'].lstrip()}\nQuestion: {row['question']}\nAnswer:"
        yield {"requests": [(context, " " + choice) for choice in choices], "gold": 3, "choices": choices}


def build_winogrande(rows):
    # 部分评分：两个候选上下文共享同一个后续片段。
    for row in rows:
        index = row["sentence"].index("_")
        target = " " + row["sentence"][index + 1 :].strip()
        contexts = [row["sentence"][:index] + option for option in (row["option1"], row["option2"])]
        yield {"requests": [(context, target) for context in contexts], "gold": int(row["answer"]) - 1}


def build_blimp(rows):
    for row in rows:
        yield {
            "requests": [("", row["sentence_good"]), ("", row["sentence_bad"])],
            "gold": 0,
            "group": row["config"],
        }


def build_mmlu(rows):
    for row in rows:
        subject = row["subject"].replace("_", " ")
        options = "".join(f"{letter}. {choice}\n" for letter, choice in zip("ABCD", row["choices"]))
        context = (
            f"The following are multiple choice questions (with answers) about {subject}.\n\n"
            f"{row['question'].strip()}\n{options}Answer:"
        )
        yield {
            "requests": [(context, f" {letter}") for letter in "ABCD"],
            "gold": int(row["answer"]),
            "group": row["subject"],
        }


def build_lambada(rows):
    for row in rows:
        context, target = row["text"].rsplit(" ", 1)
        yield {"requests": [(context, " " + target)], "gold": 0}


def wikitext_detokenize(text):
    text = text.replace("s '", "s'")
    text = re.sub(r"/' [0-9]/", r"/'[0-9]/", text)
    text = text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    for mark in (":", ";", ".", "!", "?", ","):
        text = text.replace(f" {mark} ", f"{mark} ")
    text = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", text)
    text = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", text)
    text = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", text)
    text = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', text)
    text = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", text)
    text = text.replace("= = = =", "====").replace("= = =", "===").replace("= =", "==")
    text = text.replace(" " + chr(176) + " ", chr(176))
    text = text.replace(" \n", "\n").replace("\n ", "\n").replace(" N ", " 1 ")
    return text.replace(" 's", "'s")


MULTIPLE_CHOICE = {
    "hellaswag": (build_hellaswag, True),
    "piqa": (build_piqa, True),
    "arc_easy": (build_arc, True),
    "arc_challenge": (build_arc, True),
    "openbookqa": (build_openbookqa, True),
    "sciq": (build_sciq, True),
    "winogrande": (build_winogrande, False),
    "blimp": (build_blimp, False),
    "mmlu": (build_mmlu, False),
}


# --------------------------------------------------------------------------- 打分器


def encode_pair(scorer, context, continuation):
    """与 lm-evaluation-harness 相同：上下文末尾的空白并入续写，空上下文用前缀 token。"""
    spaces = len(context) - len(context.rstrip())
    if spaces:
        continuation = context[-spaces:] + continuation
        context = context[:-spaces]
    if not context:
        return [scorer.prefix_token], scorer.encode(continuation)
    whole = scorer.encode(context + continuation)
    context_ids = scorer.encode(context)
    continuation_ids = whole[len(context_ids) :]
    if scorer.always_prefix:
        context_ids = [scorer.prefix_token] + context_ids
    return context_ids, continuation_ids


def truncate(context_ids, continuation_ids, max_length):
    ids = (context_ids + continuation_ids)[-(max_length + 1) :]
    context_length = max(1, len(ids) - len(continuation_ids))
    return ids, context_length


class JaxScorer:
    def __init__(self, checkpoint_path, *, adapt, max_length, tokens_per_batch):
        import jax
        import jax.numpy as jnp

        from ttt.checkpoint import load_checkpoint
        from ttt.data import load_tokenizer
        from ttt.inner_loop import sequence_loss
        from ttt.models.ttt_e2e.modeling_ttt_e2e import TTTE2EForCausalLM

        self.jax, self.jnp, self.sequence_loss, self.model_class = jax, jnp, sequence_loss, TTTE2EForCausalLM
        checkpoint = load_checkpoint(checkpoint_path)
        self.experiment = checkpoint.experiment
        self.params = checkpoint.state.params
        self.step = int(checkpoint.state.step)
        self.tokenizer = load_tokenizer(self.experiment.tokenizer)
        self.prefix_token = self.tokenizer.eos_token_id
        self.always_prefix = True
        config = self.experiment.model
        self.adaptive = bool(adapt and self.experiment.inner.enabled and config.use_fast_mlp)
        self.inner = self.experiment.inner if self.adaptive else replace(self.experiment.inner, enabled=False)
        self.max_length = max_length or self.experiment.training.sequence_length
        self.tokens_per_batch = tokens_per_batch
        self.functions = {}
        self.description = {
            "kind": "ttt_checkpoint",
            "checkpoint": str(checkpoint_path),
            "step": self.step,
            "adapt": self.adaptive,
            "max_length": self.max_length,
            "prefix_token": "always <|endoftext|>",
        }

    def encode(self, text):
        return self.tokenizer.encode(text, add_bos=False)

    def _function(self, bucket, greedy):
        key = (bucket, greedy)
        if key not in self.functions:
            config = self.experiment.model
            if not self.adaptive:
                # 不做测试时更新时，块长度不影响数值，只影响填充量。
                limit = config.sliding_window if config.use_fast_mlp else bucket
                config = replace(
                    config,
                    ttt_chunk_size=min(bucket, limit),
                    loss_token_block_size=min(bucket, limit, 256),
                )
            model = self.model_class(config)
            jax, jnp = self.jax, self.jnp

            def single(params, ids, mask):
                output = self.sequence_loss(model, params, ids, self.inner, mask, return_logits=greedy)
                if greedy:
                    return output.token_losses, jnp.argmax(output.logits, axis=-1).astype(jnp.int32)
                return output.token_losses, jnp.zeros((bucket,), dtype=jnp.int32)

            self.functions[key] = jax.jit(jax.vmap(single, in_axes=(None, 0, 0)))
        return self.functions[key]

    def score(self, pairs, *, greedy=False):
        prepared = [truncate(context, continuation, self.max_length) for context, continuation in pairs]
        results = [None] * len(pairs)
        by_bucket = {}
        for index, (ids, _) in enumerate(prepared):
            bucket = next(size for size in BUCKETS if size >= len(ids) - 1)
            by_bucket.setdefault(bucket, []).append(index)
        for bucket, indices in sorted(by_bucket.items()):
            budget = self.tokens_per_batch // (8 if greedy else 1)
            batch_size = max(1, min(512, budget // bucket))
            function = self._function(bucket, greedy)
            for offset in range(0, len(indices), batch_size):
                batch = indices[offset : offset + batch_size]
                ids = np.full((batch_size, bucket + 1), self.prefix_token, dtype=np.int32)
                mask = np.zeros((batch_size, bucket), dtype=np.float32)
                for row, index in enumerate(batch):
                    tokens = prepared[index][0]
                    ids[row, : len(tokens)] = tokens
                    mask[row, : len(tokens) - 1] = 1.0
                losses, predictions = function(self.params, self.jnp.asarray(ids), self.jnp.asarray(mask))
                losses, predictions = np.asarray(losses, dtype=np.float64), np.asarray(predictions)
                for row, index in enumerate(batch):
                    tokens, context_length = prepared[index]
                    span = slice(context_length - 1, len(tokens) - 1)
                    exact = (
                        bool((predictions[row, span] == np.asarray(tokens[context_length:])).all()) if greedy else None
                    )
                    results[index] = (-float(losses[row, span].sum()), exact, len(tokens) - context_length)
        return results


class HuggingFaceScorer:
    def __init__(self, name, *, device, max_length, tokens_per_batch, always_prefix=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
        self.device = device
        self.body, self.head = self.model.base_model, self.model.get_output_embeddings()
        with torch.no_grad():
            # 确认“主干输出 + 输出嵌入”与完整前向的 logits 一致，否则拒绝运行。
            probe = torch.arange(1, 17, device=device)[None]
            reference = self.model(input_ids=probe).logits
            shortcut = self.head(self.body(input_ids=probe).last_hidden_state)
            if float((reference - shortcut).abs().max()) > 1e-3:
                raise ValueError("该模型的主干输出不能直接接输出嵌入，需单独适配。")
        prefix = self.tokenizer.bos_token_id
        self.prefix_token = prefix if prefix is not None else self.tokenizer.eos_token_id
        self.always_prefix = always_prefix
        limit = getattr(self.model.config, "n_positions", None) or self.model.config.max_position_embeddings
        self.max_length = min(max_length, limit) if max_length else limit
        self.tokens_per_batch = tokens_per_batch
        self.description = {
            "kind": "huggingface",
            "model": name,
            "revision": getattr(self.model.config, "_commit_hash", None),
            "parameters": int(sum(parameter.numel() for parameter in self.model.parameters())),
            "max_length": self.max_length,
            "prefix_token": "always" if always_prefix else "only for empty context (lm-evaluation-harness convention)",
            "device": device,
        }

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def score(self, pairs, *, greedy=False):
        torch = self.torch
        prepared = [truncate(context, continuation, self.max_length) for context, continuation in pairs]
        order = sorted(range(len(prepared)), key=lambda index: -len(prepared[index][0]))
        results = [None] * len(pairs)
        offset = 0
        while offset < len(order):
            longest = len(prepared[order[offset]][0])
            batch_size = max(1, min(256, self.tokens_per_batch // longest))
            batch = order[offset : offset + batch_size]
            offset += len(batch)
            ids = torch.zeros((len(batch), longest), dtype=torch.long)
            attention = torch.zeros((len(batch), longest - 1), dtype=torch.long)
            row_index, position_index, spans = [], [], []
            for row, index in enumerate(batch):
                tokens, context_length = prepared[index]
                ids[row, : len(tokens)] = torch.tensor(tokens)
                attention[row, : len(tokens) - 1] = 1
                positions = range(context_length - 1, len(tokens) - 1)
                spans.append((len(row_index), len(positions)))
                row_index.extend([row] * len(positions))
                position_index.extend(positions)
            ids = ids.to(self.device)
            rows = torch.tensor(row_index, device=self.device)
            columns = torch.tensor(position_index, device=self.device)
            with torch.no_grad():
                hidden = self.body(input_ids=ids[:, :-1], attention_mask=attention.to(self.device)).last_hidden_state
                # 只对需要评分的位置做词表投影，避免为上下文位置计算整张词表。
                logits = self.head(hidden[rows, columns]).float()
                targets = ids[rows, columns + 1]
                token_scores = (
                    torch.log_softmax(logits, dim=-1).gather(1, targets[:, None])[:, 0].double().cpu().numpy()
                )
                hits = (logits.argmax(dim=-1) == targets).cpu().numpy() if greedy else None
            for (start, count), index in zip(spans, batch):
                exact = bool(hits[start : start + count].all()) if greedy else None
                results[index] = (float(token_scores[start : start + count].sum()), exact, count)
        return results


# --------------------------------------------------------------------------- 评测流程


def wilson_interval(correct, total, z=1.959963984540054):
    if not total:
        return None
    rate = correct / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [centre - margin, centre + margin]


def evaluate_multiple_choice(scorer, task, rows, predictions_file):
    builder, normalise = MULTIPLE_CHOICE[task]
    examples = list(builder(rows))
    pairs, owners = [], []
    for index, example in enumerate(examples):
        for context, continuation in example["requests"]:
            pairs.append(encode_pair(scorer, context, continuation))
            owners.append(index)
    scores = scorer.score(pairs)
    grouped = [[] for _ in examples]
    for owner, (loglik, _, _) in zip(owners, scores):
        grouped[owner].append(loglik)
    correct = correct_norm = 0
    groups = {}
    for index, (example, logliks) in enumerate(zip(examples, grouped)):
        prediction = int(np.argmax(logliks))
        hit = prediction == example["gold"]
        correct += hit
        record = {"task": task, "index": index, "gold": example["gold"], "prediction": prediction, "logliks": logliks}
        if normalise:
            lengths = [max(1, len(choice)) for choice in example["choices"]]
            prediction_norm = int(np.argmax([value / length for value, length in zip(logliks, lengths)]))
            correct_norm += prediction_norm == example["gold"]
            record["prediction_norm"] = prediction_norm
        if "group" in example:
            stats = groups.setdefault(example["group"], [0, 0])
            stats[0] += hit
            stats[1] += 1
            record["group"] = example["group"]
        predictions_file.write(json.dumps(record) + "\n")
    total = len(examples)
    result = {
        "samples": total,
        "choices_per_question": float(np.mean([len(example["requests"]) for example in examples])),
        "acc": correct / total,
        "acc_ci95": wilson_interval(correct, total),
    }
    if normalise:
        result["acc_norm"] = correct_norm / total
        result["acc_norm_ci95"] = wilson_interval(correct_norm, total)
    if groups:
        result["macro_acc"] = float(np.mean([hits / count for hits, count in groups.values()]))
        result["groups"] = {name: hits / count for name, (hits, count) in sorted(groups.items())}
    return result


def evaluate_lambada(scorer, rows, predictions_file):
    examples = list(build_lambada(rows))
    pairs = [encode_pair(scorer, *example["requests"][0]) for example in examples]
    scores = scorer.score(pairs, greedy=True)
    for index, (loglik, exact, _) in enumerate(scores):
        predictions_file.write(
            json.dumps({"task": "lambada_openai", "index": index, "loglik": loglik, "exact": exact}) + "\n"
        )
    hits = sum(bool(exact) for _, exact, _ in scores)
    return {
        "samples": len(scores),
        "acc": hits / len(scores),
        "acc_ci95": wilson_interval(hits, len(scores)),
        "perplexity": math.exp(-float(np.mean([loglik for loglik, _, _ in scores]))),
    }


def rolling_windows(tokens, prefix_token, max_length):
    """与 lm-evaluation-harness 的 get_rolling_token_windows(context_len=1) 等价。"""
    first = min(max_length, len(tokens))
    yield [prefix_token] + tokens[: first - 1], tokens[:first]
    predicted = first
    while predicted < len(tokens):
        count = min(len(tokens) - predicted, max_length)
        end = predicted + count
        inputs, targets = tokens[end - max_length - 1 : end - 1], tokens[end - count : end]
        yield inputs, targets
        predicted += count


def evaluate_rolling(scorer, task, texts, originals, predictions_file):
    pairs, owners = [], []
    for index, text in enumerate(texts):
        tokens = scorer.encode(text)
        for inputs, targets in rolling_windows(tokens, scorer.prefix_token, scorer.max_length):
            context = inputs[: len(inputs) - (len(targets) - 1)]
            pairs.append((context, targets))
            owners.append(index)
    scores = scorer.score(pairs)
    logliks, token_counts = np.zeros(len(texts)), np.zeros(len(texts))
    for owner, (loglik, _, count) in zip(owners, scores):
        logliks[owner] += loglik
        token_counts[owner] += count
    words = np.asarray([len(re.split(r"\s+", text)) for text in originals], dtype=np.float64)
    byte_counts = np.asarray([len(text.encode("utf-8")) for text in originals], dtype=np.float64)
    for index in range(len(texts)):
        record = {"task": task, "index": index, "loglik": float(logliks[index]), "tokens": int(token_counts[index])}
        predictions_file.write(json.dumps(record) + "\n")
    total = float(logliks.sum())
    return {
        "documents": len(texts),
        "tokens": int(token_counts.sum()),
        "words": int(words.sum()),
        "bytes": int(byte_counts.sum()),
        "token_nll": -total / float(token_counts.sum()),
        "token_perplexity": math.exp(-total / float(token_counts.sum())),
        "word_perplexity": math.exp(-total / float(words.sum())),
        "bits_per_byte": -total / float(byte_counts.sum()) / math.log(2),
        "max_length": scorer.max_length,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    # 只按换行符 LF 切分：str.splitlines 会把字符串内部的 U+2028 等字符也当作换行。
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="零样本 benchmark 评测")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--hf-model")
    parser.add_argument("--data-dir", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--limit", type=int, help="每个任务只评前 N 条，仅用于调试")
    parser.add_argument("--no-adapt", action="store_true")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--tokens-per-batch", type=int, default=65536)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--always-prefix", action="store_true", help="参考模型也始终加前缀 token")
    parser.add_argument("--text-dir", type=Path, help="pg19 任务的原始文本目录")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.checkpoint:
        scorer = JaxScorer(
            args.checkpoint, adapt=not args.no_adapt, max_length=args.max_length, tokens_per_batch=args.tokens_per_batch
        )
    else:
        scorer = HuggingFaceScorer(
            args.hf_model,
            device=args.device,
            max_length=args.max_length,
            tokens_per_batch=args.tokens_per_batch,
            always_prefix=args.always_prefix,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results, timings, inputs = {}, {}, {}

    def write_report():
        report = {
            "protocol_version": "zero_shot_loglikelihood_v1",
            "model": scorer.description,
            "limit": args.limit,
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "inputs": inputs,
            "results": results,
            "seconds": timings,
        }
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with args.output.with_suffix(".predictions.jsonl").open("w", encoding="utf-8") as predictions_file:
        for task in args.tasks:
            started = time.perf_counter()
            if task == "pg19":
                paths = sorted(args.text_dir.glob("*.txt"), key=lambda path: path.name)[: args.limit]
                texts = [path.read_text(encoding="utf-8", errors="replace") for path in paths]
                results[task] = evaluate_rolling(scorer, task, texts, texts, predictions_file)
                inputs[task] = {"directory": str(args.text_dir), "files": len(paths)}
            else:
                path = args.data_dir / f"{task}.jsonl"
                rows = read_jsonl(path)[: args.limit]
                inputs[task] = {"path": str(path), "sha256": sha256_file(path), "rows": len(rows)}
                if task in MULTIPLE_CHOICE:
                    results[task] = evaluate_multiple_choice(scorer, task, rows, predictions_file)
                elif task == "lambada_openai":
                    results[task] = evaluate_lambada(scorer, rows, predictions_file)
                elif task == "wikitext":
                    originals = [row["page"] for row in rows]
                    texts = [wikitext_detokenize(text) for text in originals]
                    results[task] = evaluate_rolling(scorer, task, texts, originals, predictions_file)
                else:
                    raise ValueError(f"未知任务：{task}")
            timings[task] = time.perf_counter() - started
            summary = {key: value for key, value in results[task].items() if key != "groups"}
            print(json.dumps({"task": task, "seconds": round(timings[task], 1), **summary}), flush=True)
            write_report()


if __name__ == "__main__":
    main()
