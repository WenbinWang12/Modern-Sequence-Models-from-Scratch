"""语言建模损失与生成式召回评测。"""

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .checkpoint import load_checkpoint
from .data import BatchStream, load_tokenizer
from .inner_loop import sequence_loss
from .models.ttt_e2e.modeling_ttt_e2e import TTTE2EForCausalLM
from .runtime import TTTSession


def evaluate_loss(model, params, inner, data_path, *, sequence_length, samples=8, seed=0):
    if samples <= 0:
        raise ValueError("samples 必须大于 0。")
    stream = BatchStream(
        data_path,
        sequence_length=sequence_length,
        batch_size=1,
        vocab_size=model.config.vocab_size,
        seed=seed,
    )
    evaluate = jax.jit(lambda weights, tokens, mask: sequence_loss(model, weights, tokens, inner, mask).loss)
    total_loss, total_tokens = 0.0, 0.0
    for _ in range(samples):
        tokens, masks = stream.next_batch_with_masks()
        count = float(masks.sum())
        total_loss += float(evaluate(params, jnp.asarray(tokens[0]), jnp.asarray(masks[0]))) * count
        total_tokens += count
    if not total_tokens:
        raise ValueError("评测样本没有可评分的标签。")
    nll = total_loss / total_tokens
    return {"nll": nll, "perplexity": math.exp(min(nll, 700)), "samples": samples, "tokens": total_tokens}


def evaluate_retrieval(session, tokenizer, records, *, max_new_tokens=32):
    results = []
    for record in records:
        prompt = record["prompt"]
        answers = record.get("answers", [record.get("answer")])
        if not isinstance(prompt, str) or not answers or not all(isinstance(answer, str) for answer in answers):
            raise ValueError("每条评测数据需要 prompt，以及 answer 或 answers 字段。")
        session.reset()
        # 答案只交给评分器；模型只读取 prompt 并自由生成。
        tokens = session.generate(
            tokenizer.encode(prompt),
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
        prediction = tokenizer.decode(tokens)
        exact = any(prediction.strip() == answer.strip() for answer in answers)
        results.append({"id": record.get("id", len(results)), "prediction": prediction, "exact_match": exact})
    if not results:
        raise ValueError("评测数据不能为空。")
    return {
        "exact_match": sum(row["exact_match"] for row in results) / len(results),
        "samples": len(results),
        "results": results,
    }


def make_probes(path, *, samples=8, haystack_characters=128, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(samples):
        code = str(int(rng.integers(100000, 1000000)))
        background = "ordinary tool log. " * (haystack_characters // 19 + 1)
        cut = int(rng.integers(0, len(background) + 1))
        context = background[:cut] + f"\nThe secret ID is {code}.\n" + background[cut:]
        rows.append({"id": index, "prompt": context + "\nReturn only the secret ID:", "answer": code})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="TTT-E2E 评测")
    parser.add_argument("--checkpoint")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output")
    parser.add_argument("--mode", choices=("loss", "retrieval", "make-probes"), default="loss")
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--haystack-characters", type=int, default=128)
    parser.add_argument("--no-adapt", action="store_true")
    args = parser.parse_args()
    if args.mode == "make-probes":
        make_probes(args.data, samples=args.samples, haystack_characters=args.haystack_characters)
        return
    if args.checkpoint is None:
        parser.error("评测需要 --checkpoint。")
    checkpoint = load_checkpoint(args.checkpoint)
    experiment = checkpoint.experiment
    model = TTTE2EForCausalLM(experiment.model)
    inner = replace(experiment.inner, enabled=False) if args.no_adapt else experiment.inner
    if args.mode == "loss":
        result = evaluate_loss(
            model,
            checkpoint.state.params,
            inner,
            args.data,
            sequence_length=args.sequence_length or experiment.training.sequence_length,
            samples=args.samples,
        )
    else:
        records = [
            json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        result = evaluate_retrieval(
            TTTSession(model, checkpoint.state.params, inner),
            load_tokenizer(experiment.tokenizer),
            records,
            max_new_tokens=args.max_new_tokens,
        )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
