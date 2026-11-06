"""文本生成入口。"""

import argparse
import json
from dataclasses import replace

from .checkpoint import load_checkpoint
from .data import load_tokenizer
from .models.ttt_e2e.modeling_ttt_e2e import TTTE2EForCausalLM
from .runtime import TTTSession


def main():
    parser = argparse.ArgumentParser(description="TTT-E2E 文本生成")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-adapt", action="store_true")
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    experiment = checkpoint.experiment
    tokenizer = load_tokenizer(experiment.tokenizer)
    if tokenizer.vocab_size > experiment.model.vocab_size:
        raise ValueError("tokenizer 的词表大于模型词表。")
    model = TTTE2EForCausalLM(experiment.model)
    inner = replace(experiment.inner, enabled=False) if args.no_adapt else experiment.inner
    session = TTTSession(model, checkpoint.state.params, inner)
    generated = session.generate(
        tokenizer.encode(args.prompt),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(json.dumps({"text": tokenizer.decode(generated), "token_ids": generated}, ensure_ascii=False))


if __name__ == "__main__":
    main()
