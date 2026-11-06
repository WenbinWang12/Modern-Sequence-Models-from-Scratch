"""参数、会话状态与执行成本统计。"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from .checkpoint import load_checkpoint
from .inner_loop import sequence_loss
from .models.ttt_e2e.modeling_ttt_e2e import TTTE2EForCausalLM
from .runtime import TTTSession


def parameter_counts(model, params):
    def count(tree):
        return sum(value.size for value in jax.tree.leaves(tree))

    return {
        "total": count(params),
        "embedding": params["embed_tokens"].size,
        "fast_mlp": count(model.initial_fast_weights(params)),
    }


def profile_session(model, params, inner, *, context_length=16, decode_tokens=8):
    if context_length < 1 or decode_tokens < 0:
        raise ValueError("context_length 必须为正，decode_tokens 必须非负。")
    if context_length + decode_tokens > model.config.max_position_embeddings:
        raise ValueError("测量长度超过模型位置上限。")
    tokens = np.arange(context_length, dtype=np.int32) % model.config.vocab_size
    session = TTTSession(model, params, inner)

    def run():
        session.reset()
        start = time.perf_counter()
        session.prefill(tokens).block_until_ready()
        prefill = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(decode_tokens):
            session.decode(int(jnp.argmax(session.state.last_logits))).block_until_ready()
        return prefill, time.perf_counter() - start

    run()
    prefill, decode = run()
    memory = jax.devices()[0].memory_stats()
    return {
        "context_length": context_length,
        "decode_tokens": decode_tokens,
        "prefill_seconds": prefill,
        "decode_seconds": decode,
        "decode_tokens_per_second": decode_tokens / decode if decode_tokens else None,
        "state_bytes": session.state_bytes(),
        "device_peak_bytes": None if memory is None else memory.get("peak_bytes_in_use"),
        "inner_updates": session.state.updates,
    }


def estimated_training_flops(model, params, inner, sequence_length):
    tokens = jnp.zeros((sequence_length + 1,), dtype=jnp.int32)
    gradient = jax.jit(jax.value_and_grad(lambda weights: sequence_loss(model, weights, tokens, inner).loss))
    analysis = gradient.lower(params).cost_analysis()
    if isinstance(analysis, list):
        analysis = analysis[0]
    flops = analysis.get("flops")
    return {
        "xla_estimated_training_flops": flops,
        "xla_estimated_flops_per_token": None if flops is None else flops / sequence_length,
    }


def main():
    parser = argparse.ArgumentParser(description="测量 TTT-E2E 成本")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--training-flops", action="store_true")
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    model = TTTE2EForCausalLM(checkpoint.experiment.model)
    result = {"parameters": parameter_counts(model, checkpoint.state.params)}
    result.update(
        profile_session(
            model,
            checkpoint.state.params,
            checkpoint.experiment.inner,
            context_length=args.context_length,
            decode_tokens=args.decode_tokens,
        )
    )
    if args.training_flops:
        result.update(
            estimated_training_flops(model, checkpoint.state.params, checkpoint.experiment.inner, args.context_length)
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
