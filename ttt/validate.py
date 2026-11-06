"""生成可复查的服务器验证报告。"""

import argparse
import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from .inner_loop import InnerLoopConfig, sequence_loss
from .lifecycle import atomic_json
from .models.ttt_e2e import TTTE2EConfig, TTTE2EForCausalLM
from .provenance import environment_report
from .runtime import TTTSession


def validation_case(name, config, *, length=8):
    model = TTTE2EForCausalLM(config)
    params = model.init_parameters(jax.random.PRNGKey(19))
    tokens = jnp.arange(length + 1, dtype=jnp.int32) % config.vocab_size
    inner = InnerLoopConfig(learning_rate=0.1)
    function = jax.jit(jax.value_and_grad(lambda weights: sequence_loss(model, weights, tokens, inner).loss))
    start = time.perf_counter()
    executable = function.lower(params).compile()
    compilation_seconds = time.perf_counter() - start
    start = time.perf_counter()
    loss, gradients = executable(params)
    loss.block_until_ready()
    execution_seconds = time.perf_counter() - start
    finite = all(bool(jnp.all(jnp.isfinite(value))) for value in jax.tree.leaves(gradients))
    if not finite or not np.isfinite(float(loss)):
        raise AssertionError(f"{name} 存在非有限损失或梯度。")
    comparison = sequence_loss(model, params, tokens, inner, return_logits=True)
    session = TTTSession(model, params, inner)
    logits = session.observe(np.asarray(tokens))
    logits.block_until_ready()
    error = float(jnp.max(jnp.abs(logits[:-1] - comparison.logits)))
    if error > 1e-3:
        raise AssertionError(f"{name} 的训练与推理结果不一致：{error}")
    changed = tokens.at[length // 2 :].set((tokens[length // 2 :] + 7) % config.vocab_size)
    changed_logits = sequence_loss(model, params, changed, inner, return_logits=True).logits
    np.testing.assert_allclose(changed_logits[: length // 2], comparison.logits[: length // 2], atol=1e-5, rtol=1e-4)
    return {
        "name": name,
        "tokens": length,
        "loss": float(loss),
        "finite_gradients": finite,
        "logits_max_error": error,
        "compilation_seconds": compilation_seconds,
        "training_seconds": execution_seconds,
        "state_bytes": session.state_bytes(),
        "inner_updates": session.state.updates,
    }


def validate(*, require_gpu=False, lengths=(64, 512)):
    if require_gpu and not all(device.platform == "gpu" for device in jax.local_devices()):
        raise RuntimeError("本次验证要求 GPU，但 JAX 未识别到 GPU。")
    base = TTTE2EConfig(
        vocab_size=259,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        compute_dtype="float32",
        post_norm=False,
        ttt_chunk_size=4,
        sliding_window=16,
        max_position_embeddings=max(lengths) + 16,
        attention_query_block_size=8,
        attention_key_block_size=16,
        loss_vocab_block_size=64,
        layer_types=("full_attention", "sliding_attention"),
    )
    cases = [validation_case("dense", base)]
    cases.append(
        validation_case(
            "gdn", replace(base, use_fast_mlp=False, ttt_suffix_length=0, layer_types=("gated_delta", "full_attention"))
        )
    )
    for target in ("mlp", "router", "delta_router"):
        config = replace(
            base,
            num_experts=4,
            num_experts_per_token=2,
            first_k_dense_layers=0,
            adaptation_target=target,
            use_fast_mlp=target == "mlp",
            ttt_suffix_length=0 if target == "delta_router" else 1,
        )
        cases.append(validation_case("moe_" + target, config))
    for mode in ("early_ffn", "middle_block", "middle_attention"):
        cases.append(validation_case(mode, replace(base, loop_mode=mode, loop_start=0, loop_end=1)))
    for length in lengths:
        cases.append(
            validation_case(
                f"context_{length}",
                replace(
                    base,
                    ttt_chunk_size=16,
                    sliding_window=64,
                    attention_query_block_size=32,
                    attention_key_block_size=64,
                ),
                length=length,
            )
        )
    return {"environment": environment_report(), "passed": True, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description="验证 TTT 生产实现")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--lengths", nargs="+", type=int, default=[64, 512])
    args = parser.parse_args()
    try:
        report = validate(require_gpu=args.require_gpu, lengths=tuple(args.lengths))
    except BaseException as error:
        atomic_json(
            args.output,
            {
                "passed": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "environment": environment_report(),
            },
        )
        raise
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
