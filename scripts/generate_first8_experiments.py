#!/usr/bin/env python3
"""Generate the first eight-GPU screening configurations.

The generated file names are deliberately self-describing.  They are used as
the output directory and as the future W&B run name, so an experiment can be
identified from a table or checkpoint without opening its JSON configuration.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

VOCAB_SIZE = 50_257
NOMINAL_PARAMETERS = 125_000_000
DATASET = "dolma3"
OPTIMIZER = "adamw"
SEED = 0
DEFAULT_TOKENS = 3_000_000_000
DEFAULT_BATCH_SIZE = 64
DEFAULT_SEQUENCE_LENGTH = 4096
DEFAULT_SMOKE_STEPS = 100


def _tpp_label(tokens: int) -> str:
    ratio = tokens / NOMINAL_PARAMETERS
    rounded = round(ratio)
    if ratio >= 1 and abs(ratio - rounded) < 1e-9:
        return f"{rounded:03d}"
    return f"{ratio:.1f}".replace(".", "p")


def _token_label(tokens: int) -> str:
    if tokens % 1_000_000_000 == 0:
        return f"{tokens // 1_000_000_000}b"
    if tokens % 1_000_000 == 0:
        return f"{tokens // 1_000_000}m"
    return str(tokens)


def _window_label(window: int | None) -> str:
    return "na" if window is None else f"{window:03d}"


def _chunk_label(chunk: int) -> str:
    return f"{chunk:03d}"


def run_name(
    *,
    wave: str,
    mixer: str,
    adaptation: str,
    window: int | None,
    chunk: int,
    tokens: int,
    sequence_length: int,
    batch_size: int,
    seed: int = SEED,
) -> str:
    """Return the stable, factor-explicit name used by W&B and checkpoints."""

    token_label = _token_label(tokens)
    context_label = f"{sequence_length // 1024}k" if sequence_length % 1024 == 0 else str(sequence_length)
    return (
        f"{wave}__data-{DATASET}__mixer-{mixer}__adapt-{adaptation}"
        f"__w-{_window_label(window)}__c-{_chunk_label(chunk)}"
        f"__p-125m__tpp-{_tpp_label(tokens)}__tok-{token_label}__ctx-{context_label}"
        f"__b-{batch_size}__opt-{OPTIMIZER}__seed-{seed:03d}"
    )


def model_config(*, mixer: str, window: int | None, chunk: int, adaptation: str) -> dict[str, Any]:
    """Build the common 125M-class model, changing only the mixer factor."""

    if mixer == "gdn":
        layer_types = ["gated_delta"] * 12
        kv_heads = 12
    else:
        layer_types = ["sliding_attention"] * 12
        kv_heads = {"mha": 12, "gqa": 4, "mqa": 1}[mixer]

    # GDN does not use a sliding attention window.  A valid positive value is
    # retained in the config because the shared model schema requires one.
    effective_window = 128 if window is None else window
    use_fast_mlp = adaptation != "swa"
    return {
        "model_type": "ttt_e2e",
        "vocab_size": VOCAB_SIZE,
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": kv_heads,
        "head_dim": 64,
        "tie_word_embeddings": True,
        "layer_types": layer_types,
        "sliding_window": effective_window,
        "max_position_embeddings": 65_536,
        "rope_theta": 500_000.0,
        "qk_norm": True,
        "attention_bias": False,
        "attention_backend": "chunked",
        "attention_query_block_size": 64,
        "attention_key_block_size": 64,
        "attention_gate": False,
        "gdn_conv_kernel_size": 4,
        "hidden_act": "silu",
        "pre_norm": True,
        "post_norm": False,
        "rms_norm_eps": 1e-6,
        "mlp_bias": False,
        "initializer_range": 0.02,
        "loss_token_block_size": 64,
        "loss_vocab_block_size": 4096,
        "multi_token_heads": 1,
        "multi_token_weight": 0.3,
        "num_experts": 0,
        "num_experts_per_token": 2,
        "num_shared_experts": 1,
        "moe_intermediate_size": None,
        "first_k_dense_layers": 2,
        "router_aux_loss_coef": 0.0001,
        "router_z_loss_coef": 0.001,
        "router_bias_update_rate": 0.001,
        "adaptation_target": "mlp",
        "router_adapter_rank": 8,
        "delta_memory_rank": 8,
        "loop_mode": "none",
        "loop_start": None,
        "loop_end": None,
        "loop_repeats": 2,
        "use_fast_mlp": use_fast_mlp,
        "ttt_suffix_length": 3 if use_fast_mlp else 0,
        "fast_intermediate_size": 2048,
        "ttt_chunk_size": chunk,
        "param_dtype": "float32",
        "compute_dtype": "bfloat16",
        "bos_token_id": VOCAB_SIZE - 1,
        "eos_token_id": VOCAB_SIZE - 1,
        "pad_token_id": VOCAB_SIZE - 1,
    }


def experiment_config(
    *,
    name: str,
    wave: str,
    mixer: str,
    adaptation: str,
    window: int | None,
    chunk: int,
    tokens: int,
    batch_size: int,
    sequence_length: int,
    steps: int,
    seed: int,
    wandb_project: str,
    wandb_entity: str,
) -> dict[str, Any]:
    """Build one complete ExperimentConfig-compatible dictionary."""

    return {
        "model": model_config(mixer=mixer, window=window, chunk=chunk, adaptation=adaptation),
        "inner": {
            # Keep the fast branch in both arms.  The no-update arm isolates
            # the effect of per-sequence TTT updates at matched capacity.
            "enabled": adaptation == "ttt-e2e",
            "learning_rate": 1.0,
            "clip_norm": 1.0,
            "track_post_loss": False,
            "first_order": False,
            "multi_token": False,
        },
        "training": {
            "steps": steps,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "learning_rate": 6e-4,
            "weight_decay": 0.1,
            "clip_norm": 1.0,
            "warmup_steps": max(1, steps // 20),
            "inner_warmup_steps": max(1, steps // 100),
            "inner_lr_start": 0.1,
            "seed": seed,
            "checkpoint_every": max(1, steps // 10),
            "log_every": max(1, steps // 100),
            "gradient_checkpointing": True,
            "data_parallel": True,
            "checkpoint_format": "orbax",
            "parallelism": "auto",
            "expert_parallel_size": 1,
            "shard_parameters": True,
        },
        "tokenizer": "tiktoken:gpt2",
        "metadata": {
            "run_name": name,
            "wave": wave,
            "factors": {
                "dataset": DATASET,
                "mixer": mixer,
                "adaptation": adaptation,
                "attention_window": window,
                "ttt_chunk_size": chunk,
                "parameter_class": "125m",
                "nominal_parameters": NOMINAL_PARAMETERS,
                "tokens": tokens,
                "tpp": tokens / NOMINAL_PARAMETERS,
                "sequence_length": sequence_length,
                "batch_size": batch_size,
                "optimizer": OPTIMIZER,
                "seed": seed,
            },
            "wandb": {
                "project": wandb_project,
                "entity": wandb_entity,
                "api_key": "",
            },
        },
    }


def _specs(
    tokens: int,
    batch_size: int,
    sequence_length: int,
    smoke_steps: int,
    wandb_project: str,
    wandb_entity: str,
) -> list[dict[str, Any]]:
    steps = math.ceil(tokens / (batch_size * sequence_length))
    specs: list[dict[str, Any]] = []

    # Wave A isolates the attention-window and update factors with MHA fixed.
    for window in (64, 128, 256, 512):
        for adaptation in ("no-update", "ttt-e2e"):
            name = run_name(
                wave="f8a",
                mixer="mha",
                adaptation=adaptation,
                window=window,
                chunk=window,
                tokens=tokens,
                sequence_length=sequence_length,
                batch_size=batch_size,
            )
            specs.append(
                {
                    "name": name,
                    "wave": "wave_a",
                    "mixer": "mha",
                    "adaptation": adaptation,
                    "window": window,
                    "chunk": window,
                    "config": experiment_config(
                        name=name,
                        wave="wave_a",
                        mixer="mha",
                        adaptation=adaptation,
                        window=window,
                        chunk=window,
                        tokens=tokens,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        steps=steps,
                        seed=SEED,
                        wandb_project=wandb_project,
                        wandb_entity=wandb_entity,
                    ),
                }
            )

    # Plain sliding-window attention is the matched-capacity anchor.  It uses
    # the same 128-token attention window while removing the fast MLP branch.
    anchor_window = 128
    anchor_adaptation = "swa"
    anchor_name = run_name(
        wave="f8a",
        mixer="mha",
        adaptation=anchor_adaptation,
        window=anchor_window,
        chunk=anchor_window,
        tokens=tokens,
        sequence_length=sequence_length,
        batch_size=batch_size,
    )
    specs.append(
        {
            "name": anchor_name,
            "wave": "wave_a",
            "mixer": "mha",
            "adaptation": anchor_adaptation,
            "window": anchor_window,
            "chunk": anchor_window,
            "config": experiment_config(
                name=anchor_name,
                wave="wave_a",
                mixer="mha",
                adaptation=anchor_adaptation,
                window=anchor_window,
                chunk=anchor_window,
                tokens=tokens,
                batch_size=batch_size,
                sequence_length=sequence_length,
                steps=steps,
                seed=SEED,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
            ),
        }
    )

    # Wave B compares mixer variants at a common 128-token attention/update
    # block.  GDN has no attention window, so its name records w-na.
    mixers: tuple[tuple[str, int | None], ...] = (("mha", 128), ("gqa", 128), ("mqa", 128), ("gdn", None))
    for mixer, window in mixers:
        for adaptation in ("no-update", "ttt-e2e"):
            name = run_name(
                wave="f8b",
                mixer=mixer,
                adaptation=adaptation,
                window=window,
                chunk=128,
                tokens=tokens,
                sequence_length=sequence_length,
                batch_size=batch_size,
            )
            specs.append(
                {
                    "name": name,
                    "wave": "wave_b",
                    "mixer": mixer,
                    "adaptation": adaptation,
                    "window": window,
                    "chunk": 128,
                    "config": experiment_config(
                        name=name,
                        wave="wave_b",
                        mixer=mixer,
                        adaptation=adaptation,
                        window=window,
                        chunk=128,
                        tokens=tokens,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        steps=steps,
                        seed=SEED,
                        wandb_project=wandb_project,
                        wandb_entity=wandb_entity,
                    ),
                }
            )

    smoke_tokens = batch_size * sequence_length * smoke_steps
    smoke_name = (
        run_name(
            wave="f8s",
            mixer="mha",
            adaptation="no-update",
            window=128,
            chunk=128,
            tokens=smoke_tokens,
            sequence_length=sequence_length,
            batch_size=batch_size,
        )
        .replace(f"__tpp-{_tpp_label(smoke_tokens)}", "__tpp-smoke")
        .replace(f"__tok-{_token_label(smoke_tokens)}", "__tok-smoke")
    )
    specs.append(
        {
            "name": smoke_name,
            "wave": "smoke",
            "mixer": "mha",
            "adaptation": "no-update",
            "window": 128,
            "chunk": 128,
            "config": experiment_config(
                name=smoke_name,
                wave="smoke",
                mixer="mha",
                adaptation="no-update",
                window=128,
                chunk=128,
                tokens=smoke_tokens,
                batch_size=batch_size,
                sequence_length=sequence_length,
                steps=smoke_steps,
                seed=SEED,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
            ),
        }
    )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Directory for JSON configs and manifest")
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--smoke-steps", type=int, default=DEFAULT_SMOKE_STEPS)
    parser.add_argument("--wandb-project", default="ttt-first8")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--output-root", default="outputs/experiments/first8")
    args = parser.parse_args()
    if args.tokens <= 0 or args.batch_size <= 0 or args.sequence_length <= 0 or args.smoke_steps <= 0:
        parser.error("tokens, batch-size, sequence-length, and smoke-steps must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    specs = _specs(
        args.tokens,
        args.batch_size,
        args.sequence_length,
        args.smoke_steps,
        args.wandb_project,
        args.wandb_entity,
    )
    manifest = {
        "schema_version": 1,
        "dataset": DATASET,
        "optimizer": OPTIMIZER,
        "nominal_parameters": NOMINAL_PARAMETERS,
        "tokens": args.tokens,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "smoke_steps": args.smoke_steps,
        "output_root": args.output_root,
        "wandb": {"project": args.wandb_project, "entity": args.wandb_entity, "api_key": ""},
        "runs": [],
    }
    for spec in specs:
        config_path = args.output / f"{spec['name']}.json"
        config_path.write_text(json.dumps(spec["config"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["runs"].append(
            {
                "name": spec["name"],
                "wave": spec["wave"],
                "mixer": spec["mixer"],
                "adaptation": spec["adaptation"],
                "window": spec["window"],
                "chunk": spec["chunk"],
                "config": config_path.name,
                "output": f"{args.output_root.rstrip('/')}/{spec['name']}",
            }
        )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Generated {len(specs)} configs in {args.output}")
    for item in manifest["runs"]:
        print(item["name"])


if __name__ == "__main__":
    main()
