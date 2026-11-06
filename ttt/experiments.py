"""生成消融实验与同 FLOPs 对照。"""

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

from .config import ExperimentConfig
from .inner_loop import InnerLoopConfig
from .models.ttt_e2e.configuration_ttt_e2e import TTTE2EConfig
from .training import TrainConfig

SIZES = {
    "125m": (768, 2048, 12, 12, 1664),
    "350m": (1024, 2816, 24, 16, 2304),
    "760m": (1536, 4096, 24, 16, 3328),
    "1b": (2048, 5376, 24, 32, 4352),
    "3b": (2560, 6912, 32, 32, 5632),
}


def attention_layout(layers, full_layers, *, position="prefix", suffix=0):
    available = layers - suffix if position == "prefix" else layers
    if not 0 <= full_layers <= available:
        raise ValueError("全注意力层数超过可放置的层数。")
    selected = {max(0, math.floor((index + 1) * available / (full_layers + 1))) for index in range(full_layers)}
    if position == "pre_ttt":
        selected = set(range(layers - suffix - full_layers, layers - suffix))
    elif position == "suffix":
        selected = set(range(layers - full_layers, layers))
    elif position != "prefix":
        raise ValueError("position 必须为 prefix、pre_ttt 或 suffix。")
    if min(selected, default=0) < 0:
        raise ValueError("全注意力布局越界。")
    return tuple("full_attention" if index in selected else "sliding_attention" for index in range(layers))


def measured_cost(measurements, name):
    record = measurements[name]
    if record.get("method") not in ("profiler", "xla_estimate"):
        raise ValueError("成本记录必须说明 method=profiler 或 xla_estimate。")
    value = record["training_flops_per_token"]
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("每 token 训练 FLOPs 必须为有限正数。")
    return value


def batch1(size, *, tokens=15_000_000_000, sequence_length=8192, batch_size=8, measurements=None):
    hidden, intermediate, layers, heads, narrowed = SIZES[size]
    steps = math.ceil(tokens / (sequence_length * batch_size))
    model = TTTE2EConfig(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        ttt_suffix_length=layers // 4,
        max_position_embeddings=32768 if size in ("125m", "350m") else 131072,
    )
    training = TrainConfig(
        steps=steps,
        batch_size=batch_size,
        sequence_length=sequence_length,
        warmup_steps=max(1, steps // 100),
        inner_warmup_steps=max(1, steps // 10),
        gradient_checkpointing=True,
        data_parallel=True,
        checkpoint_every=max(1, steps // 20),
        log_every=10,
    )
    enabled, disabled = InnerLoopConfig(), InnerLoopConfig(enabled=False)
    tokenizer = "meta-llama/Meta-Llama-3-8B"
    arms = {
        "A": ExperimentConfig(
            replace(model, use_fast_mlp=False, ttt_suffix_length=0, layer_types=("full_attention",) * layers),
            disabled,
            training,
            tokenizer,
        ),
        "D": ExperimentConfig(
            replace(
                model,
                use_fast_mlp=False,
                ttt_suffix_length=0,
                layer_types=tuple("full_attention" if (i + 1) % 4 == 0 else "gated_delta" for i in range(layers)),
            ),
            disabled,
            training,
            tokenizer,
        ),
        "E": ExperimentConfig(
            replace(model, intermediate_size=narrowed, fast_intermediate_size=narrowed), enabled, training, tokenizer
        ),
        "E_plus": ExperimentConfig(model, enabled, training, tokenizer),
    }
    pending = []
    for full in (2, 4):
        layout = attention_layout(layers, full, suffix=layers // 4)
        hybrid = replace(model, layer_types=layout)
        arms[f"S{full}"] = ExperimentConfig(
            replace(hybrid, use_fast_mlp=False, ttt_suffix_length=0), disabled, training, tokenizer
        )
        arms[f"C{full}"] = ExperimentConfig(hybrid, disabled, training, tokenizer)
        arms[f"F{full}"] = ExperimentConfig(hybrid, enabled, training, tokenizer)
        iso = f"C{full}_isoflop"
        if measurements is None:
            pending.append({"arm": iso, "required_measurements": [f"{size}/C{full}", f"{size}/F{full}"]})
        else:
            ratio = measured_cost(measurements, f"{size}/F{full}") / measured_cost(measurements, f"{size}/C{full}")
            arms[iso] = ExperimentConfig(hybrid, disabled, replace(training, steps=math.ceil(steps * ratio)), tokenizer)
    return arms, pending


def moe_ablations(size="125m", *, experts=8, tokens=100_000_000):
    arms, _ = batch1(size, tokens=tokens)
    seed = arms["F2"]
    model = replace(
        seed.model,
        num_experts=experts,
        num_experts_per_token=2,
        moe_intermediate_size=max(64, seed.model.intermediate_size // 4),
    )
    result = {"route_mlp": replace(seed, model=model)}
    result["route_e2e"] = replace(seed, model=replace(model, use_fast_mlp=False, adaptation_target="router"))
    result["route_delta"] = replace(
        seed,
        model=replace(model, use_fast_mlp=False, adaptation_target="delta_router", ttt_suffix_length=0),
        inner=InnerLoopConfig(enabled=False),
    )
    baseline = replace(model, use_fast_mlp=False, ttt_suffix_length=0)
    for mode in ("none", "early_ffn", "middle_block", "middle_attention"):
        current = replace(baseline, loop_mode=mode, loop_start=None, loop_end=None)
        result[f"loop_{mode}"] = replace(seed, model=current, inner=InnerLoopConfig(enabled=False))
    return result


def write_experiments(
    directory, sizes=("125m", "760m"), *, measurements=None, tokens=15_000_000_000, include_moe=False
):
    directory = Path(directory)
    manifest = {"artifact": "v0.16", "experiments": [], "pending": []}
    for size in sizes:
        arms, pending = batch1(size, tokens=tokens, measurements=measurements)
        if include_moe:
            arms.update(moe_ablations(size, tokens=tokens))
        for name, experiment in arms.items():
            path = directory / size / f"{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(experiment.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            manifest["experiments"].append(
                {
                    "size": size,
                    "arm": name,
                    "path": str(path),
                    "scheduled_tokens": experiment.training.steps
                    * experiment.training.batch_size
                    * experiment.training.sequence_length,
                }
            )
        manifest["pending"].extend({"size": size, **row} for row in pending)
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="生成 artifact 消融配置")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sizes", nargs="+", choices=tuple(SIZES), default=["125m", "760m"])
    parser.add_argument("--measurements")
    parser.add_argument("--tokens", type=int, default=15_000_000_000)
    parser.add_argument("--include-moe", action="store_true")
    args = parser.parse_args()
    measurements = json.loads(Path(args.measurements).read_text(encoding="utf-8")) if args.measurements else None
    print(
        json.dumps(
            write_experiments(
                args.output, args.sizes, measurements=measurements, tokens=args.tokens, include_moe=args.include_moe
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
