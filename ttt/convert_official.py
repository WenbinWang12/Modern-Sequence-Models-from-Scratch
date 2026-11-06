"""将官方 Orbax 参数转换为本项目检查点。"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from .checkpoint import save_checkpoint
from .config import load_experiment
from .models.ttt_e2e import TTTE2EForCausalLM
from .training import TrainState


def _read(tree, *path):
    for name in path:
        if not isinstance(tree, dict) or name not in tree:
            raise ValueError(f"官方参数缺少路径：{'/'.join(path)}")
        tree = tree[name]
    return tree


def convert_parameters(weights, model):
    config = model.config
    if config.num_experts or config.loop_mode != "none" or "gated_delta" in config.layer_types or config.attention_gate:
        raise ValueError("官方转换仅适用于稠密 Attention/MLP 模型。")
    root = _read(weights, "language_model", "model")
    collection = _read(root, "h")
    split = "prefix_blocks" in collection
    layers = []

    def linear(source):
        # 官方 NormalLinear 保存 [输入, 输出]；本项目保存 [输出, 输入]。
        return {"weight": _read(source, "weight").T}

    def mlp(source):
        return {
            name: linear(_read(source, official))
            for name, official in (("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3"))
        }

    for index in range(config.num_hidden_layers):
        suffix = index >= config.prefix_length
        local_index = index - config.prefix_length if suffix else index
        block = (
            _read(collection, "suffix_blocks" if suffix else "prefix_blocks") if split else _read(collection, "blocks")
        )
        block = jax.tree.map(lambda value: value[local_index if split else index], block)
        attention = _read(block, "seq_modeling_block")
        layer = {
            "attention": {
                name: linear(_read(attention, official))
                for name, official in (("q_proj", "wq"), ("k_proj", "wk"), ("v_proj", "wv"), ("o_proj", "wo"))
            },
            "mlp": mlp(_read(block, "feed_forward")),
        }
        if config.qk_norm:
            layer["attention"].update({name: _read(attention, name, "weight") for name in ("q_norm", "k_norm")})
        for name, official in (
            ("attention_norm", "seq_norm"),
            ("attention_post_norm", "seq_post_norm"),
            ("mlp_norm", "ffn_norm"),
            ("mlp_post_norm", "ffn_post_norm"),
        ):
            layer[name] = _read(block, official, "weight")
        if suffix and config.use_fast_mlp:
            prime = (
                block if split else jax.tree.map(lambda value: value[local_index], _read(collection, "prime_storage"))
            )
            layer["fast_mlp"] = mlp(_read(prime, "feed_forward_prime"))
            layer["fast_norm"] = _read(prime, "ffn_prime_norm", "weight")
            layer["fast_post_norm"] = _read(prime, "ffn_prime_post_norm", "weight")
        layers.append(layer)
    params = {
        "embed_tokens": _read(root, "wte", "weight"),
        "norm": _read(root, "ln_f", "weight"),
        "layers": tuple(layers),
    }
    if not config.tie_word_embeddings:
        params["lm_head"] = linear(_read(weights, "language_model", "lm_head"))
    template = jax.eval_shape(model.init_parameters, jax.random.PRNGKey(0))
    if jax.tree.structure(params) != jax.tree.structure(template):
        raise ValueError("转换后的参数结构不匹配，请检查官方配置。")
    for (path, actual), expected in zip(jax.tree.flatten_with_path(params)[0], jax.tree.leaves(template)):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(
                f"官方参数 {jax.tree_util.keystr(path)} 的形状或精度不匹配：{actual.shape}/{actual.dtype}，预期 {expected.shape}/{expected.dtype}"
            )
    return params


def main():
    import orbax.checkpoint as ocp

    parser = argparse.ArgumentParser(description="转换已下载的官方 model_weights")
    parser.add_argument("--source", required=True, help="官方检查点中的 model_weights 目录")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    experiment = load_experiment(args.config)
    with ocp.PyTreeCheckpointer() as checkpointer:
        weights = checkpointer.restore(Path(args.source).resolve())
    params = convert_parameters(weights, TTTE2EForCausalLM(experiment.model))
    save_checkpoint(args.output, TrainState(params, None, jnp.asarray(0, dtype=jnp.int32)), experiment)


if __name__ == "__main__":
    main()
