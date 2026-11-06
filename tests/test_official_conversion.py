"""官方保存布局中的层堆叠和线性矩阵方向。"""

import jax
import numpy as np

from ttt.convert_official import convert_parameters


def test_official_stacked_layout(model_and_params):
    model, params = model_and_params

    def linear(value):
        return {"weight": value["weight"].T}

    def mlp(value):
        return {"w1": linear(value["gate_proj"]), "w2": linear(value["down_proj"]), "w3": linear(value["up_proj"])}

    blocks = []
    for layer in params["layers"]:
        attention = layer["attention"]
        block = {
            "seq_modeling_block": {
                name: linear(attention[current])
                for name, current in (("wq", "q_proj"), ("wk", "k_proj"), ("wv", "v_proj"), ("wo", "o_proj"))
            },
            "feed_forward": mlp(layer["mlp"]),
        }
        block["seq_modeling_block"].update({name: {"weight": attention[name]} for name in ("q_norm", "k_norm")})
        block.update(
            {
                name: {"weight": layer[current]}
                for name, current in (
                    ("seq_norm", "attention_norm"),
                    ("seq_post_norm", "attention_post_norm"),
                    ("ffn_norm", "mlp_norm"),
                    ("ffn_post_norm", "mlp_post_norm"),
                )
            }
        )
        blocks.append(block)
    stacked = jax.tree.map(lambda *values: np.stack(values), *blocks)
    primes = [
        {
            "feed_forward_prime": mlp(layer["fast_mlp"]),
            "ffn_prime_norm": {"weight": layer["fast_norm"]},
            "ffn_prime_post_norm": {"weight": layer["fast_post_norm"]},
        }
        for layer in params["layers"]
        if "fast_mlp" in layer
    ]
    storage = jax.tree.map(lambda *values: np.stack(values), *primes)
    weights = {
        "language_model": {
            "model": {
                "wte": {"weight": params["embed_tokens"]},
                "ln_f": {"weight": params["norm"]},
                "h": {"blocks": stacked, "prime_storage": storage},
            }
        }
    }
    actual = convert_parameters(weights, model)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(params)):
        np.testing.assert_array_equal(a, b)
