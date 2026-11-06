"""TTT-E2E 的模型结构与前向计算。"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..registry import register_model
from .attention import KVCache, attention, empty_cache, grow_cache, init_attention
from .gated_delta import DeltaCache, empty_delta_cache, gated_delta, init_gated_delta
from .layers import init_linear, init_mlp, init_norm, linear, rms_norm, swiglu
from .moe import init_moe, init_router_adapter, moe_forward
from .utils import TTTE2EPreTrainedModel

__all__ = ["ModelOutput", "LayerState", "TTTE2EForCausalLM"]


class ModelOutput(NamedTuple):
    logits: jax.Array | None
    cache: tuple | None
    aux: dict | None = None


class LayerState(NamedTuple):
    attention: object
    memory: jax.Array


class TTTE2EForCausalLM(TTTE2EPreTrainedModel):
    def __init__(self, config, gradient_checkpointing=False):
        super().__init__(config)
        self.gradient_checkpointing = gradient_checkpointing

    def is_moe(self, index):
        return self.config.num_experts > 0 and index >= self.config.first_k_dense_layers

    def loop_mode(self, index):
        config = self.config
        return config.loop_mode if config.loop_start <= index < config.loop_end else "none"

    def init_parameters(self, key):
        config = self.config
        keys = jax.random.split(key, config.num_hidden_layers + 2)
        embedding = jax.random.normal(keys[0], (config.vocab_size, config.hidden_size))
        params = {
            "embed_tokens": (embedding * config.initializer_range).astype(config.param_dtype),
            "norm": init_norm(config.hidden_size, config),
        }
        layers = []
        for index in range(config.num_hidden_layers):
            layer_keys = jax.random.split(keys[index + 1], 4)
            initializer = init_gated_delta if config.layer_types[index] == "gated_delta" else init_attention
            layer = {
                "attention": initializer(layer_keys[0], config),
                "attention_norm": init_norm(config.hidden_size, config),
                "attention_post_norm": init_norm(config.hidden_size, config),
                "mlp_norm": init_norm(config.hidden_size, config),
                "mlp_post_norm": init_norm(config.hidden_size, config),
            }
            if self.is_moe(index):
                layer["moe"] = init_moe(layer_keys[1], config)
            else:
                layer["mlp"] = init_mlp(layer_keys[1], config, config.intermediate_size)
            if index in config.ttt_layer_indices:
                if config.use_fast_mlp:
                    layer["fast_mlp"] = init_mlp(layer_keys[2], config, config.fast_intermediate_size)
                    layer["fast_norm"] = init_norm(config.hidden_size, config)
                    layer["fast_post_norm"] = init_norm(config.hidden_size, config)
                if config.adaptation_target == "router":
                    layer["fast_router"] = init_router_adapter(layer_keys[3], config)
            layers.append(layer)
        params["layers"] = tuple(layers)
        if not config.tie_word_embeddings:
            params["lm_head"] = init_linear(keys[-1], config.hidden_size, config.vocab_size, config)
        return params

    def initial_fast_weights(self, params):
        name = "fast_router" if self.config.adaptation_target == "router" else "fast_mlp"
        return tuple(
            params["layers"][index][name] for index in self.config.ttt_layer_indices if name in params["layers"][index]
        )

    def _empty_layer_state(self, index, capacity):
        config = self.config

        def mixer_state():
            if config.layer_types[index] == "gated_delta":
                return empty_delta_cache(config)
            length = (
                min(capacity, config.sliding_window) if config.layer_types[index] == "sliding_attention" else capacity
            )
            return empty_cache(config, length)

        def block_state():
            current = (
                tuple(mixer_state() for _ in range(config.loop_repeats))
                if self.loop_mode(index) == "middle_attention"
                else mixer_state()
            )
            if self.is_moe(index) and config.adaptation_target == "delta_router":
                current = LayerState(
                    current, jnp.zeros((config.delta_memory_rank, config.delta_memory_rank), dtype=jnp.float32)
                )
            return current

        if self.loop_mode(index) == "middle_block":
            return tuple(block_state() for _ in range(config.loop_repeats))
        return block_state()

    def init_cache(self, layer_indices, capacity):
        return tuple(self._empty_layer_state(index, capacity) for index in layer_indices)

    def grow_layer_cache(self, index, cache, capacity):
        if isinstance(cache, LayerState):
            return LayerState(self.grow_layer_cache(index, cache.attention, capacity), cache.memory)
        if isinstance(cache, DeltaCache):
            return cache
        if isinstance(cache, KVCache):
            sliding = self.config.layer_types[index] == "sliding_attention"
            return grow_cache(cache, min(capacity, self.config.sliding_window) if sliding else capacity, sliding)
        return tuple(self.grow_layer_cache(index, item, capacity) for item in cache)

    def empty_aux(self):
        return {
            "loads": jnp.zeros((self.config.num_hidden_layers, self.config.num_experts)),
            "aux": jnp.array(0.0),
            "z": jnp.array(0.0),
            "blocks": jnp.array(0.0),
        }

    def _attention_branch(self, params, hidden, kind, cache, position):
        config = self.config
        branch = rms_norm(hidden, params["attention_norm"], config.rms_norm_eps) if config.pre_norm else hidden
        if kind == "gated_delta":
            branch, cache = gated_delta(params["attention"], branch, config, cache)
        else:
            branch, cache = attention(params["attention"], branch, config, kind, cache=cache, position=position)
        if config.post_norm:
            branch = rms_norm(branch, params["attention_post_norm"], config.rms_norm_eps)
        return hidden + branch, cache

    def _block(self, params, hidden, index, fast_weights, cache, position, token_mask, load_index=None):
        config = self.config
        memory = cache.memory if isinstance(cache, LayerState) else None
        mixer_cache = cache.attention if isinstance(cache, LayerState) else cache
        attention_repeats = config.loop_repeats if self.loop_mode(index) == "middle_attention" else 1
        mixer_states = []
        for visit in range(attention_repeats):
            current_cache = mixer_cache[visit] if mixer_cache is not None and attention_repeats > 1 else mixer_cache
            hidden, current_cache = self._attention_branch(
                params, hidden, config.layer_types[index], current_cache, position
            )
            mixer_states.append(current_cache)
        mixer_cache = tuple(mixer_states) if attention_repeats > 1 else mixer_states[0]
        if "fast_mlp" in params:
            current_fast = fast_weights if config.adaptation_target == "mlp" else params["fast_mlp"]
            branch = rms_norm(hidden, params["fast_norm"], config.rms_norm_eps) if config.pre_norm else hidden
            branch = swiglu(current_fast, branch, config.compute_dtype)
            if config.post_norm:
                branch = rms_norm(branch, params["fast_post_norm"], config.rms_norm_eps)
            hidden = hidden + branch
        aux = self.empty_aux()
        repeats = config.loop_repeats if self.loop_mode(index) == "early_ffn" else 1
        for _ in range(repeats):
            branch = rms_norm(hidden, params["mlp_norm"], config.rms_norm_eps) if config.pre_norm else hidden
            if "moe" in params:
                router = fast_weights if config.adaptation_target == "router" else None
                branch, memory, statistics = moe_forward(params["moe"], branch, config, router, memory, token_mask)
                aux["loads"] = aux["loads"].at[index if load_index is None else load_index].add(statistics["loads"])
                aux["aux"] += statistics["aux"]
                aux["z"] += statistics["z"]
                aux["blocks"] += 1
            else:
                branch = swiglu(params["mlp"], branch, config.compute_dtype)
            if config.post_norm:
                branch = rms_norm(branch, params["mlp_post_norm"], config.rms_norm_eps)
            hidden = hidden + branch
        if cache is None:
            next_cache = None
        else:
            next_cache = LayerState(mixer_cache, memory) if isinstance(cache, LayerState) else mixer_cache
        return hidden, next_cache, aux

    def _run_layer(self, layer, hidden, index, fast, layer_cache, position, token_mask, load_index):
        totals = self.empty_aux()
        repeats = self.config.loop_repeats if self.loop_mode(index) == "middle_block" else 1
        visits = []
        for visit in range(repeats):
            current_cache = layer_cache[visit] if layer_cache is not None and repeats > 1 else layer_cache

            def block(weights, x, current_fast, current_state, mask, row):
                return self._block(weights, x, index, current_fast, current_state, position, mask, row)

            function = jax.checkpoint(block) if self.gradient_checkpointing else block
            hidden, current_cache, auxiliary = function(layer, hidden, fast, current_cache, token_mask, load_index)
            totals = jax.tree.map(jnp.add, totals, auxiliary)
            visits.append(current_cache)
        return hidden, tuple(visits) if repeats > 1 else visits[0], totals

    def _layers(self, params, hidden, indices, fast_weights, cache, position, token_mask=None, return_aux=False):
        indices = tuple(indices)
        next_cache, totals = [], self.empty_aux()

        def signature(index):
            return (
                self.config.layer_types[index],
                self.is_moe(index),
                index in self.config.ttt_layer_indices,
                self.loop_mode(index),
            )

        def fast_at(index):
            offset = index - self.config.prefix_length
            return fast_weights[offset] if offset >= 0 and fast_weights else None

        begin = 0
        while begin < len(indices):
            index = indices[begin]
            end = begin + 1
            if self.gradient_checkpointing:
                while end < len(indices) and signature(indices[end]) == signature(index):
                    end += 1
            if end - begin == 1:
                hidden, current_cache, auxiliary = self._run_layer(
                    params["layers"][index],
                    hidden,
                    index,
                    fast_at(index),
                    None if cache is None else cache[begin],
                    position,
                    token_mask,
                    index,
                )
                next_cache.append(current_cache)
            else:
                # 参数仍按原格式保存；相邻同构层共用一个扫描体，避免展开巨大的编译图。
                group = indices[begin:end]
                layers = jax.tree.map(lambda *values: jnp.stack(values), *(params["layers"][i] for i in group))
                fast = jax.tree.map(lambda *values: jnp.stack(values), *(fast_at(i) for i in group))
                states = None if cache is None else jax.tree.map(lambda *values: jnp.stack(values), *cache[begin:end])

                def step(x, batch):
                    layer, current_fast, state, row = batch
                    x, state, statistics = self._run_layer(
                        layer, x, index, current_fast, state, position, token_mask, row
                    )
                    return x, (state, statistics)

                hidden, (states, statistics) = jax.lax.scan(
                    step, hidden, (layers, fast, states, jnp.asarray(group, dtype=jnp.int32))
                )
                auxiliary = jax.tree.map(lambda values: jnp.sum(values, axis=0), statistics)
                next_cache.extend(jax.tree.map(lambda value, i=i: value[i], states) for i in range(len(group)))
            totals = jax.tree.map(jnp.add, totals, auxiliary)
            begin = end
        result = (hidden, tuple(next_cache) if cache is not None else None)
        return (*result, totals) if return_aux else result

    def prefix(self, params, input_ids, cache=None, position=0, *, token_mask=None, return_aux=False):
        hidden = params["embed_tokens"][input_ids].astype(self.config.compute_dtype)
        return self._layers(
            params, hidden, range(self.config.prefix_length), (), cache, position, token_mask, return_aux
        )

    def suffix_hidden(self, params, hidden, fast_weights, cache=None, position=0, *, token_mask=None, return_aux=False):
        result = self._layers(
            params,
            hidden,
            range(self.config.prefix_length, self.config.num_hidden_layers),
            fast_weights,
            cache,
            position,
            token_mask,
            return_aux,
        )
        hidden = rms_norm(result[0], params["norm"], self.config.rms_norm_eps)
        return (hidden, *result[1:])

    def output_projection(self, params):
        if self.config.tie_word_embeddings:
            return params["embed_tokens"], None
        head = params["lm_head"]
        return head["weight"], head.get("bias")

    def project_logits(self, params, hidden):
        if self.config.tie_word_embeddings:
            logits = hidden @ params["embed_tokens"].astype(self.config.compute_dtype).T
        else:
            logits = linear(params["lm_head"], hidden, self.config.compute_dtype)
        return logits.astype(jnp.float32)

    def suffix(self, params, hidden, fast_weights, cache=None, position=0):
        hidden, cache = self.suffix_hidden(params, hidden, fast_weights, cache, position)
        return ModelOutput(self.project_logits(params, hidden), cache)

    def forward(self, params=None, input_ids=None, fast_weights=None, cache=None, position=0):
        if params is None:
            params = self.params
        if params is None:
            raise ValueError("forward 需要传入 params，或先调用 load_parameters。")
        if input_ids is None:
            raise ValueError("forward 需要传入 input_ids。")
        if fast_weights is None:
            fast_weights = self.initial_fast_weights(params)
        split = self.config.prefix_length
        prefix_cache = None if cache is None else cache[:split]
        suffix_cache = None if cache is None else cache[split:]
        hidden, prefix_cache = self.prefix(params, input_ids, prefix_cache, position)
        output = self.suffix(params, hidden, fast_weights, suffix_cache, position)
        return ModelOutput(output.logits, None if cache is None else prefix_cache + output.cache)


register_model(TTTE2EForCausalLM)
