"""TTT-E2E 模型配置。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar

from ..base_config import PretrainedConfig
from ..registry import register_config

__all__ = ["TTTE2EConfig"]


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数，实际为 {value!r}。")


def _positive_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须是有限的正数，实际为 {value!r}。")


@dataclass(frozen=True, kw_only=True)
class TTTE2EConfig(PretrainedConfig):
    """模型结构参数，层索引从 0 开始。"""

    model_type: ClassVar[str] = "ttt_e2e"

    # 模型尺寸
    vocab_size: int = 128256
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int | None = None  # 默认与查询头数相同。
    head_dim: int | None = None  # 默认取 hidden_size // num_attention_heads。
    tie_word_embeddings: bool = True

    # 注意力
    layer_types: tuple[str, ...] | None = None  # 默认所有层使用滑窗注意力。
    sliding_window: int = 8192
    max_position_embeddings: int = 32768
    rope_theta: float = 500000.0
    qk_norm: bool = True
    attention_bias: bool = False
    attention_backend: str = "chunked"
    attention_query_block_size: int = 128
    attention_key_block_size: int = 256
    attention_gate: bool = False
    gdn_conv_kernel_size: int = 4

    # FFN 与归一化
    hidden_act: str = "silu"
    pre_norm: bool = True
    post_norm: bool = True
    rms_norm_eps: float = 1e-6
    mlp_bias: bool = False
    initializer_range: float = 0.02
    loss_token_block_size: int = 64
    loss_vocab_block_size: int = 4096
    num_experts: int = 0
    num_experts_per_token: int = 2
    num_shared_experts: int = 1
    moe_intermediate_size: int | None = None
    first_k_dense_layers: int = 2
    router_aux_loss_coef: float = 0.0001
    router_z_loss_coef: float = 0.001
    router_bias_update_rate: float = 0.001
    adaptation_target: str = "mlp"
    router_adapter_rank: int = 8
    delta_memory_rank: int = 8
    loop_mode: str = "none"
    loop_start: int | None = None
    loop_end: int | None = None
    loop_repeats: int = 2

    # 快权重 MLP′
    use_fast_mlp: bool = True  # 是否添加 MLP′，更新开关由训练或推理模块管理。
    ttt_suffix_length: int | None = None  # 默认最后四分之一层，至少一层。
    fast_intermediate_size: int | None = None  # 默认与普通 FFN 相同。
    ttt_chunk_size: int = 1024  # 每次内循环更新使用的 token 数。

    # 精度与特殊 token
    param_dtype: str = "float32"
    compute_dtype: str = "bfloat16"
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "sliding_window",
            "max_position_embeddings",
            "ttt_chunk_size",
            "attention_query_block_size",
            "attention_key_block_size",
            "loss_token_block_size",
            "loss_vocab_block_size",
            "gdn_conv_kernel_size",
            "num_experts_per_token",
            "router_adapter_rank",
            "delta_memory_rank",
            "loop_repeats",
        ):
            _positive_int(name, getattr(self, name))
        for name in (
            "tie_word_embeddings",
            "qk_norm",
            "attention_bias",
            "pre_norm",
            "post_norm",
            "mlp_bias",
            "use_fast_mlp",
            "attention_gate",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} 必须是布尔值。")
        for name in ("rope_theta", "rms_norm_eps", "initializer_range"):
            _positive_float(name, getattr(self, name))
        if self.attention_backend not in ("reference", "chunked", "cudnn"):
            raise ValueError("attention_backend 必须为 reference、chunked 或 cudnn。")

        kv_heads = self.num_key_value_heads
        if kv_heads is None:
            kv_heads = self.num_attention_heads
        _positive_int("num_key_value_heads", kv_heads)
        if self.num_attention_heads % kv_heads:
            raise ValueError("num_attention_heads 必须能被 num_key_value_heads 整除。")
        object.__setattr__(self, "num_key_value_heads", kv_heads)

        head_dim = self.head_dim
        if head_dim is None:
            if self.hidden_size % self.num_attention_heads:
                raise ValueError("未指定 head_dim 时，hidden_size 必须能被 num_attention_heads 整除。")
            head_dim = self.hidden_size // self.num_attention_heads
        _positive_int("head_dim", head_dim)
        if head_dim % 2:
            raise ValueError("旋转位置编码要求 head_dim 为偶数。")
        object.__setattr__(self, "head_dim", head_dim)

        layer_types = self.layer_types
        if layer_types is None:
            layer_types = ("sliding_attention",) * self.num_hidden_layers
        if not isinstance(layer_types, (tuple, list)):
            raise ValueError("layer_types 必须是元组或列表，每层对应一个注意力类型。")
        layer_types = tuple(layer_types)
        if len(layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types 的元素数量必须等于 num_hidden_layers。")
        if any(kind not in ("sliding_attention", "full_attention", "gated_delta") for kind in layer_types):
            raise ValueError("注意力类型必须为 sliding_attention、full_attention 或 gated_delta。")
        object.__setattr__(self, "layer_types", layer_types)

        suffix_length = self.ttt_suffix_length
        if suffix_length is None:
            suffix_length = (
                max(1, self.num_hidden_layers // 4) if self.use_fast_mlp or self.adaptation_target == "router" else 0
            )
        if isinstance(suffix_length, bool) or not isinstance(suffix_length, int):
            raise ValueError("ttt_suffix_length 必须是整数。")
        if self.use_fast_mlp or self.adaptation_target == "router":
            if not 1 <= suffix_length <= self.num_hidden_layers:
                raise ValueError("启用快权重 MLP 时，ttt_suffix_length 必须位于 [1, num_hidden_layers]。")
            if "sliding_attention" in layer_types and self.ttt_chunk_size > self.sliding_window:
                raise ValueError("启用快权重 MLP 时，滑窗长度 sliding_window 必须覆盖更新块 ttt_chunk_size。")
        elif suffix_length != 0:
            raise ValueError("use_fast_mlp=False 时，ttt_suffix_length 必须为 0。")
        object.__setattr__(self, "ttt_suffix_length", suffix_length)

        fast_width = self.fast_intermediate_size
        if fast_width is None:
            fast_width = self.intermediate_size
        _positive_int("fast_intermediate_size", fast_width)
        object.__setattr__(self, "fast_intermediate_size", fast_width)
        for name in ("num_experts", "num_shared_experts", "first_k_dense_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} 必须是非负整数。")
        if self.num_experts and self.num_experts_per_token > self.num_experts:
            raise ValueError("num_experts_per_token 不能大于 num_experts。")
        width = self.intermediate_size if self.moe_intermediate_size is None else self.moe_intermediate_size
        _positive_int("moe_intermediate_size", width)
        object.__setattr__(self, "moe_intermediate_size", width)
        for name in ("router_aux_loss_coef", "router_z_loss_coef", "router_bias_update_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是有限的非负数。")
        if self.adaptation_target not in ("mlp", "router", "delta_router"):
            raise ValueError("adaptation_target 必须为 mlp、router 或 delta_router。")
        if self.adaptation_target != "mlp" and not self.num_experts:
            raise ValueError("路由适应需要启用 MoE。")
        if self.adaptation_target != "mlp" and self.use_fast_mlp:
            raise ValueError("路由适应配置应关闭 use_fast_mlp，以分别比较三条适应路线。")
        if self.adaptation_target == "router":
            if self.num_experts_per_token < 2:
                raise ValueError("可微路由适应至少需要 top-2 路由。")
            if self.num_hidden_layers - suffix_length < self.first_k_dense_layers:
                raise ValueError("路由适应的后段必须全部为 MoE 层。")
        if self.loop_mode not in ("none", "early_ffn", "middle_block", "middle_attention"):
            raise ValueError("未知的 loop_mode。")
        start = self.loop_start
        end = self.loop_end
        if start is None:
            start = 0 if self.loop_mode == "early_ffn" else self.num_hidden_layers // 4
        if end is None:
            end = (
                min(3, self.prefix_length, self.first_k_dense_layers if self.num_experts else 3)
                if self.loop_mode == "early_ffn"
                else min(self.num_hidden_layers * 3 // 4, self.prefix_length)
            )
        if self.loop_mode != "none" and not (0 <= start < end <= self.prefix_length):
            raise ValueError("循环范围必须位于不适应快权重的前段，且不能为空。")
        if self.loop_mode == "early_ffn" and self.num_experts and end > self.first_k_dense_layers:
            raise ValueError("early_ffn 只能循环前段稠密 FFN。")
        object.__setattr__(self, "loop_start", start)
        object.__setattr__(self, "loop_end", end)

        if self.hidden_act != "silu":
            raise ValueError("初版 TTT-E2E 使用 SwiGLU，hidden_act 必须为 'silu'。")
        for name in ("param_dtype", "compute_dtype"):
            if getattr(self, name) not in ("float32", "bfloat16", "float16"):
                raise ValueError(f"{name} 必须为 'float32'、'bfloat16' 或 'float16'。")
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            token_id = getattr(self, name)
            if token_id is not None and (
                isinstance(token_id, bool) or not isinstance(token_id, int) or not 0 <= token_id < self.vocab_size
            ):
                raise ValueError(f"{name} 必须为 None 或位于 [0, vocab_size) 的整数。")

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        """使用全注意力、保留完整因果历史的层索引，从 0 开始。"""
        return tuple(i for i, kind in enumerate(self.layer_types) if kind == "full_attention")

    @property
    def ttt_layer_indices(self) -> tuple[int, ...]:
        """包含额外 MLP′ 的层索引，从 0 开始。"""
        return tuple(range(self.num_hidden_layers - self.ttt_suffix_length, self.num_hidden_layers))

    @property
    def prefix_length(self) -> int:
        """前段不包含 MLP′ 的层数。"""
        return self.num_hidden_layers - self.ttt_suffix_length

    def to_dict(self) -> dict[str, Any]:
        """转为可序列化为 JSON/YAML 的字典，包含已经补全的默认值。"""
        result = asdict(self)
        result["layer_types"] = list(self.layer_types)
        return {"model_type": self.model_type, **result}

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> TTTE2EConfig:
        """从字典加载配置，拒绝拼错的字段名和其他模型类型。"""
        values = dict(config)
        model_type = values.pop("model_type", cls.model_type)
        if model_type != cls.model_type:
            raise ValueError(f"model_type 应为 {cls.model_type!r}，实际为 {model_type!r}。")
        unknown = values.keys() - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"未知的 TTTE2EConfig 字段：{', '.join(sorted(unknown))}。")
        return cls(**values)


register_config(TTTE2EConfig)
