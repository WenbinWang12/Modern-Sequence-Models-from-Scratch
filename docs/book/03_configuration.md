# 3. 配置、词表与维度

## 问题

词表不是一个孤立的 tokenizer 参数。词表大小会直接决定输入 embedding 的第一维；如果使用 tied embeddings，输出分类头也共享这个矩阵，因此分类头的词表维度随之固定。

## 代码

- `ttt/models/ttt_e2e/configuration_ttt_e2e.py`：模型尺寸、词表、层类型、MLP′、MoE、Loop 和 dtype；
- `ttt/models/base_config.py`：配置保存和加载；
- `ttt/models/auto.py`：从 `model_type` 自动构建配置和模型。

配置里最重要的尺寸关系是：

```text
embed_tokens: [vocab_size, hidden_size]
lm_head:      [hidden_size, vocab_size]  # tie_word_embeddings=False
```

当 `tie_word_embeddings=True` 时，`lm_head` 不再单独分配参数，而是使用 `embed_tokens` 的转置。

## 实验

```python
from ttt.models import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("configs/runs/tiny_gpu.json")
model = AutoModelForCausalLM.from_config(config, mode="train")
```

## 证据

`tests/test_configuration.py` 检查默认值、非法维度、C/F 配置配对、配置保存/加载和自动注册。

## 下一步

下一章沿着一次 token 的前向路径，观察 Attention、MLP′、原 FFN 和快权重更新之间的顺序。
