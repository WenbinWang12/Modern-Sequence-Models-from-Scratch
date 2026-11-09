# 13. Configuration, vocabulary, and dimensions

## Configuration is an architecture contract

Vocabulary size fixes the number of embedding rows. With tied embeddings, the output head shares the same matrix. Hidden size fixes the interfaces of attention, MLPs, normalization, and residuals. Sequence length changes positional state, cache size, and memory.

Configuration should therefore express structural constraints, not hide choices inside forward code.

## Four layers of configuration

| Layer | Examples | Effect |
|---|---|---|
| Model | hidden_size, num_layers, num_heads | tensor shapes and forward graph |
| Vocabulary | vocab_size, special IDs, tied embeddings | embedding and lm_head |
| Fast weights | inner.enabled, update layers, steps, learning rate | TTT state and gradients |
| Training | batch, sequence length, optimizer, dtype | cost and recovery |

C_k and F_k use the same architecture. C_k disables test-time updates; F_k enables them. Changing width or vocabulary would create a different model, not a clean control.

## Shape relationship

~~~text
embed_tokens: [vocab_size, hidden_size]
hidden state: [batch, sequence, hidden_size]
lm_head:      [hidden_size, vocab_size]  # untied
logits:       [batch, sequence, vocab_size]
~~~

With tied embeddings, lm_head uses the transpose of the embedding matrix. The last logits dimension is still vocab_size. The MLP may change its intermediate width, but it must return to hidden_size before the residual and vocabulary projection.

## Code and validation

ttt/models/ttt_e2e/configuration_ttt_e2e.py defines the model, MLP-prime, MoE, loop, and dtype fields. ttt/models/base_config.py serializes configurations. ttt/models/auto.py and ttt/models/registry.py provide model construction without training-script branches.

~~~python
from ttt.models import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("configs/runs/tiny_gpu.json")
model = AutoModelForCausalLM.from_config(config, mode="train")
~~~

Configuration loading should immediately validate divisibility, positive dimensions, head counts, vocabulary ranges, and fast-weight layer positions. The expanded configuration is written into experiment.json so defaults cannot silently drift.

## Transition

The next chapter follows one token through attention, the static FFN, MLP-prime, and the fast-weight state.
