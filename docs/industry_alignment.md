# 与 teleai 模型包的工程对齐

`F:/teleai/IndexShare/From_YZH/mindformers_formal_v1.0_simlarity_production` 采用的是模型框架组织方式，代表代码要被训练器、推理器、配置系统、权重转换工具和测试系统共同调用。当前 TTT 的算法实现已具备研究验证能力，但原来的入口主要面向直接导入 Python 类，工程接口还不完整。

## 已对齐的接口

| 框架习惯 | TTT 当前实现 |
|---|---|
| `configuration_<model>.py` | `models/ttt_e2e/configuration_ttt_e2e.py` |
| 基础模型配置基类 | `models/base_config.py` 的 `PretrainedConfig` |
| 基础模型类 | `models/base_model.py` 的 `PreTrainedModel` |
| 模型注册表 | `models/registry.py` |
| 自动构建入口 | `models/auto.py` 的 `AutoConfig`、`AutoModelForCausalLM` |
| 训练/推理入口分开 | `modeling_ttt_e2e_train.py`、`modeling_ttt_e2e_infer.py` |
| 预训练目录配置 | `config.save_pretrained(path)` 写入 `config.json` |
| 统一导出 | `models/__init__.py` 与 `models/ttt_e2e/__init__.py` |

## 推荐调用方式

```python
from ttt.models import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("configs/125m/ck.json")
model = AutoModelForCausalLM.from_config(config, mode="train")
inference_model = AutoModelForCausalLM.from_config(config, mode="infer")
```

`mode="train"` 和 `mode="infer"` 只选择模型入口和梯度检查点策略，不改变参数结构。现有的 `TTTE2EConfig(...)`、`TTTE2EForCausalLM(...)` 直接导入方式继续兼容。

## 仍需继续对齐的部分

当前项目与完整 MindFormers 工程还有明确差距：

- 训练器仍是 JAX/Optax 的项目内入口，没有接入通用 Trainer 注册系统。
- `from_pretrained` 已支持项目自己的 NPZ/Orbax 检查点；统一的 safetensors 权重格式和跨框架权重协议仍需补充。
- 配置实验仍以 JSON 为主，没有引入 YAML 组合、继承和运行覆盖机制。
- 模型实现仍集中在 `modeling_ttt_e2e.py`，训练与推理入口已经分开，但底层前向代码还没有完全拆成 `Model`、`ForCausalLM`、训练头和推理头。
- 当前注册表只覆盖 TTT-E2E，尚未形成多模型插件发现和版本兼容策略。
- 文件头版权、发布许可证、API 文档生成和分层 CI 还需要按项目实际许可证补齐，不能直接复制 teleai 的版权声明。

后续审计应先确认这些框架边界，再进入注意力、MoE 和内循环的数值细节。这样可以把算法研究代码和生产集成接口分开检查。
