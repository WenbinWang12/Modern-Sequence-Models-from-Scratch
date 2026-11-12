# 与 teleai 模型包的工程对齐

`F:/teleai/IndexShare/From_YZH/mindformers_formal_v1.0_simlarity_production` 采用的是模型框架组织方式，代表代码要被训练器、推理器、配置系统、权重转换工具和测试系统共同调用。当前 TTT 的算法实现已具备研究验证能力，但原来的入口主要面向直接导入 Python 类，工程接口还不完整。

本轮还对照了 `F:/teleai/MLA_SWA` 的实验目录和 `mindformers` 正式包。这里借鉴的是责任边界和公开接口，不复制 MindSpore、Ascend 或 teleai 的运行时依赖；TTT 的计算后端仍然是 JAX，目标硬件是 NVIDIA GPU。

| teleai 中的工程习惯 | TTT 当前对应物 | 仍需补齐的生产规范 |
|---|---|---|
| `models/<model>/configuration_*.py`、`modeling_*.py` | `ttt/models/ttt_e2e/` | 文件头许可证、完整类型注解、稳定公共 API 和更细的模块文档 |
| `models/auto` 与注册器 | `ttt/models/auto.py`、`registry.py` | 让 train/infer/base 三种入口都通过同一注册协议构建，减少硬编码导入 |
| `Trainer`、`TrainingArguments`、callbacks | `ttt/trainer/`、`ttt/train.py`、`ttt/training.py` | Trainer 门面已加入；日志回调、统一评估和完整生命周期仍需继续接入 |
| YAML 配方与配置覆盖 | `configs/**/*.json`、`ttt/config.py` | 提供 schema 版本、实验覆盖、配置摘要和可读的训练配方 |
| `toolkit/weight_convert` 与 safetensors 文档 | `ttt/convert_official.py`、`ttt/checkpoint.py` | 提供 official config 模板、格式版本、参数映射和公开发布检查 |
| `tools/logger.py`、monitor、callback | `print`、`metrics.jsonl` 和零散脚本 | 统一结构化日志、事件级回调、吞吐/显存/编译阶段指标 |
| 分层 `tests/ut`、`tests/st` 和 CI | `tests/`、GitHub workflows | 修复依赖安装 CI，分离静态检查、CPU 单测、GPU smoke 和多卡验收 |

## 计划中的代码改造顺序

1. 先稳定公共模型接口：统一 `AutoConfig`、`AutoModelForCausalLM`、配置 schema 版本和 train/infer/base 的构建协议。
2. 继续完善已加入的 `ttt/trainer/`：用 `TrainingArguments` 保存运行参数，用 `Trainer` 管理 train/eval/predict、恢复、日志和回调；保留现有 `ttt.train` 作为兼容 CLI。
3. 将配置、数据 manifest、tokenizer 身份、git commit 和命令行写入统一 run metadata；把初始化失败也纳入 status 生命周期。
4. 增加权重转换模板、格式版本和发布前检查，再补分层 CI 与 H100 多卡 smoke 入口。

每一步都保持现有 Python API 的兼容包装，避免为了外观重写已经有数值验证的核心算子。

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
