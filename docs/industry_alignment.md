# 工程接口与生产规范

研究模型要成为可复现的软件，需要同时被配置系统、训练入口、推理入口、检查点工具和测试系统调用。本页记录仓库当前的公共接口、责任边界和仍需补齐的工程工作，不讨论具体外部项目。

## 当前接口

| 工程责任 | 当前实现 | 后续工作 |
|---|---|---|
| 模型配置与构建 | `ttt/models/ttt_e2e/`、`AutoConfig`、`AutoModelForCausalLM` | 稳定 schema 版本、配置摘要和更多模型插件 |
| 训练与恢复 | `ttt/train.py`、`ttt/trainer/`、`TrainingArguments` | 统一评估、预测和完整生命周期 |
| 回调与日志 | `ttt/callbacks.py`、`metrics.jsonl` | 增加吞吐、显存、编译和数据读取监控 |
| 实验配置 | `configs/**/*.json`、`ttt/config.py` | 配置覆盖、继承和运行前 schema 检查 |
| 检查点 | `ttt/checkpoint.py`、`ttt/convert_official.py` | 格式版本、参数映射和发布前完整性检查 |
| 分布式训练 | `ttt/distributed.py`、Orbax | 多机故障处理、拓扑记录和更完整的恢复演练 |
| 测试与 CI | `tests/`、GitHub workflows | 分离静态检查、CPU 单测、GPU smoke 和多卡验收 |

## 代码组织

模型包提供配置、基础模型、注册表和自动构建入口：

```python
from ttt.models import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("configs/125m/ck.json")
model = AutoModelForCausalLM.from_config(config, mode="train")
inference_model = AutoModelForCausalLM.from_config(config, mode="infer")
```

`mode="train"` 和 `mode="infer"` 选择不同的执行入口与梯度检查点策略，但不改变参数结构。已有的直接类导入方式继续兼容。

训练器门面保存配置、数据路径、恢复策略和回调；底层 `ttt.train` 保留命令行兼容入口。运行目录记录 `run_metadata.json`、`environment.json`、`status.json`、`metrics.jsonl` 和检查点，使一次运行可以被审阅和恢复。

## 仍需补齐

- 统一 tokenizer 身份、数据检查报告和初始化失败阶段的记录。
- 增加权重格式版本、跨格式参数映射和发布前检查。
- 完善配置组合、运行覆盖和解析后摘要。
- 将模型前向、训练头、推理头和状态管理拆成更清晰的模块边界。
- 建立多模型插件发现、版本兼容和分层 CI 策略。
- 根据项目实际许可证补齐文件版权、API 文档和发布检查。

这些工作与模型算子数值正确性分开验收：工程接口通过不代表训练质量已经得到证明，训练 loss 下降也不代表恢复、分布式和发布流程完整。
