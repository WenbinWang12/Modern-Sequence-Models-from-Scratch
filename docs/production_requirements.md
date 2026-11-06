# Artifact 实现对照

来源：Claude artifact `MWGskh5N1XD5uWuvQs1ggj`，本地源文件 v0.16（2026-10-27）。

| 要求 | 代码 | 验证与边界 |
|---|---|---|
| 不缩窄 FFN，后段另加 MLP′ | `models/ttt_e2e` | 参数形状、C/F 配对；E 单独使用缩窄配方 |
| 滑窗、全注意力、GQA、QK-Norm、可选输出门 | `attention.py` | 因果性、流式结果、历史 KV 沿用 |
| 不分配完整 T² 分数与 T×V logits | `kernels.py`、`loss.py` | 参考值、梯度和二阶导对照 |
| SGD 内循环、完整二阶外循环 | `inner_loop.py`、`training.py` | 解析例子、有限差分、共享专家批处理 HVP |
| 时间与层级重算 | 内循环、模型 | 嵌套时间 scan、层 remat；不等于已证明 128K 可用 |
| Gated Delta 基线 | `gated_delta.py` | 原生因果实现，未宣称逐项复现另一项目 kernel |
| 共享专家、top-k、不丢 token | MoE、grouped GEMM | 显式专家混合对照；导数路径仍需优化 |
| 三条路由、四种 Loop | MoE、模型配置 | 梯度与流式一致性；Loop 限于静态前段 |
| 十二组消融、同 FLOPs | `experiments.py` | 十组直接生成，两组须成本输入 |
| 分片数据、SFT 掩码 | `dataset.py`、`data.py` | 跨片采样、SHA256、恢复、分区一致性 |
| 全局参数与优化器分片 | `distributed.py` | 两逻辑设备验证；GPU/多机实测单独留证 |
| 原子提交、直接分片恢复 | checkpoint、lifecycle | 参数、优化器、下一步一致；不恢复未提交目录 |
| 信号保存、运行状态、环境与版本 | train、provenance | 步边界保存；SIGKILL 依赖上一完整检查点 |
| 会话隔离、持久化、容量与接口 | runtime、serving 等 | 边界、随机流、容量拒绝、认证、失败回滚 |
| Scaling N/D 与留出 | `scaling.py` | 多尺寸、多训练量拟合；无伪造结果 |
| 官方 Orbax 导入 | `convert_official.py` | 已知布局数组测试；尚无真实官方 3B 复现 |

## 尚需外部条件和专项工程的验收

| 项目 | 条件或缺口 |
|---|---|
| 官方 3B / 128K / 完整 RULER | requester-pays GCS 访问与计费、对应 tokenizer、正式基准；随机模型不能替代 |
| 1B、多机与专家通信吞吐 | 见实际 GPU 报告；通用 XLA 分片不等于优化的 all-to-all |
| Megatron + SGLang 对照 | 两后端完整高阶梯度与会话实现、同语料同权重对照；普通 LM 路径不能当作 TTT 支持 |
| 32K / 128K 正式训练 | 对应规模、GPU 数量、长序列数据、显存与稳定性记录 |
| 预训练与 scaling 结论 | 真实 N/D 交叉实验和留出规模；不会自动发起大规模训练 |
| 语料生产流程 | 数据许可、清洗、语义去重、混合比例与污染审查依赖具体数据源 |
| RL 与 Agent 原型 | 规划尚未确定奖励、环境、轨迹格式和任务数据；不添加不可运行的占位训练器 |
| 正式 SFT 配方 | 已有监督掩码预处理；提示段是否参与内循环、对话模板及阶段数据比例仍需确定 |
| 高并发部署 | 目前为单进程、独立会话、限并发；连续批处理、跨实例迁移仍需专项实现 |

该表描述交付范围，不将小模型通过写成全部研究规划完成或生产吞吐达标。服务器报告记录 GPU、驱动、依赖、代码摘要和 PBS 作业号，见 `validation.md`。
