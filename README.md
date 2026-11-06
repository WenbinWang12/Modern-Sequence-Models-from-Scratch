# TTT-Model-train-from-scratch

TTT-E2E from-scratch model training implementation.

按 Claude artifact v0.16 实现的 JAX 模型与训练、推理工具。代码注释使用中文。

当前验证结果、对应代码摘要和 PBS 作业号见 [验证记录](docs/validation.md)。官方 3B 权重与 128K 复现本轮暂缓。

模型包的工程接口和 teleai 的对照见 [工程对齐说明](docs/industry_alignment.md)。推荐通过 `AutoConfig` 和 `AutoModelForCausalLM` 构建模型；直接导入原有类的方式继续兼容。

从随机初始化开始训练的步骤见 [from-scratch 训练指南](docs/training_from_scratch.md)，贡献代码请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

后段计算顺序为 Attention → MLP′ → 原 FFN。普通 FFN 不缩窄；内循环只更新快权重，外循环通过内循环继续求导。C/F 对照使用同一架构，仅改变更新开关。

## 安装

Linux 使用 Python 3.12，Windows 使用 Python 3.13。依赖锁定在 `uv.lock`，JAX 为 0.4.38，Orbax 为 0.10.3。

Windows 使用 Python 3.13：Python 3.12 的低精度系统时钟会触发 Orbax 0.10.3 的小文件保存统计错误。Linux GPU 环境仍使用 Python 3.12。

```bash
# CPU 开发
uv sync --frozen --python 3.13 --extra dev --extra checkpoint --extra serve
# Linux NVIDIA GPU
uv sync --frozen --python 3.12 --extra dev --extra checkpoint --extra serve --extra cuda
```

分词和 scaling 拟合按需添加 `--extra tokenizer`、`--extra analysis`。Windows 的 Python 为 `.venv\Scripts\python.exe`；Linux 为 `.venv/bin/python`。下面假定已经激活虚拟环境。

## 最小运行

```bash
python -m ttt.data --input examples/sample.txt --output outputs/tokens.npy
python -m ttt.train --config configs/tiny.json --data outputs/tokens.npy --output outputs/tiny
python -m ttt.generate --checkpoint outputs/tiny/step_00000004.npz --prompt "The project" --max-new-tokens 16
python -m ttt.eval_retrieval --checkpoint outputs/tiny/step_00000004.npz --data outputs/tokens.npy --samples 2
python -m ttt.profile --checkpoint outputs/tiny/step_00000004.npz --context-length 16 --decode-tokens 4 --training-flops
```

tiny 的四步训练用于检查流程，不代表语言能力或召回能力。

正式配置默认保存 Orbax 分片检查点；tiny 显式使用 NPZ。中断后可用 `--auto-resume` 或 `--resume <检查点>`，配置、数据指纹和采样种子必须一致。改变训练阶段使用 `--initialize-from <检查点>`，重新建立优化器。

## 数据

支持一维整数 `.npy` 和带 SHA256 校验的分片清单。分片可以跨文件采样，不需要把整个语料载入内存。

```bash
# JSONL，每行包含 text
python -m ttt.data --input corpus.jsonl --output data/pretrain --sharded --tokenizer byte
# JSONL，每行包含 prompt 和 response；只监督 response 与 EOS
python -m ttt.data --input sft.jsonl --output data/sft --supervised --tokenizer byte
python -m ttt.train --config configs/tiny.json --data data/pretrain/manifest.json --output outputs/sharded
```

正式实验应替换 tokenizer，并同步设置词表与特殊 token ID。SFT 输入应预先按目标对话模板组织；代码不会自动套用另一种聊天模板。预处理保留文档 EOS，不提供语料许可审查、语义去重或质量过滤。

当前监督掩码同时作用于内外循环。正式 SFT 阶段是否让提示段参与 NTP 更新，需要在后训练方案中另行确定。

## 架构与消融

配置由 `model`、`inner`、`training`、`tokenizer` 四部分组成。

| 参数 | 含义 |
|---|---|
| `layer_types` | 逐层滑窗、全注意力或 Gated Delta |
| `attention_backend` | `chunked`、`reference`、`cudnn` |
| `use_fast_mlp`、`ttt_suffix_length` | MLP′ 的存在与位置 |
| `inner.enabled` | 内循环更新开关 |
| `num_experts`、`num_experts_per_token` | 专家数量与 top-k |
| `adaptation_target` | `mlp`、`router`、`delta_router` |
| `loop_mode` | 无循环、前段稠密 FFN、中段整块、中段 Attention |
| `parallelism`、`expert_parallel_size` | 本机 pmap 或全局 mesh 分片 |

默认分块注意力与词表交叉熵保留高阶导。`reference` 用于数值对照；`cudnn` 只支持无缓存前向，不能直接用于当前带缓存的 TTT 和服务路径。

```bash
python -m ttt.experiments --output configs/ablations --sizes 125m 760m --include-moe
# 加入实际成本后生成同 FLOPs 对照
python -m ttt.experiments --output configs/ablations --measurements costs.json
```

每个尺寸有 A、D、E、E_plus、S2/C2/F2、S4/C4/F4，以及两组 C_isoflop。没有成本输入时，两组同 FLOPs 实验在清单中标为 pending。成本文件按 `125m/C2` 等键提供 `method`（`profiler` 或 `xla_estimate`）和 `training_flops_per_token`，不会把 3.4 倍耗时当作 FLOPs 比。

生成配置不会自动提交正式训练。125M/760M 是底座配方名称；加入 MLP′、词表或专家后的实际参数量通过 `ttt.profile` 或验证报告统计。

## 更新边界与会话

块内使用更新前的快权重预测。全部 next-token 标签已被观测后才执行 SGD，下一块使用新权重。prefill 尾块保留缓冲。历史 KV 沿用过去的前向值；求更新梯度时的回放不会替换历史 KV。

每个会话独立保存快权重、KV/Delta 状态、未完成块、位置与随机数状态。Orbax 会话快照校验模型参数版本和更新策略。服务预留最大上下文的持久状态容量；这个预算不包含设备编译缓存、梯度和临时工作区。

```bash
python -m ttt.serve --checkpoint outputs/tiny/step_00000004.npz --snapshot-dir outputs/sessions
```

接口包括 `POST /sessions`、`POST /sessions/{id}/generate`、`POST /sessions/{id}/snapshot`、`POST /sessions/{id}/restore`、`DELETE /sessions/{id}`。生成请求使用 `token_ids`、`max_new_tokens`，以及可选温度和 top-k。非本机监听需要环境变量 `TTT_API_KEY`，通过 `Authorization: Bearer ...` 认证。单进程共享参数和编译函数，限制并发并锁定单个会话；失败请求回滚状态。

## 多设备与集群

`training.parallelism="mesh"` 使用 JAX 全局数组，参数与优化器按 `data/expert` 两轴分片。Orbax 直接恢复到目标设备布局。多机由 `JAX_COORDINATOR_ADDRESS`、`JAX_PROCESS_COUNT`、`JAX_PROCESS_ID` 初始化，每节点一个进程，按全局样本编号确定性采样。

开启 `gradient_checkpointing` 时，相邻同构层通过扫描执行，减少大模型逐层展开的编译开销。参数和检查点仍按逐层结构保存。

SUTD 使用 PBS 项目 `20260204`，在安装依赖后的代码目录提交：

```bash
qsub -P 20260204 scripts/sutd_gpu_test.pbs
qsub -P 20260204 scripts/sutd_mesh_test.pbs
qsub -P 20260204 scripts/sutd_multihost_test.pbs
```

首次环境准备见 `scripts/bootstrap_sutd.sh`。多机启动器使用 `pbsdsh`，在作业分配的节点启动。实测结果和作业号见 `docs/validation.md`。

## 官方权重、评测与 scaling

```bash
python -m ttt.convert_official --source /path/to/model_weights --config official-config.json --output outputs/official.orbax
```

转换器支持官方堆叠 blocks/prime_storage 和拆分 prefix/suffix 两种布局，并转换 Linear 矩阵方向。尺寸、归一化、词表和精度须与官方配置一致。官方缩窄 E 与 E_plus 不能直接互换。转换只导入权重，不恢复官方优化器。

尚未取得官方 requester-pays GCS 检查点的访问与计费条件。布局测试使用已知结构数组，不代表官方 3B/128K 复现。

召回评测支持 `{"prompt":"...","answer":"..."}` 或 `answers` 列表，答案只交给评分器。内置探针不是完整 RULER。`python -m ttt.scaling` 接收含 `parameters/tokens/loss/held_out` 的 JSONL，拟合 N/D scaling law，输出 bootstrap 区间和留出结果，不生成训练结果。

## 审计与检查

推荐顺序（模型文件位于 `ttt/models/ttt_e2e/`，其余模块位于 `ttt/`）：

1. `models/ttt_e2e/configuration_ttt_e2e.py`、`layers.py`：配置和基础算子。
2. `kernels.py`、`attention.py`、`gated_delta.py`：序列混合和缓存。
3. `grouped_gemm.py`、`moe.py`、`modeling_ttt_e2e.py`：专家、路由、Loop、整模。
4. `loss.py`、`inner_loop.py`、`training.py`：损失、SGD、完整外循环梯度。
5. `distributed.py`、`checkpoint.py`、`dataset.py`、`train.py`：分片、数据和恢复。
6. `runtime.py`、`runtime_kernels.py`、`session_store.py`、`serving.py`、`serve.py`：会话与接口。
7. 实验、评测、成本、scaling 与验证入口，以及 `tests/`。

```bash
python -m pytest -q
ruff check ttt tests scripts
ruff format --check ttt tests scripts
python -m ttt.validate --output outputs/validation.json --require-gpu
python -m ttt.validate_mesh --output outputs/mesh.json --require-gpu
```

## 性能边界

MoE 推理前向使用 grouped ragged GEMM。为处理 JAX 0.4.38 的批处理与高阶导限制，求导时的前向和导数都使用按专家的矩阵乘法参考路径。全局 mesh 由 XLA 分片，尚无专门优化的专家 all-to-all 调度。分块注意力保留高阶导，没有声称达到融合 FlashAttention 的吞吐。

代码验证、正式规模吞吐和论文结论分别记录。Megatron/SGLang、完整 RULER、长上下文正式训练与 RL/Agent 环境的缺口见 `docs/production_requirements.md`。
