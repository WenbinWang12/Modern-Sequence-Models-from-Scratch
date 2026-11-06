# 验证记录

测试日期：2026-10-29 至 2026-10-30。SUTD PBS 项目：`20260204`。

## 当前代码

`ttt/` 下全部 Python 文件、`pyproject.toml` 与 `uv.lock` 的联合 SHA256：

`866a789173346e02cc926bd3e5fab543112372d296f473e1ed44189e2c710891`

服务器目录 `core5` 保存的是上一轮 GPU 验证发布；本轮模型框架接口改造已在本地完成，尚未同步到服务器。通过 `ttt.provenance.source_digest()` 可重新计算本地摘要；测试、文档和 PBS 脚本不在该摘要内。普通 wheel 安装不包含项目根文件，摘要仅覆盖包内 Python 代码；依赖版本另外记录在环境报告中。

| 检查 | 当前结果 | 证据 |
|---|---|---|
| 本地 CPU 全套 | Windows/Python 3.13.18，70 passed，无失败或跳过，1284.93 秒 | `results/local-tests-industry.xml` |
| 远端 CPU 全套 | `milu-01-512G`、Python 3.10.12，70 passed、0 failures、0 errors、0 skipped，369.057 秒 | `docs/reports/remote-cpu-tests.xml` |
| 分层扫描对照 | 上述全套包含 8 种架构的前向、完整外循环梯度和流式状态对照 | `tests/test_layer_scan.py` |
| 融合 JIT 对照 | 8 passed，322.23 秒；完整外循环梯度与流式解码均启用 JIT | `results/layer-scan-final.xml` |
| 训练入口与精确续训 | 四层 MoE 启用分层扫描；NPZ 与双逻辑设备 mesh/Orbax 均完成四步，从第二步恢复后参数、优化器及数据状态逐项相同 | `results/train-resume-current.json` |
| Ruff、格式与 Python 编译 | 全部通过 | `ruff check`、`ruff format --check`、`compileall` |
| 服务器源码与依赖 | 上一轮 core5 发布使用 Orbax 0.10.3；本轮接口改造尚未同步 | `results/sutd/core5/release-files.json` |
| 普通 wheel 与源码包 | 上一轮发布版本完成离线构建和独立目录安装；本轮新增框架接口尚未重新构建 wheel | `results/packaging-validation.json` |
| 模型框架接口 | 配置注册、自动构建、训练/推理模式和 NPZ 检查点加载通过 | `results/registry-final.xml` |
| 当前版本单 GPU 回归 | 已提交，尚未计为通过 | PBS `22681.aimc-hn2` |
| 当前版本四 GPU / 14.71 亿参数 | 已提交，尚未计为通过 | PBS `22682.aimc-hn2` |
| 当前版本两 GPU / 两进程 | 与四卡作业顺序执行，尚未计为通过 | PBS `22682.aimc-hn2` |

当前版本将相邻同构层改为扫描执行，参数和检查点格式保持不变。一次仅编译、不分配大模型参数的本地 CPU 诊断中，HLO 从 257,593 行降至 84,525 行，当前版本完成 lowering 用时 21.06 秒、编译用时 69.05 秒。这是编译诊断，不是 GPU 训练吞吐。

分层扫描、融合 JIT 和训练入口续训的原始报告对应摘要 `1bb4e112f0dddc3a429e88ec34a1b7a6210f1f656f76b6c28e7a5aa699af52b9`。之后加入模型框架接口、真实检查点加载和安装目录处理；模型核心计算路径保持原有测试覆盖。

## 历史 GPU 验证

以下结果各自保留实际代码摘要，不能替代当前版本回归。

| 代码 | 检查 | 结果与证据 |
|---|---|---|
| core4 | 单 GPU 全套及架构检查 | 58 passed，474.97 秒；10 组检查通过；PBS `22280.aimc-hn2`，`results/sutd/core4/` |
| core4 | 同节点两 GPU、两进程 | 每进程一张 GPU；完整训练、分片检查点和恢复后下一步逐项相同；PBS `22279.aimc-hn2`，`results/sutd/core4/process-cuda-validation.json` |
| core3 | 四 GPU、14.71 亿参数 | 两步完整元训练通过；PBS `22267.aimc-hn2`，`results/sutd/core3/1b-validation.json` |
| core2 | 单 GPU 与双 GPU | 57 项测试、10 组架构检查、小模型分片及恢复通过；PBS `22225.aimc-hn2`、`22226.aimc-hn2`，`results/sutd/core2/` |
| core1 | 单 GPU | 45 项测试和 10 组架构检查通过；PBS `22202.aimc-hn2`，`results/sutd/core1/` |

core4 摘要为 `20a0c4f85f8a84ba87ec97f5bc040640b61951471f194d1d1be6a292ce9d345f`；core3 为 `28d71ee37140eb0adc4b0dfd99a243129009acb3f2570c5e5d0e7c127e82e712`。

GPU 为 Tesla V100-SXM2-32GB。aimc-gn1 驱动为 580.178.04，aimc-gn3 为 580.95.05。Python 3.12.16、JAX/JAXlib 0.4.38、Optax 0.2.4；core3 的 Orbax 为 0.11.5，core4 及当前版本为 0.10.3。每份 JSON 均含环境、代码摘要与 PBS 作业号。

10 组检查包括稠密模型、GDN、三条 MoE 适应路线、三种 Loop，以及 64/512 token 上下文，验证因果性、梯度有限性、训练与流式结果一致性。core4 最大 logits 误差为 5.96e-8。这些检查不评价模型质量。

历史四卡测试实际参数量为 1,471,034,456，mesh 为 data=2、expert=2。配置为 24 层、隐藏维度 2048、原 FFN 5376、4 个路由专家、top-2、1 个共享专家、末 6 层 MLP′、词表 128256、FP16 计算和 FP32 参数。batch=2、序列长度=4，两步损失为 11.66306 和 4.46780，梯度有限；初始化 184.33 秒，首步含编译 1978.11 秒，第二步 4.49 秒。它使用随机初始化和极短序列，只证明该尺寸可执行完整训练，不代表模型质量、长上下文可行性或正式吞吐。大模型未写出巨型检查点；检查点完整性由小模型验证。

## 已修复问题及保留的失败证据

- 两 GPU 的 14.71 亿参数测试显存不足，见 `results/sutd/core2/1b-validation.json`；随后四卡成功。
- 多进程 UUID 广播发生整数类型提升，已显式恢复为 uint8 后再解码，并添加回归检查。
- Orbax 0.11.5 调用 JAX 0.4.38 不存在的分布式接口，已锁定 Orbax 0.10.3，并在双 GPU 双进程中验证恢复。相同上游问题见 [MaxText issue 1273](https://github.com/AI-Hypercomputer/maxtext/issues/1273)。会话快照使用独立元数据与数组目录，并整体原子提交。
- JAX 0.4.38 的 ragged_dot 对批处理和嵌套扫描求导有限制。推理保留原生 grouped GEMM；求导的前向及导数采用按专家的矩阵乘法参考路径，并检查完整外循环梯度。
- Windows/Python 3.12 的完整回归曾为 67 passed、2 failed：Orbax 0.10.3 将短保存操作的耗时记为零，产生 NaN 吞吐统计，随后 humanize 日志格式化失败；并非训练数值出现 NaN。原始结果保存在 `results/local-tests-windows312.xml`。Windows 验收改用 Python 3.13，其高精度系统时钟变化见 [Python 官方说明](https://docs.python.org/3.13/whatsnew/3.13.html#time)。Linux GPU 环境仍使用 3.12。
- 跨节点 PBS 作业 `22232.aimc-hn2`、`22242.aimc-hn2` 在脚本启动前反复返回 `Exit_status=-30`，已停止重启，不计为多机通过。当前 JAX CPU 后端也未通过所用多进程计算路径。
- 2026-10-30 对空闲 gn4 节点的单卡尝试 `22679.aimc-hn2` 也在脚本执行前反复重启，未产生测试输出，已取消并改投其他节点。

排队作业 `22680.aimc-hn2` 在加入双进程检查后主动取消，由 `22682.aimc-hn2` 取代，没有执行测试。当前排队状态原始记录见 `results/sutd/core5/cluster-jobs.json`。

此前本地两逻辑设备分片验证和训练入口恢复分别见 `results/mesh-local-final.json`、`results/train-resume.json`；它们记录 core4 摘要。恢复检查包括参数、优化器、数据状态及下一步结果。

## 尚未验收

物理跨节点计算、官方 3B 权重、完整 RULER、32K/128K 正式训练、专家通信优化和高并发部署尚未通过验收。官方权重与 128K 阶段本轮暂缓。数据、后训练配方与后续工程条件见 `production_requirements.md`。
