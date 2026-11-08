# 9. GPU、多卡与检查点验证

## 三个问题必须分开

GPU 验证至少回答三个问题：代码能否在目标后端运行；分片、通信和 checkpoint 恢复是否正确；目标规模是否能在给定显存和时间预算内完成。一个 tiny 模型跑通只能回答第一个问题，不能替代后两个。

本书采用三级门槛：

1. **功能级**：单 GPU 前向、反向、有限 loss、因果性和流式一致性。
2. **系统级**：多卡 mesh、expert shard、all-to-all、保存与恢复。
3. **规模级**：正式配置的初始化、编译、稳态吞吐、峰值显存和长时间稳定性。

每一级都应有独立报告；失败时保留日志和环境，而不是只写“GPU 不可用”。

## 代码入口

- scripts/validate_gpu.sh 执行单 GPU 架构与流式一致性检查。
- scripts/validate_mesh.sh 检查 mesh、专家 shard 和恢复。
- ttt/validate.py、ttt/validate_mesh.py 输出机器可读 JSON 报告。
- scripts/sutd_*.pbs 是调度器入口，负责申请资源；模型验证逻辑仍在 Python 中。

调度脚本应该传入共享目录、进程数、rank 和 coordinator 地址，不能把这些信息硬编码到模型模块。这样换成其他 GPU 集群时，只需重写资源申请层。

## 验证命令

~~~bash
bash scripts/validate_gpu.sh outputs/gpu-validation.json
bash scripts/validate_mesh.sh \
  outputs/mesh-validation.json \
  outputs/mesh-checkpoint
~~~

正式规模实验还要记录首步编译时间、编译后稳态 tokens/s、峰值显存、通信时间、loss、梯度范数和 checkpoint 大小。首步慢并不等于稳态慢；恢复成功也不等于恢复后的采样状态正确。

## 失败边界

常见失败应按阶段分类：设备发现失败、依赖/驱动不匹配、XLA 编译失败、shape 或 sharding 错误、通信超时、OOM、checkpoint 不兼容。每类失败都应保留命令、commit、配置和环境摘要，方便在另一台机器上重放。

## 进入正式规模

只有三级门槛都通过，才进入 125M、760M、长上下文、RULER 或官方 checkpoint 的实验。前沿章节可以先提出假设，但最终结论仍要回到同一套数据、成本和恢复证据。

下一部分回到数据：模型容量越大，数据来源、清洗、去重、切分、tokenizer、packing 和 manifest 的错误代价越高。数据章节会把这些输入逐一固定下来。
