# 9. GPU、多卡与检查点验证

## 问题

GPU 验证需要回答三个不同问题：代码是否能运行，分片和恢复是否正确，大模型是否能在目标显存中完成训练。一个结果不能替代另外两个。

## 代码

- `scripts/validate_gpu.sh`：单机 GPU 架构和流式一致性；
- `scripts/validate_mesh.sh`：mesh、专家 shard、Orbax 恢复；
- `ttt/validate.py`、`ttt/validate_mesh.py`：生成 JSON 报告；
- `scripts/sutd_*.pbs`：SUTD 专用调度入口。

## 实验

```bash
bash scripts/validate_gpu.sh outputs/gpu-validation.json
bash scripts/validate_mesh.sh outputs/mesh-validation.json outputs/mesh-checkpoint
```

多机必须由调度系统提供共享目录、coordinator 地址、进程数和 rank。不要在作业分配之外直接运行 GPU 验证。

## 证据

报告保存设备、驱动、依赖、代码摘要、编译时间、稳态步时间、损失、梯度、内存和检查点恢复结果。

## 下一步

只有在这些证据稳定后，才进入正式规模、长上下文、RULER 或后训练实验。
