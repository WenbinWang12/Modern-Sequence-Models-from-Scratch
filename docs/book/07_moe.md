# 7. MoE、路由与专家并行

## 问题

MoE 不只是把 FFN 复制多份。路由、共享专家、top-k、不丢 token、专家矩阵乘法和专家并行必须同时成立。

## 代码

- `ttt/models/ttt_e2e/moe.py`：三条适应路线、共享专家和 top-k 路由；
- `ttt/models/ttt_e2e/grouped_gemm.py`：分组矩阵乘法及高阶导参考路径；
- `ttt/distributed.py`：`data/expert` mesh 轴和参数分片。

## 实验

两卡 mesh 配置：

```bash
bash scripts/run_gpu_train.sh \
  --config configs/runs/mesh_tiny.json \
  --data outputs/sample-tokens.npy \
  --output outputs/mesh-tiny
bash scripts/validate_mesh.sh outputs/mesh-validation.json outputs/mesh-checkpoint
```

## 证据

报告会记录专家全局形状、本地 shard 形状、两步训练指标和检查点恢复后的下一步一致性。

## 下一步

在固定硬件和数据后，才可以比较不同路由、Loop 和 MLP′ 位置的真实成本。
