# 7. MoE、路由与专家并行

## MoE 改变的是计算路径

MoE 不是把 FFN 复制几份然后求平均，而是让每个 token 经过一个路由器选择少数专家。这样可以在保持较低激活计算量的同时增加总参数，但代价是路由不均衡、token 重排、专家通信和 checkpoint shard 都变得复杂。

一个 token 的路径可以抽象为：

~~~text
hidden state
    ↓
router logits → top-k expert ids + gate weights
    ↓
group by expert → expert FFN/MLP′ → restore token order
    ↓
weighted sum + residual
~~~

路由、专家计算和还原顺序必须共同定义。只测试专家层的输出形状，无法发现 token 被错误地送到另一个专家或排序恢复错误。

## 本项目的控制变量

为了让 MoE 消融可解释，先固定：专家数、top-k、是否有 shared experts、容量因子、是否允许 token drop、路由损失和通信 mesh。route_mlp、route_delta 和 route_e2e 分别表示不同的路由位置；它们不应同时改变 hidden size 或训练 tokens。

共享专家负责处理所有 token 的稳定变换，稀疏专家负责容量扩展。是否使用 shared expert 是架构变量，不是实现细节；它会影响参数量、激活 FLOPs 和路由负载。

## 代码入口

- ttt/models/ttt_e2e/moe.py 实现路由、shared experts、top-k 和 token dispatch。
- ttt/models/ttt_e2e/grouped_gemm.py 提供分组矩阵乘法与高阶实现路径。
- ttt/distributed.py 定义 data/expert mesh 轴、参数分片和设备间布局。
- configs/ablations/ 保存 125M 和 760M 的对照配置，避免用手工命令拼出不可追踪的实验。

先使用无 token drop 的配置验证正确性，再优化容量和通信。负载均衡指标至少包括每个专家的 token 数、路由熵、最大/平均负载比和 all-to-all 时间。

## 两卡验证

~~~bash
bash scripts/run_gpu_train.sh \
  --config configs/runs/mesh_tiny.json \
  --data outputs/sample-tokens.npy \
  --output outputs/mesh-tiny

bash scripts/validate_mesh.sh \
  outputs/mesh-validation.json \
  outputs/mesh-checkpoint
~~~

验证报告应记录全局专家形状、本地 shard 形状、路由计数、两步训练指标和恢复后的下一步一致性。没有通信时间和显存记录时，不能从 loss 下降推出 MoE 更高效。

## 下一章要比较什么

模型结构增加之后，最容易犯的错误是把参数更多、训练更久或数据顺序不同造成的收益误认为架构收益。下一章建立消融、成本和 scaling 的共同实验协议。
