# 4. 前向顺序与快权重

## 问题

TTT-E2E 的关键不是把一个 MLP 换掉，而是在后段层额外加入 MLP′，并让内循环只更新快权重。普通 FFN 保持原宽度，外循环仍然通过内循环求导。

## 代码

完整前向在 `ttt/models/ttt_e2e/modeling_ttt_e2e.py`：

```text
Attention → MLP′（后段层，可更新快权重）→ 原 FFN
```

`ttt/runtime.py` 保留每个会话的快权重、KV/Delta 状态、未完成块和随机数状态。

## 实验

```bash
python3 -m pytest -q tests/test_runtime_consistency.py tests/test_causality.py
```

在 GPU 机器上使用：

```bash
bash scripts/validate_gpu.sh outputs/gpu-validation.json
```

## 证据

验证会比较整段训练 logits 与流式会话 logits，检查修改后半段 token 不会改变此前已经产生的因果输出。

## 下一步

下一章进入内循环：标签什么时候算作已观测，SGD 更新发生在哪个块边界，外循环如何穿过这个更新。
