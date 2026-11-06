# 6. 注意力、缓存与流式推理

## 问题

训练可以一次处理一个块，生成却要逐 token 运行。KV cache、滑窗边界、历史位置和 TTT 的未完成块必须保持一致。

## 代码

- `ttt/models/ttt_e2e/attention.py`：因果注意力、GQA 和 KV cache；
- `ttt/models/ttt_e2e/gated_delta.py`：递归状态；
- `ttt/runtime.py`：会话状态和 decode；
- `ttt/session_store.py`：会话快照的原子保存与恢复。

## 实验

```bash
python3 -m pytest -q tests/test_causality.py tests/test_runtime_consistency.py tests/test_session_store.py
```

## 证据

会话测试覆盖位置 0、块边界、尾块、随机流、恢复后继续生成、容量限制和 HTTP 回滚。

## 下一步

MoE 会引入专家路由、共享专家和专家并行，进一步扩大设备布局和检查点恢复的边界。
