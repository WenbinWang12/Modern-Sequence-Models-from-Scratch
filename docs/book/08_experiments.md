# 8. 消融与 scaling

## 问题

A、D、E、E_plus、S/C/F 和 Loop 消融必须有明确的架构差异、共同的训练协议和实际成本记录，不能只比较配置名称。

## 代码

- `ttt/experiments.py`：生成消融配置和成本清单；
- `ttt/profile.py`：参数、状态和执行成本统计；
- `ttt/scaling.py`：N/D scaling law 和留出检查。

## 实验

```bash
python3 -m ttt.experiments --output configs/ablations --sizes 125m 760m --include-moe
python3 -m ttt.profile --checkpoint CHECKPOINT --context-length 1024 --decode-tokens 16 --training-flops
python3 -m ttt.scaling --input scaling.jsonl --output outputs/scaling.json
```

同 FLOPs 实验需要实际 profiler 或 XLA 成本输入。没有成本输入时，配置会保持 pending。

## 证据

每组实验保存配置、代码 commit、数据清单、训练 tokens、loss、held-out loss、硬件和成本测量方式。

## 下一步

进入 GPU 验证，把小模型正确性和正式规模吞吐分开记录。
