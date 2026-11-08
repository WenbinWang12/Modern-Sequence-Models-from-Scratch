# 8. 消融、成本与 scaling

## 先写比较协议，再写结果

A、D、E、E_plus、S/C/F 和 Loop 等名称只有在结构差异被定义后才有意义。一个公平实验需要先确定不变项，再列出唯一变化项；否则结果表会把不同模型、不同数据和不同训练预算混在一起。

推荐把每个实验写成一行：

| 字段 | 含义 |
|---|---|
| variant | 唯一的结构开关组合 |
| tokens | 实际消耗的有效 token 数 |
| params | 总参数与激活参数 |
| flops | 理论或 profiler 测量的训练 FLOPs |
| compile_s | 首次编译时间 |
| steady_tokens_s | 编译后的稳态吞吐 |
| peak_memory | 峰值显存 |
| loss | 训练和 held-out loss |
| status | success、pending 或 failed 及原因 |

参数量相同不代表计算量相同；同样的 FLOPs 也不代表显存、通信和编译成本相同。报告必须同时保存质量与系统成本。

## 三组实验轴

第一组是**结构轴**：是否开启 TTT、MLP′ 放在哪些层、Loop 放在 attention 还是 FFN、是否使用 MoE。第二组是**规模轴**：hidden size、层数、序列长度、训练 tokens 和设备数。第三组是**运行时轴**：prefill/decode、完整前向/流式前向、不同 batch 和 cache 长度。

每次只改变一组轴，另外两组固定。若需要改变多个变量，应拆成多行实验并在 manifest 中写明原因。

## 代码入口

- ttt/experiments.py 生成消融配置和参数/激活成本清单。
- ttt/profile.py 记录参数、状态、编译和执行成本。
- ttt/scaling.py 聚合 N/D scaling 数据并检查缺失输入。
- configs/ablations/manifest.json 是实验矩阵的索引，不要把目录名当成协议。

~~~bash
python3 -m ttt.experiments \
  --output configs/ablations/generated \
  --sizes 125m 760m \
  --include-moe

python3 -m ttt.profile \
  --checkpoint CHECKPOINT \
  --context-length 1024 \
  --decode-tokens 16 \
  --training-flops

python3 -m ttt.scaling \
  --input scaling.jsonl \
  --output outputs/scaling.json
~~~

没有 profiler 或 XLA 成本输入时，scaling 结果应标为 pending，而不是填入估计数字。估计值可以用于规划，但不能作为论文式结论。

## 证据如何支持结论

一个可发布的结果至少能回到：配置文件、代码 commit、数据 manifest、训练 tokens、loss 曲线、硬件、成本测量方式和失败记录。图表只展示聚合结果，原始 JSONL 和报告保存完整证据。

## 下一章要验证什么

消融协议确定了“比较什么”，GPU 验证确定了“结果是否真的在目标硬件上成立”。下一章把功能正确性、分片恢复和目标规模训练拆成三个层级，避免用一次小模型运行替代完整验证。
