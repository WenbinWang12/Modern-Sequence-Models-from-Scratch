# 2. 仓库与可复现实验

## 从一次运行到一个实验对象

一条训练命令只能说明程序曾经启动过，不能说明结果为什么这样。可复现实验需要把运行看成一个有身份的对象：它由代码版本、配置、数据版本、环境和状态共同定义。

~~~text
experiment id
├── source: git commit
├── config: model / optimizer / data
├── input: manifest and SHA256
├── environment: Python / JAX / device
├── state: parameters / optimizer / sampler
└── evidence: metrics / reports / status
~~~

缺少其中任何一项，复现就会变成“尽量重新跑一次”。尤其要区分两种看似相同的动作：从 checkpoint 恢复训练，和只加载参数、重新开始一个新实验。

## 恢复语义

--resume 恢复的是一个训练状态，包括模型参数、优化器动量、已经消耗的数据位置和随机数状态。它适用于作业中断后继续训练；配置、数据和 batch 语义必须兼容。

--initialize-from 只读取参数，随后以新的优化器和新的实验身份开始。它适用于从 baseline 初始化 TTT 或进行迁移实验。把这两种语义混用，会让曲线中的 step、学习率和有效 token 数失去意义。

## 代码入口

- ttt/provenance.py 收集 Python、JAX、设备、驱动、PBS 信息和源码摘要。
- scripts/record_environment.sh 把环境快照写成 JSON，供训练目录和报告引用。
- ttt/checkpoint.py 保存参数、优化器和数据采样状态，并在恢复时检查身份。
- docs/reproducibility.md 给出从安装到恢复的完整命令。

训练目录采用“机器可读为主、人工可读为辅”的方式：metrics.jsonl 适合程序聚合，status.json 说明成功或失败，experiment.json 保存输入，Markdown 报告再引用这些文件，而不是手工抄写数字。

## 一个可审计的运行

~~~bash
git rev-parse HEAD
bash scripts/record_environment.sh --output outputs/run/environment.json
sha256sum data/pretrain/manifest.json

bash scripts/run_gpu_train.sh \
  --config configs/runs/tiny_gpu.json \
  --data outputs/sample-tokens.npy \
  --output outputs/tiny-gpu \
  --auto-resume
~~~

如果命令因为资源不足、编译失败或数据损坏退出，也应保留输出目录和错误状态。失败记录同样是证据：它能告诉我们问题发生在数据读取、参数初始化、编译、通信还是恢复阶段。

## 结果如何比较

两个运行只有在以下字段一致或被明确声明为变量时才可直接比较：训练 tokens、tokenizer 和词表、序列长度、有效 batch、优化器与学习率、随机种子、模型结构和数据 manifest。GPU 型号不同并不会自动改变 loss 的含义，但会改变编译时间、吞吐和显存，因此性能报告必须同时保留硬件信息。

远端完整 CPU 测试结果记录在 docs/reports/remote-cpu-tests.xml；通用训练入口记录在 docs/reports/remote-generic-run.md。它们分别证明代码行为和启动路径，而不是替代正式规模实验。

## 下一章要解决什么

实验身份确定之后，模型配置才有可比的含义。下一章从词表开始，解释 vocab_size 如何约束 embedding 和分类头，以及模型配置怎样被保存、校验和自动构建。
