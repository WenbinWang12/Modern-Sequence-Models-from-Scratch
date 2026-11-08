# 19. 训练前数据检查

## Preflight 解决什么问题

长时间训练开始后才发现词表不匹配、manifest 指向旧文件或验证集为空，代价远高于提前几分钟检查。preflight 是训练前的“输入验收”，它不训练模型，只验证数据与配置能否组成一个自洽的实验。

## 检查清单

训练前至少确认：

1. manifest、tokenizer 和模型的 vocab_size 相同；
2. 所有 shard 存在且 SHA256 一致；
3. token id 没有越界；
4. sequence length、packing、EOS 和 mask 规则一致；
5. 训练、验证和污染报告都存在；
6. 混合配方的目标 token 数可满足；
7. seed、shuffle、起始位置和 checkpoint 语义已定义；
8. 输出目录为空或明确使用 resume/initialize-from。

检查结果应保存为 JSON，包含每一项的状态、计数、输入摘要和失败原因。状态分为 pass、warning 和 fail；warning 不能被静默吞掉，正式规模实验需要在报告中解释。

## 与模型配置的交叉检查

数据检查不能只在 dataset 模块内部完成。它还要读取模型配置，检查 vocab_size、special token ID、sequence length、dtype 和 batch 约束。这样可以在模型初始化前发现“数据能读、模型也能建，但两者无法相连”的错误。

## 运行方式

小规模 preflight 可以在 CPU 上完成，不需要启动 GPU 编译。正式作业通过后，把 preflight JSON 和 manifest 摘要一起写入 experiment.json。训练中若输入摘要变化，checkpoint 恢复应拒绝继续。

## 下一章

输入验收通过后，模型才有可靠的比较对象。下一章建立 dense Causal LM baseline，并把它作为后续 TTT、Loop 和 MoE 的共同参照。
