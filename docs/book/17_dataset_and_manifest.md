# 17. Dataset、分片与 manifest

## 为什么要有 manifest

训练目录中不能只放一堆 shard 文件，然后依赖文件名排序。manifest 是数据版本的入口，它说明有哪些分片、每个分片包含多少 token、使用什么词表和 packing 规则，以及如何验证文件没有变化。

一个最小 manifest 可以写成：

~~~json
{
  "format_version": 1,
  "tokenizer": "tokenizer-v1",
  "vocab_size": 259,
  "sequence_length": 128,
  "shards": [
    {
      "path": "shard-00000.npy",
      "tokens": 1000000,
      "sha256": "..."
    }
  ]
}
~~~

路径应相对于数据根目录，避免把某台机器的绝对路径写进实验。生产数据还应记录来源分布、文档簇规则、packing 配置和生成代码 commit。

## 读取与采样

dataset loader 要把逻辑索引映射到物理 shard，并处理边界、短 shard 和损坏文件。随机采样的 seed、epoch、起始位置和 shuffle 算法属于训练状态；恢复时若只加载模型参数而没有恢复采样器，下一批数据就会改变。

在分布式训练中，global batch、local batch、host 数和 shard 轴必须写进运行记录。一个 checkpoint 恢复成功，但每个 rank 读取了不同的数据顺序，仍然不能视为严格恢复。

## 数据完整性

开始训练前可以快速检查 manifest 的格式、词表范围和文件摘要；作业运行中按 shard 记录读入计数和异常。发现 SHA256 不一致时应停止，而不是继续使用“可能只是缓存不同”的文件。

manifest 不是数据许可文件。来源、许可和安全审查仍由前面的数据章节负责；manifest 只描述这一次可供训练读取的派生版本。

## 代码入口与证据

ttt/dataset.py 负责索引、batch 和采样；ttt/data.py 负责 token 文件和 manifest；ttt/checkpoint.py 保存采样器状态。训练目录应同时保存 manifest 的摘要和解析后的关键字段，报告引用它们而不是手工复制。

## 下一章

单一来源的数据便于验证，但真实基模通常要混合网页、书籍、代码和科学文本。下一章讨论 mixture 的比例、采样权重和 token 预算。
