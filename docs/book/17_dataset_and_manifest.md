# 10　Dataset、分片与 Manifest：把数据变成可恢复输入

## 为什么一个目录不够

如果训练脚本只是读取目录中的所有 `.npy`，它依赖了文件系统的排序、隐藏文件和人工约定。换一台机器、增加一个分片或清理临时文件，都可能改变 token 流。

Manifest 把这些隐含条件写成数据版本。当前 loader 期待的核心字段是：

~~~json
{
  "format_version": 1,
  "vocab_size": 259,
  "shards": [
    {
      "tokens": "v-00000.npy",
      "mask": "v-00000.mask.npy",
      "count": 256,
      "sha256": "<token-file-sha256>",
      "mask_sha256": "<mask-file-sha256>"
    }
  ]
}
~~~

`tokens` 和 `mask` 是相对于 manifest 所在目录的文件名，loader 还要求分片不能逃出该目录。这个限制看似保守，却避免 manifest 通过绝对路径偷偷读取另一台机器的文件。

## TokenDataset 做了哪些安全检查

`TokenDataset.__init__` 会检查 manifest 版本和 vocab_size；逐个验证 token 与 mask 文件的 SHA256；使用 memory mapping 读取一维非空整数数组；检查 token 是否位于 `[0, vocab_size)`；检查 mask shape、有限性和 `[0,1]` 范围；最后累积每个 shard 的结束 offset。

这些检查有不同作用。Hash 证明内容没有被替换，范围检查证明输入不会让 embedding 越界，shape 检查证明 token 与 mask 能配对。一个通过 hash 的文件仍可能是错误词表下生成的合法整数，因此 manifest 的 vocab_size 仍然必要。

当前 `TokenDataset` 还会计算分片端点，并让任意全局 start 经过 `searchsorted` 找到物理 shard。窗口跨 shard 时，代码按各分片的局部 offset 取值。这是让大数据不必完整载入内存的关键，但跨 shard 的窗口仍然是同一条逻辑 token 流。

## Manifest 身份与文件身份

manifest 有自己的 SHA256，分片也有各自 SHA256。两层都需要保留：

- 分片 hash 用来发现单个文件损坏或替换；
- manifest hash 用来发现分片顺序、计数或配置被改；
- 输入原始文件 hash 用来追溯 token 是从什么内容生成的。

如果只保存分片 hash，却把 manifest 重新排序，单个文件都没有变化，训练序列却会改变。反过来，manifest 一样但文件内容被替换，分片 hash 会阻止读取。

当前 prepare_sharded_data 给每次生成的数据分配 UUID 版本，并拒绝覆盖已有 manifest。这能保护旧数据，但也说明“重复运行得到字节完全相同的目录”并非当前目标。若需要内容寻址，应把版本从随机 UUID 改为输入与配置摘要的派生值。

## Dataset 与随机采样器

Dataset 描述“有哪些 token”，采样器描述“下一步取哪里”。`BatchStream` 使用 NumPy Generator 随机取 start；state_dict 保存 generator state、数据 SHA256、sequence length、batch size 和 seed。

恢复时必须同时验证这些字段。仅使用同一个 seed 重新创建 generator，不会跳到中断时的内部状态；仅保存 generator state 而不验证数据摘要，又可能在新数据上继续执行。

`sample_rows(step, indices)` 使用 `SeedSequence` 从 seed、step 和 index 派生独立窗口，适合固定探针和评估样本。它与 `next_batch` 的随机流不是同一个采样协议，报告中要区分。

## 大数据的实际约束

memory mapping 减少了 Python 堆内存，但不会消除存储带宽和随机访问成本。窗口完全随机时，操作系统缓存命中率可能低；分片太小会增加文件管理和 hash 开销，太大又会让单个文件难以迁移。

工程上通常会把逻辑 shard、物理 shard 和训练 batch 分开设计。逻辑上可以按文档簇组织，物理上按顺序写大文件，训练时再按 worker/rank 切分。当前仓库是教学实现，先保证校验与恢复，再谈异步预取、缓存和分布式采样。

下一章讨论多个来源如何混合。只有知道 sampler 读取的是哪个来源，我们才能控制训练配方。
