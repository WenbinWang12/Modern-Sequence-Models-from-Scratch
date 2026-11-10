# 13　训练前验收：在提交长作业之前发现错误

## Preflight 为什么值得单独成章

一个 4 步 tiny 训练通常能在几分钟内发现错误；一个 3 天作业可能在下载、编译和几千步之后才失败。Preflight 不训练模型，它把“输入能否组成一个自洽实验”提前变成明确的 pass、warning 或 fail。

它不是只检查文件存在。它需要把数据、tokenizer、模型配置、采样和输出目录放到同一个关系图里。

## 八项验收

**词表对齐。** manifest 的 vocab_size、tokenizer 的 vocab_size、model config 的 vocab_size 必须相同。特殊 token ID 必须位于词表范围内，且数据中的 BOS/EOS 约定与 tokenizer 一致。

**文件完整。** 每个 token 和 mask shard 存在，SHA256 一致，count 与数组长度一致，数组是一维整数，mask shape 相同。

**样本可训练。** sequence length 至少为 1，窗口能取得 length+1 个 token，有效 mask 不全为零，token 不越界。

**边界清楚。** 文档切分、EOS、packing 和是否跨文档监督已经写入版本记录。若当前实现允许跨 EOS 取窗口，报告不能声称它隔离了文档上下文。

**数据分布。** train/validation 的文档簇不重叠；来源、语言、长度和有效 token 统计可以对账；污染报告没有未解释的 blocked 项。

**采样确定。** seed、混合配方、shuffle、rank 分工、起始位置和 resume 语义有记录。当前仓库单流随机 sampler 的限制也要显示出来。

**模型接口。** sequence length 不超过 max_position_embeddings；ttt_chunk_size、sliding_window 和 batch mask 满足配置约束；fast MLP 和 C/F 变体使用兼容架构。

**输出安全。** 输出目录为空，或明确指定了 resume/initialize-from；checkpoint 的数据指纹与当前输入一致。

## 把验收写成报告

推荐每一项写成结构化记录：

~~~json
{
  "name": "vocab_alignment",
  "status": "pass",
  "inputs": {
    "manifest": "<sha256>",
    "config": "<sha256>"
  },
  "observed": {
    "manifest_vocab_size": 259,
    "config_vocab_size": 259
  },
  "message": "values agree"
}
~~~

`warning` 应该包含人工决定和影响范围；`fail` 应阻止提交长作业。报告本身保存到 run 目录，并被 experiment.json 引用。

当前仓库还没有独立的 `ttt.preflight` 命令。读者可以先使用数据 loader 的校验和 tiny 训练完成最小验收；后续应把上面的检查抽成无 GPU 的命令，方便调度器在申请 GPU 之前运行。

## 一个真实的验收报告

下面是一个 tiny 运行的简化报告。文件和窗口检查通过，跨 EOS 的连续流被标成 warning，而词表不一致直接 fail：

~~~json
{
  "checks": [
    {"name": "file_integrity", "status": "pass", "message": "2 shards and masks match their hashes"},
    {"name": "boundary_semantics", "status": "warning", "message": "windows may cross EOS; document isolation is not claimed"},
    {"name": "vocab_alignment", "status": "fail", "observed": {"manifest": 259, "tokenizer": 259, "model": 32000}, "message": "model vocab_size differs"}
  ],
  "ready_for_long_run": false
}
~~~

这个结果不能靠“先跑起来再说”处理。应先确认模型配置是否应当使用 byte tokenizer 的 259 个位置；如果确实要换成 32K tokenizer，就必须重新生成 token shard、更新 manifest 和 experiment identity，再重新验收。只有把 fail 修复、把 warning 的人工决定写入报告，才可以提交长作业。当前代码没有独立的 preflight 命令，远端可以先运行：

~~~bash
python -c "from ttt.dataset import TokenDataset; TokenDataset('runs/example_shards/manifest.json', 259); print('manifest and shards: pass')"
~~~

这条命令只能证明 loader 的文件和范围检查通过，不能替代上面未实现的 split、污染和分布检查。

## 从本章进入模型

验收通过只说明输入自洽，不说明模型已经学会语言。下一章定义 dense baseline：固定这些输入，测量 loss、梯度、恢复和计算成本，随后才有资格讨论 TTT。
