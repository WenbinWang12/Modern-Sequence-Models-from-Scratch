# 11　Dataset、分片与 Manifest：把数据变成可恢复输入

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

## 一个跨分片窗口的手算

把两个分片写成最小例子：

| 物理 shard | token 内容 | count | 全局结束 offset |
|---|---|---:|---:|
| <code>v-00000.npy</code> | <code>[10, 11, 12, 13]</code> | 4 | 4 |
| <code>v-00001.npy</code> | <code>[20, 21, 22, 23, 24]</code> | 5 | 9 |

这里的 offset 是代码数组索引的边界，采用半开区间：第一个分片覆盖全局位置 [0,4)，第二个覆盖 [4,9)。如果请求 <code>start=2, length=5</code>，窗口要读取全局位置 2、3、4、5、6。<code>searchsorted([4,9], 2, side='right')</code> 找到第一个结束端点 4，因此先从第一个分片的局部 offset 2 读取 [12,13]，再从第二个分片的局部 offset 0 读取 [20,21,22]。拼接结果是：

~~~text
global positions: 2, 3 | 4, 5, 6
physical shard:   v-0   | v-1
local offsets:       2, 3 | 0, 1, 2
tokens:            12,13 | 20,21,22
~~~

端点处能看出 <code>side='right'</code> 的必要性：全局位置 4 已经属于第二个分片，其局部 offset 为 \(4-4=0\)。<code>searchsorted([4,9], 4, side='right')</code> 返回分片索引 1；默认的左侧查找会返回 0，把这个位置错分到已经结束的第一片。

如果模型窗口长度是 5，训练代码还需要额外读取一个 token 来构造 next-token target；因此可用窗口长度和底层读取长度不是同一个数字。分片边界不会自动插入 EOS，也不会重置 attention；它只是存储边界。是否隔离文档由 token 化和 packing 规则决定。

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

## 本章最小检查、预期产物与失败诊断

在已经完成环境安装的远端 CPU 节点，从仓库根目录运行下面这一个片段。它使用仓库自带的 JSONL 记录：若 <code>outputs/dataset-sample/manifest.json</code> 不存在，就生成每片最多 256 token 的数据；若已经存在，就先核对输入摘要与 byte 编码模式再读回，绝不覆盖 manifest。若数据版本不同，改用新输出目录，不要修改旧清单来绕过检查。本轮书稿审阅没有执行该命令。

我们读取两个长度为 5 的原始 token 窗口。第一个从首片结束前两个位置开始，跨到第二片；第二个恰好从第二片起点开始。这同时覆盖“跨界”和“恰好落在端点”两种情况。每个窗口随后构造 4 个输入、4 个目标和4个目标 mask；源码中的 <code>rows</code> 已经对 mask 去掉第一列，因此这里不能再给 mask 移位。

~~~bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np
from ttt.data import ByteTokenizer
from ttt.dataset import TokenDataset, file_sha256, prepare_sharded_data

source = Path("examples/sample.jsonl")
output = Path("outputs/dataset-sample")
manifest_path = output / "manifest.json"
if not manifest_path.exists():
    prepare_sharded_data(source, output, ByteTokenizer(), shard_tokens=256)

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if (manifest["input_sha256"] != file_sha256(source)
        or manifest["tokenizer"] != "byte"
        or manifest["supervised"] is not False):
    raise ValueError("Existing data differs from this recipe; use a new output directory.")

dataset = TokenDataset(manifest_path, ByteTokenizer.vocab_size)
if len(dataset.shards) < 2 or dataset.shards[0].size < 2 or dataset.shards[1].size < 5:
    raise ValueError("This example needs two shards with at least 2 and 5 tokens.")
boundary = int(dataset.ends[0])
starts = [boundary - 2, boundary]
tokens, masks = dataset.rows(starts, 5)
expected = np.stack([
    np.concatenate([dataset.shards[0][-2:], dataset.shards[1][:3]]),
    dataset.shards[1][:5],
])
np.testing.assert_array_equal(tokens, expected)
if tokens.shape != (2, 5) or masks.shape != (2, 4):
    raise ValueError("Unexpected window or target-mask shape.")

report = {
    "scope": "two_windows_at_first_shard_boundary",
    "manifest_sha256": dataset.fingerprint,
    "cumulative_ends": dataset.ends.tolist(),
    "starts": starts,
    "tokens": tokens.tolist(),
    "X": tokens[:, :-1].tolist(),
    "Y": tokens[:, 1:].tolist(),
    "M": masks.tolist(),
    "matches_direct_shard_read": True,
}
path = output / "rows-check.json"
path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY
~~~

首次生成后应得到 manifest、token/mask 分片和各自摘要，检查成功时另写 <code>outputs/dataset-sample/rows-check.json</code>。报告包含真实的累计端点、起点、原始 token、\(X,Y,M\) 和 manifest 摘要。<code>tokens</code> 的形状应为 <code>[2,5]</code>，其余三项为 <code>[2,4]</code>。代码把 loader 结果与直接读取两片数组的结果比较；它验证这两个窗口的 token 顺序，并打印实际 mask 供核对，不代表所有窗口或文档边界语义已经验收。

manifest 为每片保存 <code>count</code>，没有 <code>start</code> 或 <code>end</code> 字段；报告中的 <code>cumulative_ends</code> 是 loader 累积计算的。如果比较失败，检查分片顺序、count 与数组长度，再按上面的手算例子核对 <code>side='right'</code> 和局部 offset。若摘要或词表检查失败，先找回正确版本，不能仅重新填写摘要。成功报告会被下一次成功检查更新；异常不会写出新报告，旧文件的存在不能证明本次运行成功。

## 本章小结与下一问

Dataset 是逻辑 token 流和采样协议，shard 只是存储片段，manifest 是两者之间可校验的契约。恢复不仅要有 seed，还要有数据 digest、采样位置和窗口协议。下一章讨论这些窗口来自哪些来源，以及 sampler 如何实现目标配方。
