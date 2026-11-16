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

这些检查回答不同问题。摘要比较用于发现文件内容与已记录版本不一致，范围检查避免 embedding 索引越界，shape 检查保证 token 与 mask 能配对。摘要一致依赖可信的参考清单以及忽略实际不可行的 SHA256 碰撞这一前提；它不证明语料正确、获得授权，或在检查后不会再被改动。如果文件与清单同时被替换，再次计算摘要也不能恢复旧版本的身份。

`vocab_size` 也不是 tokenizer 的身份证明。假设两个大小同为 259 的词表，一个把 ID 68 解读为 A，另一个把它解读为 B。文件中的整数可以全部合法、所有摘要也能一致，但模型收到的含义已变。当前 loader 检查清单与模型的词表大小是否相等，不能识别这种等大小映射变化。实验记录还必须保存 tokenizer 文件摘要、特殊 token ID 和版本；不能把一次范围检查写成 tokenizer 身份验证。

当前 `TokenDataset` 还会计算分片端点，并让任意全局 start 经过 `searchsorted` 找到物理 shard。窗口跨 shard 时，代码按各分片的局部 offset 取值。这是让大数据不必完整载入内存的关键，但跨 shard 的窗口仍然是同一条逻辑 token 流。

## 真实语料中的 `.idx` 与 `.bin`

前面的 manifest 是本仓库教学 loader 使用的格式。公开预训练语料还常见另一种布局：一个 `.bin` 保存连续的 token ID，旁边的 `.idx` 保存如何把这条连续数组切成样本。`.bin` 不是文本文件，而是定长整数的字节流；用 `vim` 打开看到乱码是正常现象，并不表示编码损坏。

以 Megatron 风格的 `MMIDIDX` 文件为例，`.idx` 开头有一个固定魔数 `MMIDIDX\0\0`，随后记录格式版本、元素 dtype、sequence 数量和 document 数量。文件后面依次存放每个 sequence 的长度、每个 sequence 在 `.bin` 中的字节起点，以及 document 到 sequence 的边界索引。若 dtype 是 little-endian `int32`，一个 token 占 4 个字节；因此长度为 4097 的 sequence 在 `.bin` 中占 `4097 × 4 = 16388` 字节。这里的 offset 是字节 offset，不是 token 下标，不能直接拿来当数组元素索引。

这里的字段顺序来自 [Megatron Core 的 IndexedDataset 实现](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/indexed_dataset.py)。我们采用它来解释已有文件，不把外部实现的全部采样、缓存和多模态行为默认为本仓库已经实现。

| `.idx` 字段 | 含义 | 例子 |
|---|---|---:|
| magic | 文件格式标识 | `MMIDIDX` |
| version | 索引格式版本 | `1` |
| dtype | token 存储类型 | `int32` |
| sequence_count | 可采样 sequence 数量 | `49130` |
| document_count | 文档边界数量 | `49131` |
| sizes | 每个 sequence 的 token 数 | `4097` |
| pointers | 每个 sequence 在 `.bin` 中的字节起点 | `0, 16388, ...` |

检查这类数据时，先看索引，不要扫描整个百 GB 文件。仓库提供的只读检查器会读取文件头、前几个 length/offset 和少量 token，并检查 `.bin` 大小是否足以覆盖索引声明的最后一个 sequence：

```bash
python3 scripts/inspect_indexed_dataset.py /path/1
```

需要增加抽样数量时，再显式传入 `--sequences` 和 `--tokens`：

```bash
python3 scripts/inspect_indexed_dataset.py /path/1 --sequences 3 --tokens 32
```

如果已知模型词表大小，可以附加 `--vocab-size 131072`，让工具检查抽样 token 是否落在 `[0,131072)`；这仍然是抽样检查，不会宣称整个 `.bin` 已被扫描。

输出是 JSON，至少应能回答：魔数和版本是否正确、dtype 是什么、sequence/document 数量是多少、抽样的 sequence 是否同长、最后一个索引范围是否落在 `.bin` 内。`sequence_lengths` 和 `sequence_offsets` 只包含抽样记录，抽样数由 `sampled_sequence_count` 标明；最后一条序列单独放在 `last_sequence` 中。检查器只读文件，不修改数据，也不会把整个数据集加载进内存；正式实验仍需把数据目录、tokenizer 摘要、检查报告和上游发布版本一起写入 manifest。解析出整数 ID 还不等于读懂文本，只有使用生成这些 ID 的同一 tokenizer，才能把 token ID 解码回字符串。

这个布局与本仓库的 `TokenDataset` manifest 不是同一个 schema。两者都需要记录长度、顺序、dtype、词表和摘要，但不能把 `.idx` 当成 JSON manifest，也不能把字节 offset 当成 token offset。接入外部 indexed dataset 时，应先生成一份本仓库认可的适配 manifest，或者在训练入口显式声明使用哪一种 loader，避免“文件能打开”被误认为“训练协议相同”。

检查报告还应和训练运行关联，而不是散落在临时目录。启动训练前设置 `TTT_DATA_INSPECTION_REPORT`，训练器会把报告的路径、大小和 SHA256 写入 `run_metadata.json`；环境变量指向不存在的文件时会拒绝启动。这样可以区分“检查过某个文件”和“这次训练实际登记了哪份检查证据”。

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

上例的 `length=5` 是底层读取长度，所以构成 4 个预测位置。如果模型配置的窗口长度是 5，训练代码则需要读取 6 个 token 来构造 next-token target。分片边界不会自动插入 EOS，也不会重置 attention；它只是存储边界。是否隔离文档由 token 化和 packing 规则决定。

把手算推广到任意分片：令第 \(j\) 片有 \(n_j\) 个 token，定义累计端点

\[
e_0=0,\qquad e_j=\sum_{k=1}^{j}n_k.
\]

零起点全局索引 \(p\) 属于满足 \(e_{j-1}\le p<e_j\) 的第 \(j\) 片，片内索引为 \(p-e_{j-1}\)。在代码中，`searchsorted` 返回零起点分片索引 \(j-1\)。例如第二片的 \(e_1=4\)，全局 \(p=6\) 对应局部 \(6-4=2\)，读到 22。这个公式说明为什么只改变物理分片大小，在保持 token 顺序与 mask 完全相同的前提下，不应改变任何合法窗口的结果。

文档边界是另一层信息。当前 manifest 没有文档起止位置，也没有完整的 token 到源文档反向索引。若要定位污染样本或解释某段训练数据来自哪里，需要另存文档 offset 表，并关联源记录 ID；仅靠分片名和 token offset 不足以完成这个追溯。上一章还说明了单 NPY 与分片准备入口的 mask、空文本政策差别；直接换入口并不满足“token 和 mask 完全相同”的前提。

## Manifest 身份与文件身份

manifest 有自己的 SHA256，分片也有各自 SHA256。两层都需要保留：

- 分片 hash 用来发现单个文件损坏或替换；
- manifest hash 用来发现分片顺序、计数或配置被改；
- 输入原始文件 hash 用来追溯 token 是从什么内容生成的。

如果只保存分片 hash，却把 manifest 重新排序，单个文件都没有变化，训练序列却会改变。反过来，保留可信的 manifest 而替换文件内容，会在摘要不匹配时阻止读取。它检查的是加载时读到的文件，不能锁住文件并阻止之后的修改；已发布数据目录需要按不可变版本管理。

当前 `prepare_sharded_data` 给每次生成的数据分配 UUID 版本，并在开始时拒绝覆盖已有 manifest。每个输出目录应由一个写入任务使用；这个存在性检查不是多进程写入锁。随机版本号也说明“重复运行得到字节完全相同的目录”并非当前目标。若需要内容寻址，应把版本从输入与完整处理配置的摘要派生，同时明确格式版本。

### 来源摘要必须对应实际消费的字节

只在处理结束后对源文件计算摘要会留下一个漏洞。假设程序已经读取 `{"text":"A"}` 并写出 A 的 token，此时另一个进程把文件改成 `{"text":"Z"}`。结束时直接计算文件摘要，会把 Z 的身份附到 A 的 token 上。分片自身完全没有损坏，普通分片校验却发现不了这个错误。

现在准备程序在读取前计算源摘要，读取时对实际消费的原始字节继续计算摘要，在发布 manifest 前再计算一次源摘要；三者必须一致。空行和 CRLF 换行也进入消费摘要，因为源文件身份指完整字节，而不只是解析后的 JSON 内容。发现差异就拒绝发布 manifest。失败时已经写出的分片可能保留在目录中，但没有成功清单引用它们，不能把这些文件当作完成的数据版本。

这个检查绑定了分片与实际消费的字节，并检测所比较版本之间的差异。它不证明文件在整个处理期间从未变化：文件可能变动后恢复，也可能在最终检查后再次变化。正式制备仍应使用只读、不可变的输入版本。`tests/test_dataset.py` 中的回归在编码 A 时把源文件改为 Z，要求准备失败且没有 manifest；另一项覆盖空行和 CRLF 的源摘要。这些测试已编写，本轮没有执行。

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

Dataset 提供可校验的逻辑 token 流，shard 是存储片段，manifest 记录分片顺序与身份；采样器另外决定下一步读取哪个窗口。恢复不仅要有 seed，还要有数据 digest、随机生成器状态和窗口协议。下一章讨论这些窗口来自哪些来源，以及 sampler 如何实现目标配方。
