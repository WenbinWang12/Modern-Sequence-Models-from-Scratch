# 13　训练前验收：先检查输入，再启动长作业

## 为什么文件能打开还不够

假设训练刚刚开始就报“标签超出词表”。此时 GPU、编译器和模型都已经启动，但真正的问题可能只是：数据使用了一个 32K tokenizer，模型配置却只有 259 个输出位置。这个错误不需要训练几千步才能发现。把数据、tokenizer、模型和运行配置放在一起检查，就是训练前验收，也常称为 preflight。

验收的任务是发现输入之间的矛盾。它与评测不同：文件完整、标签合法，不能证明数据有代表性；一个 tiny 模型能计算有限 loss，也不能证明它已经学会语言。我们需要先回答“能否执行声明的实验”，再回答“实验结果怎么样”。

## 沿着一个标签追踪检查条件

一个目标 ID 首先必须对应正确 tokenizer 的词表。即使整数没有越界，用错 tokenizer 仍可能把同一个 ID 解释成不同内容。因此只检查 \(0\le y<V\) 不够，还应固定 tokenizer 文件、归一化规则和特殊 token 语义。

接着，模型必须有能力给这个目标分配概率。当前分片 loader 要求 manifest 的 <code>vocab_size</code> 与传入的模型词表大小完全相等。这是本仓库的接口约束，不是所有语言模型的数学定理：模型输出空间大于 tokenizer 实际使用范围在数学上可以成立，但不能据此绕过这里的版本检查。更不能通过手改 manifest 数字，把不同 tokenizer 的数据伪装成一致。

第三，目标必须来自正确的位置。长度为 \(L\) 的 next-token 预测需要 \(L+1\) 个连续 token；输入取 <code>tokens[:-1]</code>，目标取 <code>tokens[1:]</code>，loss mask 必须与这些目标对应。如果整窗 mask 为零，程序可能仍返回一个数值安全的零损失，但这表示没有监督信号，不表示模型预测完美。

最后，要检查上下文是否符合声明。当前连续流窗口可跨越 EOS，loss mask 不会自动清空 attention 或快权重。若研究要求每篇文档独立，就需要额外实现并检验状态隔离，不能靠验收报告的一句“文档独立”代替。

## 哪些检查已有实现

| 对象 | 现有代码检查什么 | 仍需上游或人工检查什么 |
|---|---|---|
| 分片文件 | 路径、SHA256、非空一维整数、count | 原始来源、许可证、抽取质量 |
| token 与 mask | ID 范围、mask 等长且有限、值在 \([0,1]\) | 特殊 token 语义、mask 是否覆盖正确目标 |
| manifest | 格式版本、词表大小、分片列表 | tokenizer 文件版本、文档反查、切分与污染 |
| 窗口 | 长度与索引边界 | 来源比例、足够多窗口中的有效监督分布 |
| 恢复 | 数据指纹与采样配置的一致性 | 连续运行与恢复运行的实际下一步对照 |

前三行主要对应 <code>ttt/dataset.py</code> 中的 <code>TokenDataset</code>；采样状态对应 <code>ttt/data.py</code> 中的 <code>BatchStream</code>。这些检查并没有组成一个覆盖全表的 <code>ttt.preflight</code> 命令。把尚未实现的分析单独列出来，能避免读者把 loader 成功误解成全部验收通过。

## 从已有样例生成一份小报告

下面的命令使用仓库自带的四条 JSONL 教学记录。在目标 Linux 节点、已完成环境安装的仓库根目录运行。第一次先生成分片；生成器拒绝覆盖已有 manifest，重复检查时直接运行第二段即可，改变数据则使用新目录。

~~~bash
.venv/bin/python -m ttt.data \
  --input examples/sample.jsonl \
  --output outputs/preflight-example \
  --sharded --tokenizer byte --shard-tokens 256
~~~

现在检查 manifest，并取第一个长度为 17 的窗口，构造 16 个预测目标。这里固定起点是为了让初学者方便核对；它不代表随机采样覆盖率。

~~~bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from ttt.dataset import TokenDataset

dataset = TokenDataset("outputs/preflight-example/manifest.json", 259)
if dataset.size < 17:
    raise ValueError("The stream must contain at least 17 tokens.")
tokens, mask = dataset.rows([0], 17)
if tokens.shape != (1, 17) or mask.shape != (1, 16):
    raise ValueError("Unexpected token or target-mask shape.")
valid_weight = float(mask.sum())
if valid_weight <= 0:
    raise ValueError("The inspected window has no supervised target.")
report = {
    "scope": "loader_invariants_and_one_window",
    "manifest_sha256": dataset.fingerprint,
    "stream_tokens": dataset.size,
    "window_tokens": tokens.tolist(),
    "target_mask": mask.tolist(),
    "valid_weight": valid_weight,
    "passed": True,
}
path = Path("outputs/preflight-example/window-check.json")
path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY
~~~

从代码可推知，成功报告的 <code>window_tokens</code> 有一行 17 个整数，<code>target_mask</code> 有一行 16 个权重，<code>valid_weight</code> 大于零。这里不给出假装实测的 hash 或 token 总数；运行后产生的 JSON 才是该数据版本的记录。若前面的检查抛出异常，本段不会写出新的成功报告。重复运行时还应注意旧报告的时间和摘要，不能把旧文件误当成这次成功。

这个例子验证的是 loader 的既有约束和一个窗口。它没有统计所有窗口，也没有检查数据污染、来源混合、tokenizer 文件身份或模型训练。因此报告用 <code>scope</code> 明确它覆盖的范围。

## 故意制造一次不匹配

先完成上面的成功检查，再保持 manifest 不变，只把 loader 接收的模型词表大小改成 32,000：

~~~bash
.venv/bin/python - <<'PY'
from ttt.dataset import TokenDataset

try:
    TokenDataset("outputs/preflight-example/manifest.json", 32000)
except ValueError as error:
    print(type(error).__name__, str(error))
else:
    raise RuntimeError("Expected the vocabulary mismatch to be rejected.")
PY
~~~

当前实现应抛出 <code>ValueError</code>，消息为“数据清单版本或词表大小不匹配。”这条预期来自源码中的同名分支，执行后才能确认本机行为。由于同一份 manifest 刚刚用 259 成功读取，唯一变化是传入的词表大小，读者可以把失败定位到这个接口，而不是怀疑文本或文件下载。这里没有修改原数据，也没有把失败记录写成成功。

## 从错误信息找到对应对象

遇到“数据清单版本或词表大小不匹配”，先并排查看 manifest、模型配置和 tokenizer 的实际词表大小。不要先修改模型层数或学习率。遇到“分片校验失败”，先检查文件是否完整以及是否误用了另一版本的 manifest；重新计算 hash 并填回清单会掩盖替换，不能代替寻找正确文件。

若 token 和 mask 都合法，训练却一直得到零 loss，应统计有效 mask 权重，并把一条短文档的输入、目标与 mask 对照打印。若 loss 有限但验证集异常容易，再回到切分和近重复检查。不同问题对应不同对象，一次范围检查不可能回答所有问题。

## 长作业之前还缺哪几份证据

完整验收还需要独立的来源与清洗记录、按文档簇的切分与污染检查、tokenizer 身份、声明的 packing 语义，以及恢复对照。对每一项，可以记录 pass、warning、fail 或 not_checked，但必须同时保存检查范围和依据；未执行的检查不能默认为 pass。

<code>warning</code> 用于已知且被接受的限制，例如研究明确允许跨文档连续流；它需要写明对结论的影响。<code>fail</code> 用于实验无法按定义执行的矛盾，例如词表不匹配。<code>not_checked</code> 表示没有证据，不能因为程序没报错就消失。

这些记录可放在运行目录旁的独立报告中；当前训练器的 <code>experiment.json</code> 保存实验配置，不会自动收集这些上游报告。保存路径和摘要应由实验组织者明确记录。下一章使用这些输入建立一个普通 dense 语言模型，让我们进一步检查梯度、优化和恢复，而不是立刻把复杂结构加入尚未验证的数据路径。
