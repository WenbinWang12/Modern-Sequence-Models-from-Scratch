# TTT-E2E 与长上下文：从状态正确到质量证据

**资料快照：2026-11-11。代码审阅基线：<code>0614bd5</code>。状态：训练与运行时已实现；32K/128K 质量复现未完成。**

## 为什么“能输入长文本”不是结论

假设答案位于一本书的开头，问题出现在结尾。模型即使成功接受整本书，也可能丢掉答案；即使生成正确答案，也可能是训练数据里已经见过它。长上下文实验必须同时回答：历史是否被有效利用，未来信息是否泄漏，资源代价是否可接受。

本节沿用[内外循环](../book/05_inner_outer_loop.md)和[KV cache](../book/06_attention_cache.md)的状态定义。目标是设计一条从 tiny 正确性到目标长度质量评测的证据链，而不是给出未经测量的显卡配置承诺。

## 来源与假设

[End-to-End Test-Time Training for Long Context, v2，2025-12-31](https://arxiv.org/abs/2512.23675v2)研究通过测试上下文上的 next-token 学习更新权重，并在训练时学习适合这一过程的初始化。论文报告的 3B、164B 训练 token 和 128K 场景是作者的实验设置，不是本仓库已复现的结果。

本地待检验假设是：相同模型和 checkpoint，在相同上下文上启用更新后，后续未见 token 的预测或任务得分改善；该改善在记录额外内循环计算后仍有使用价值。这是一种配对评估，不能替代“分别训练两个完整方法”的比较。

## 状态、长度与资源的联系

设输入被切成 chunk \(c\)，静态参数为 \(\theta\)，快权重为 \(\phi_c\)。\(X_c,Y_c,M_c\) 分别是本块的输入、下一 token 目标与损失掩码。先用当前状态预测，再从已经观测到的目标计算更新；未触发梯度裁剪时，SGD 写成：

$$
\mathcal L_c=\mathcal L(X_c,Y_c,M_c;\theta,\phi_c),\qquad
\phi_{c+1}=\phi_c-\eta_{\mathrm{inner}}\nabla_{\phi_c}\mathcal L_c.
$$

实际 <code>sgd_update</code> 还把梯度乘以 \(\min(1,\tau/\sqrt{\|g\|_2^2+10^{-12}})\)，其中 \(g=\nabla_{\phi_c}\mathcal L_c\)，\(\tau\) 是 <code>clip_norm</code>。梯度过大时，更新方向保留而步幅减小。只有后续 chunk 使用 \(\phi_{c+1}\)。如果答案先参与更新、再用于给同一预测评分，就产生标签泄漏。

快权重维度可以固定，但这不意味着整个模型的内存固定：还必须统计 KV、prefix 激活、卷积状态、未完成 chunk 和编译临时缓冲。

一个 full-attention 层的理想 KV 存储约为

$$
M_{\mathrm{KV}}=2L n_{\mathrm{kv}}d_h b_{\mathrm{dtype}},
$$

这里暂取单样本，\(n_{\mathrm{kv}}\) 为 KV 头数，\(b_{\mathrm{dtype}}\) 为每个元素的字节数。令 \(n_{\mathrm{kv}}=12,d_h=64,b_{\mathrm{dtype}}=2\)：32,768 token 对应每层约 96 MiB，131,072 token 对应约 384 MiB。它们只计算 KV，不包含模型权重、梯度、优化器和激活，也不是本仓库峰值显存测量。

滑窗层可限制 KV 历史，但实际分配还取决于实现。当前 <code>configs/runs/tiny_gpu.json</code> 的第一层是 full attention；把后缀换成滑窗不会消除前缀历史增长。因此不能由 tiny 脚本推断论文中的上下文长度无关延迟。

## 当前实现与缺口

| 项目 | 状态 | 代码与范围 |
|---|---|---|
| chunk 预测与梯度更新 | 已实现 | <code>ttt/inner_loop.py</code>；不满块评分但暂不提交更新 |
| 会话 prefill/decode | 已实现 | <code>ttt/runtime.py</code>；保存快权重、缓存和未完成块 |
| 64/512 长度功能检查 | 已实现入口 | <code>ttt/validate.py</code>；随机 tiny 权重，不是质量数据 |
| 给定 checkpoint 的 loss 与检索探针 | 实验框架 | <code>ttt/eval_retrieval.py</code>；自定义探针不能冒充完整 RULER |
| 32K/128K 训练与正式评测协议 | 计划 | 仍需固定配置、数据切分、资源测量和原始报告 |

从零训练本地模型不需要官方权重；只有官方 checkpoint 对齐实验才需要取得相应文件、访问权限、tokenizer 和转换验证。两条路径应分别记录。

## 最小运行、输出与失败诊断

先在准备好的远端 GPU 节点运行功能检查：

~~~bash
mkdir -p outputs/frontier-context
git rev-parse HEAD > outputs/frontier-context/source-commit.txt
.venv/bin/python -m ttt.validate --require-gpu --lengths 64 512 \
  --output outputs/frontier-context/functional.json
~~~

报告应有 <code>passed: true</code>，以及 <code>context_64</code>、<code>context_512</code> 的有限梯度、streaming logits 误差和状态字节数。当前代码在 logits 最大误差大于 \(10^{-3}\) 时失败，因果性另用断言检查。异常报告包含 <code>error_type</code> 和 <code>error</code>。本次书稿编辑没有执行这条命令。

完成训练并有一个真实 checkpoint 后，下面是配对评估模板。两个路径必须替换为已有文件；这里没有暗示它们已在仓库中生成。

~~~bash
CHECKPOINT=/absolute/path/to/trained-checkpoint
VALIDATION=/absolute/path/to/heldout-manifest.json
.venv/bin/python -m ttt.eval_retrieval --checkpoint "$CHECKPOINT" \
  --data "$VALIDATION" --mode loss --sequence-length 64 --samples 8 \
  --output outputs/frontier-context/adapt.json
.venv/bin/python -m ttt.eval_retrieval --checkpoint "$CHECKPOINT" \
  --data "$VALIDATION" --mode loss --sequence-length 64 --samples 8 --no-adapt \
  --output outputs/frontier-context/no-adapt.json
~~~

两个输出的差异只回答该 checkpoint 在该采样协议下启用更新的效果。验证数据不能用训练用的 sample 文件代替；8 个样本只适合入口检查，不足以建立稳定的质量结论。模型位置上限、词表或 manifest 不匹配时先修正协议，不能通过跳过检查获得结果。

## 扩展到 32K 和 128K 的准入条件

| 阶段 | 必须已有的证据 | 常见失败 | 下一步 |
|---|---|---|---|
| tiny，64/512 | 因果、梯度、full/streaming 一致 | 非有限值、状态错位 | 修复正确性 |
| 32K，指定模型 | 位置上限合法、真实长度数据、预热后吞吐、峰值内存 | OOM、编译过长、数据不足 | 缩减模型或改变预算并重记配置 |
| 128K，指定模型 | 32K 通过、长度扩展策略、完整预算、任务与无污染切分 | 内存增长、检索退化、时间超限 | 保留失败记录，不能外推成功 |
| 正式质量比较 | 多种子/置信区间、数据和训练成本匹配 | 单次波动被当收益 | 追加独立运行 |

这里不写“某张 GPU 必定足够”：精度、全注意力层数、batch、编译策略和训练/推理模式都会改变门槛。把 <code>--lengths</code> 改成 131072 仍只是在 tiny 验证器上扩大功能用例，不等于运行 3B 或重现论文。

## 指标、决策与下一次运行

同一表中记录有效 token、held-out loss、检索分数、预热后 prefill/decode 时间、编译时间、峰值显存与状态大小；缺少的指标写“未测”。<code>ttt.profile</code>可提供部分时间和状态统计，但合成 token profile 不测任务质量。

| 研究结论 | 当前证据 | 决策 |
|---|---|---|
| 本地实现是否支持 TTT 训练与流式状态？ | 有代码及验证入口；本节没有新增运行报告 | 允许做 tiny 检查 |
| 32K/128K 是否提高质量或降低成本？ | 尚无本专题固定协议下的结果 | 未决 |
| 是否复现官方 3B 结果？ | 缺完整权重/数据/评测对齐链 | 不作复现声明 |

下一次先保存 64/512 的原始报告，再选择一个已训练小模型做适应开关的配对 held-out 评估。只有这两步的输入、输出和误差都可解释，才扩大长度。
