# TTT 与线性注意力：相似从哪里来，等价止于哪里

**资料快照：2026-11-11。代码审阅基线：<code>0614bd5</code>。状态：局部代数示例可执行；独立模型等价性与质量实验未实现。**

## 从“把一句话记住”到一个具体问题

设模型刚读到“门牌号是 42”，后面需要回答门牌号。注意力可以保存先前位置的 key/value；另一种做法是更新一个小模型，让某个 key 映射到相应 value。两者都在利用历史，但这句直觉不足以证明它们相同：读写顺序、损失函数、归一化和状态参数化都会改变运算。

本节先研究一个可手算的线性模型。读完后应能指出哪一步产生了 delta rule，以及为什么这不能直接证明仓库的 TTT-E2E 等价于普通线性注意力。需要的背景是矩阵乘法和平方损失梯度；next-token 交叉熵的区别见[内外循环](../book/05_inner_outer_loop.md)。

## 来源与待检验主张

[Learning to (Learn at Test Time), v4，2025-08-31](https://arxiv.org/abs/2407.04620v4)把可学习模型参数用作序列状态，并研究 TTT-Linear 与 TTT-MLP。[Test-Time Training with KV Binding Is Secretly Linear Attention, v4，2026-05-12](https://arxiv.org/abs/2602.21204v4)分析一类 KV-binding TTT 与 learned linear attention 的关系。

这里检验的是受明确假设限制的更新关系。不能把论文标题扩大为“所有梯度更新都等于标准线性注意力”，更不能跳过从论文模型到本地 next-token MLP′ 的映射。

## 一个线性模型的更新推导

令列向量 \(k_t\in\mathbb R^{d_k}\)、\(v_t\in\mathbb R^{d_v}\) 分别为 key 和 value，快矩阵 \(W_t\in\mathbb R^{d_v\times d_k}\)。为避免与全书词表大小 \(V\) 混淆，这里只用小写 \(v_t\) 表示 value。平方重建损失为

$$
\ell_t(W)=\frac12\|Wk_t-v_t\|_2^2.
$$

把残差记为 \(e_t=Wk_t-v_t\)。矩阵元素 \(W_{ij}\) 只影响残差的第 \(i\) 项，其导数为 \(k_{t,j}\)，因此
\(\partial\ell_t/\partial W_{ij}=e_{t,i}k_{t,j}\)。把元素重新组成矩阵，得到

$$
\nabla_W\ell_t=(Wk_t-v_t)k_t^\mathsf T.
$$

一步学习率为 \(\eta\) 的 SGD 更新是

$$
W_t=W_{t-1}-\eta(W_{t-1}k_t-v_t)k_t^\mathsf T
=W_{t-1}(I-\eta k_tk_t^\mathsf T)+\eta v_tk_t^\mathsf T.
$$

右侧第一项先修正原状态，第二项写入新关联。这是线性快模型在上述损失下的 delta 更新。相比之下，简单的加法线性注意力状态为
\(S_t=S_{t-1}+v_tk_t^\mathsf T\)；它没有预测误差修正项，因此一般不等于 \(W_t\)。是否还包含衰减、归一化分母或学习得到的特征映射，需要逐个定义。

取 \(W_0=0\)、\(k_1=(1,0)^\mathsf T\)、\(v_1=(2,-1)^\mathsf T\)、\(\eta=0.5\)，得到

$$
W_1=\begin{pmatrix}1&0\\-0.5&0\end{pmatrix}.
$$

再使用同一个 key 和 \(v_2=(0,1)^\mathsf T\)，delta 更新得到
\(W_2=\bigl(\begin{smallmatrix}0.5&0\\0.25&0\end{smallmatrix}\bigr)\)。
若只把 \(\eta v_2k_2^\mathsf T\) 加到 \(W_1\)，得到的第一列却是 \((1,0)^\mathsf T\)。第二步才暴露区别；只检查零初始化的第一步会漏掉它。

若查询列向量为 \(q_t\in\mathbb R^{d_k}\)，可用更新后的 \(W_t\) 读出 \(W_tq_t\in\mathbb R^{d_v}\)。这是本例约定的“先写后读”。仓库的 next-token TTT-E2E 则先预测当前 chunk，再使用观测到的标签更新供后续 chunk 使用。代数成立不等于因果协议相同。

## 与当前代码的对应关系

| 能力 | 状态 | 真实入口与限制 |
|---|---|---|
| 带门控 delta 递推 | 已实现 | <code>ttt/models/ttt_e2e/gated_delta.py</code>；还有卷积、key/query 归一化、衰减与输出门控 |
| next-token 快权重更新 | 已实现 | <code>ttt/inner_loop.py</code>；快 MLP′ 的目标是 token 交叉熵，不是上面的 KV 平方重建 |
| 下面的两步线性恒等式 | 教学实验 | 自包含 NumPy 命令，不调用模型训练器 |
| 同状态维度的 TTT-Linear 与线性注意力训练对照 | 计划 | 尚无独立配置、数据协议和结果报告 |

## 最小运行与预期产物

登录远端 CPU 机器后，在已安装 NumPy 的环境运行下面的完整片段。它检查两种写法是否相同，并检查与加法更新是否不同；本次编辑未执行它。

~~~bash
mkdir -p outputs/frontier-linear
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np

w = np.zeros((2, 2))
additive = w.copy()
k = np.array([1.0, 0.0])
eta = 0.5
errors = []
for v in (np.array([2.0, -1.0]), np.array([0.0, 1.0])):
    gradient_form = w - eta * np.outer(w @ k - v, k)
    recurrence_form = w @ (np.eye(2) - eta * np.outer(k, k)) + eta * np.outer(v, k)
    errors.append(float(np.max(np.abs(gradient_form - recurrence_form))))
    np.testing.assert_allclose(gradient_form, recurrence_form, atol=1e-12, rtol=0)
    w = gradient_form
    additive += eta * np.outer(v, k)
np.testing.assert_allclose(w, [[0.5, 0.0], [0.25, 0.0]])
assert not np.allclose(w, additive)
report = {"scope": "two-step-linear-squared-loss",
          "max_errors": errors, "delta_state": w.tolist(),
          "additive_state": additive.tolist(), "passed": True}
Path("outputs/frontier-linear/algebra.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
PY
~~~

成功时两次 <code>max_errors</code> 应接近零，<code>delta_state</code> 与手算一致。若失败，先检查列向量约定、外积方向和读写时间。它不测量语言模型质量，也不验证论文里所有可并行形式。

## 对照、指标与决策

完整实验应先固定 \(d_k,d_v\)、初态、key/value/query、学习率、loss 和读写顺序。之后才能比较特征映射、门控或训练质量；一次同时替换多个组件会失去归因依据。

| 问题 | 判据 | 当前记录 |
|---|---|---|
| 两步 SGD 与展开递推是否相同？ | float64 最大误差不超过 \(10^{-12}\) | 命令已写，未执行 |
| 本地 Gated Delta 是否等于该简化更新？ | 逐项关闭额外变换后比较状态与输出 | 尚未建立该对照 |
| 参数/计算匹配时质量是否相当？ | 固定验证集损失、有效 token、FLOPs、吞吐与显存 | 未运行 |

任何一步若使用不同初态、mask 或更新时序，应判为协议不匹配，而不是方法被证伪。下一次先保存代数报告，再实现显式的 TTT-Linear/递推两条路径；在此之前，本专题只支持条件化推导，不支持一般等价或性能结论。
