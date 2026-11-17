四个模型最终都训练到 12,000 步、{{ n(r.training.swa.tokens_seen) }} 个 token，没有出现发散或 loss 尖峰。训练期间发生过用户级 NFS 磁盘配额导致的检查点写入中断；监督脚本从最近检查点自动恢复，并把事件写入 `interruptions.jsonl`。{% if r.interruptions.total %}本次运行共记录 {{ n(r.interruptions.total) }} 次自动恢复。{% endif %}梯度裁剪很少触发：预热之后，外循环梯度范数超过阈值 1.0 的记录步占 {{ pct(r.training.swa.clipped_fraction, 1) }}%（滑窗注意力）和 {{ pct(r.training.ttt.clipped_fraction, 1) }}%（TTT-E2E）。

训练损失不能直接横向比较。全注意力在训练时每个位置可以读到窗口内的全部历史，滑窗注意力只能读到最近 512 个位置，所以前者的训练损失天然更低；而 TTT-E2E 的训练损失是“已经按测试时的方式更新过快权重”之后的损失，和另外两者的含义也不同。图 2 左只用于确认四个运行都稳定收敛。可比的是中、右两栏：同一个评测协议、同一批 held-out 窗口。

{% if r.validation_curves.ttt %}从验证曲线可以看出一个和最终数字同样重要的现象：TTT-E2E 从训练早期就领先于滑窗注意力，而且差距在整个训练过程中没有缩小（图 7 右）。这说明收益不是“训练不足时的暂时优势”。{% endif %}

TTT-E2E 的内循环在训练过程中始终处于活跃状态。快权重范数从初值 {{ f1(r.training.ttt.initial_fast_weight_norm) }} 增长到 {{ f1(r.training.ttt.final_fast_weight_norm) }}，每次内循环更新的范数稳定在 {{ f2(r.training.ttt.final_inner_update_norm) }} 附近，内循环梯度范数在 {{ f2(r.training.ttt.final_inner_grad_norm) }} 左右——它略大于裁剪阈值 \(\tau=1\)，也就是说多数更新步都被裁剪到了固定步长。换句话说，外循环学到的是“在固定步长下，朝哪个方向更新最有用”，而不是“更新多大”。这一点在第 4.8 节的内循环学习率消融中可以看到对应的现象。
