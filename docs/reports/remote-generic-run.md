# 通用 Linux 入口远端记录

机器：`milu-01-512G`，Python 3.10.12，JAX 0.4.38，Optax 0.2.4，CPU。

远端使用系统 pip 安装了仓库版本 `0.1.0`，然后执行：

```bash
bash scripts/prepare_sample.sh outputs/sample-tokens.npy
bash scripts/run_train.sh \
  --config configs/tiny.json \
  --data outputs/sample-tokens.npy \
  --output outputs/remote-tiny
bash scripts/run_train.sh \
  --config configs/tiny.json \
  --data outputs/sample-tokens.npy \
  --output outputs/remote-tiny \
  --auto-resume
```

结果：示例数据 436 tokens；4 步训练均产生有限损失和梯度；最终 loss 为 `5.553642749786377`；自动恢复命令识别已有 `step_00000004.npz` 并正常结束。

这次记录验证了通用 `prepare_sample.sh`、`run_train.sh`、系统 Python fallback、NPZ 检查点和自动恢复路径。GPU 机器需要先执行 `bootstrap_linux.sh`，再使用 `run_gpu_train.sh`。
