# 2. 仓库与实验记录

## 问题

一次训练结果只有在别人能知道代码、依赖、数据、设备和命令时才有意义。实验书把这些信息当成训练产物的一部分。

## 代码

- `ttt/provenance.py`：记录 Python、JAX、设备、驱动、PBS 信息和源码摘要；
- `scripts/record_environment.sh`：保存环境 JSON；
- `docs/reproducibility.md`：完整复现命令；
- `ttt/checkpoint.py`：保存参数、优化器和数据采样状态。

## 实验

```bash
git rev-parse HEAD
bash scripts/record_environment.sh --output outputs/run/environment.json
sha256sum data/pretrain/manifest.json
```

恢复训练：

```bash
bash scripts/run_gpu_train.sh \
  --config configs/runs/tiny_gpu.json \
  --data outputs/sample-tokens.npy \
  --output outputs/tiny-gpu \
  --auto-resume
```

`--resume` 要求配置、优化器和数据状态一致；`--initialize-from` 只读取参数并重新建立优化器。

## 证据

仓库中的检查点恢复测试会比较连续训练和断点恢复后的参数、优化器、数据状态及下一步结果。远端 CPU 全套报告记录在 `docs/reports/remote-cpu-tests.xml`。

## 下一步

下一章进入模型本身：为什么词表大小会决定 embedding 和分类头的形状，配置对象如何把这些约束固定下来。
