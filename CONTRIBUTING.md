# 贡献指南

## 开发环境

Linux GPU 环境使用 Python 3.12；Windows CPU 开发使用 Python 3.13。依赖由 `uv.lock` 锁定。

```bash
uv sync --frozen --extra dev --extra checkpoint --extra serve
```

## 提交前检查

```bash
ruff check ttt tests scripts
ruff format --check ttt tests scripts
python -m compileall -q ttt scripts
pytest -q
```

涉及模型结构、内循环、缓存或分片恢复的修改，需要补充对应测试。GPU 和多进程验证脚本位于 `scripts/`，不要在 PBS 分配之外直接运行 GPU 工作负载。

## Pull request

请说明改动的行为、配置变化、验证命令和已知限制。模型质量、长上下文能力和正式吞吐需要真实数据、权重和硬件证据，不能用随机初始化的小模型结果代替。

提交代码时请保持公开数据和权重的许可合规，不要把模型权重、语料、凭据、日志或本地环境目录加入仓库。
