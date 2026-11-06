# 13. Dataset、切分与 manifest

## 问题

训练 loader 必须知道每个 token 分片的内容、长度和校验值，还要能在中断后恢复采样位置。

## 代码

- `ttt/dataset.py`：分片 token 数据、mmap 和监督 mask；
- `ttt/data.py`：tokenizer、批次采样和数据准备；
- `ttt/checkpoint.py`：保存数据指纹和采样状态。

## 检查项

- token 是否全部位于 `[0, vocab_size)`；
- manifest 中的 SHA256 是否匹配；
- mask 是否与 token 长度一致；
- train/validation 是否存在文档泄漏；
- resume 后采样序列是否和连续运行一致。

## 练习

构造两个 token 分片，故意修改其中一个字节，确认 loader 在训练开始前报告 SHA256 错误。
