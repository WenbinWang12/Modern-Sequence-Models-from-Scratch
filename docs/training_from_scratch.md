# From-scratch 训练

这个仓库提供从随机初始化开始构建现代序列模型的代码；TTT-E2E 是前沿主线之一。最小示例用于检查训练链路，不代表语言能力。

## 安装

```bash
# Windows CPU
uv sync --frozen --python 3.13 --extra dev --extra checkpoint --extra serve

# Linux NVIDIA GPU
uv sync --frozen --python 3.12 --extra dev --extra checkpoint --extra serve --extra cuda
```

## 准备数据

普通文本会先转换成一维 token 数组。正式训练应提供自己的 tokenizer、词表大小、特殊 token 和数据许可记录。

```bash
python -m ttt.data \
  --input examples/sample.txt \
  --output outputs/sample-tokens.npy \
  --tokenizer byte
```

JSONL 语料可以生成带 SHA256 清单的分片数据，也可以生成带监督掩码的 SFT 数据：

```bash
python -m ttt.data --input corpus.jsonl --output data/pretrain --sharded --tokenizer byte
python -m ttt.data --input sft.jsonl --output data/sft --supervised --tokenizer byte
```

## 运行最小训练

```bash
python -m ttt.train \
  --config configs/tiny.json \
  --data outputs/sample-tokens.npy \
  --output outputs/tiny
```

`configs/tiny.json` 是流程配置，包含模型、内循环、外循环、数据长度和检查点格式。正式配置默认使用 Orbax 分片检查点；tiny 配置使用 NPZ 方便检查。训练中断后可使用 `--auto-resume`，或者显式传入 `--resume`。

## 检查输出

```bash
python -m ttt.generate \
  --checkpoint outputs/tiny/step_00000004.npz \
  --prompt "The project" \
  --max-new-tokens 16

python -m ttt.profile \
  --checkpoint outputs/tiny/step_00000004.npz \
  --context-length 16 \
  --decode-tokens 4
```

当前训练入口支持稠密、GDN、MoE、三条适应路线和四种 Loop。正式预训练、SFT、RULER、长上下文、RL 或 Agent 训练需要另外确定数据、目标、奖励和评测协议。
