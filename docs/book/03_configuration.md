# 3. 配置、词表与维度

## 为什么配置要先于模型代码

一个基模的形状关系一旦确定，很多后续选择就不再是自由的。词表大小决定输入 embedding 的行数；如果使用 tied embeddings，输出分类头共享同一组权重；隐藏维度决定 attention、MLP、归一化和残差的接口；序列长度则会影响位置编码、缓存和显存。

因此配置不是把一堆超参数塞进 JSON，而是用一份可序列化的对象表达模型的结构约束。模型代码只消费配置，不在 forward 中偷偷写入实验选择。

## 四层配置

本项目把配置按职责分成四层：

| 层 | 典型字段 | 影响 |
|---|---|---|
| 模型 | hidden_size、num_layers、num_heads | 参数形状和前向图 |
| 词表 | vocab_size、特殊 token ID、tie_word_embeddings | embedding 与 lm_head |
| 快权重 | inner.enabled、更新层、步数、学习率 | TTT 的状态和梯度路径 |
| 训练 | batch、序列长度、优化器、精度 | 运行成本和状态恢复 |

C_k 和 F_k 必须使用相同的架构配置。C_k 可以关闭测试时更新，F_k 打开内循环，但二者不能通过改变隐藏维度或词表来制造“不公平”的对照。

## 关键形状

~~~text
embed_tokens: [vocab_size, hidden_size]
hidden state: [batch, sequence, hidden_size]
lm_head:      [hidden_size, vocab_size]  # 不共享权重时
logits:       [batch, sequence, vocab_size]
~~~

当 tie_word_embeddings=True 时，lm_head 使用 embedding 的转置，参数量减少，但 logits 的最后一维仍然是 vocab_size。这就是为什么“选定 tokenizer 后，MLP 后的分类头维度”通常已经固定：MLP 输出必须回到 hidden_size，分类头再映射到词表空间。只有更换词表或显式增加投影层，最后一维才会改变。

## 代码入口

- ttt/models/ttt_e2e/configuration_ttt_e2e.py 定义 TTT-E2E 的模型、MLP′、MoE、loop 和 dtype 字段。
- ttt/models/base_config.py 提供 JSON 序列化、保存与加载。
- ttt/models/auto.py 根据 model_type 自动构建配置和模型。
- ttt/models/registry.py 注册新模型，避免在训练脚本中写分支判断。

~~~python
from ttt.models import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("configs/runs/tiny_gpu.json")
model = AutoModelForCausalLM.from_config(config, mode="train")
~~~

配置加载后应立即做整除性、正数性、head 数量、词表范围和快权重层位置检查。越早失败，越不会在数小时编译后才发现形状不合法。

## 如何验证配置没有漂移

tests/test_configuration.py 覆盖默认值、非法维度、C/F 配对、保存加载和自动注册。正式运行时还应把解析后的配置写入 experiment.json，而不是只保存原始命令行参数。这样可以看到默认值展开后的真实实验。

## 下一章要追踪什么

配置只告诉我们张量应该是什么形状，不能说明一个 token 如何穿过模型。下一章沿着单个 token 的前向路径，区分 Attention、原始 FFN、MLP′ 和快权重更新。
