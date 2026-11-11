# 第五轮内容缺口审阅与补充路线

审阅依据：当前仓库正文、英文版、README、工程说明、所有 editorial/reviews 记录、当前提交历史，以及本次对长期会话要求的回看。仓库和父目录没有找到可检索的 Claude session export 或独立 claude.artifact 原文；可审计依据是仓库正文、production_requirements.md、编辑记录和历史上下文。

本轮先修正了三类已经确认的事实或结构问题：序章中英文统一使用监督权重分母；英文 examples README 从十个改为十二个标准库检查；Frontier Lab 移到主线之后并标为可选。序章还修正了“只生成数据却称为 checkpoint”的表述。

## 需要补充的核心内容

### P1：数学与深度学习工具箱

当前序章和 notation 能定义符号，但读者随后会很快遇到矩阵乘法、QKV、softmax、链式法则、二阶梯度、top-k 和 Adam。缺少一个从标量到向量、矩阵、张量 shape，再到导数、反向传播、SGD、AdamW 和学习率调度的集中桥接。建议放在 notation 之后、实验契约之前，并为 attention、TTT、MoE 各自链接先修小节。

### P1：标准 Transformer 基线的逐层实现

当前内容从配置和 dense baseline 直接进入 TTT。完全零基础的读者还需要先看到 embedding、位置编码、QKV、因果 mask、残差、Norm、SwiGLU、初始化和输出头如何组成一个普通 block。建议增加一章或扩充 Part III：用一个 token 的 shape walkthrough 对照每个代码文件，再说明 TTT 只改变哪一段状态。

### P1：统一评估与统计协议

held-out loss、retrieval、RULER、长上下文和 Frontier 质量指标目前分散在不同页面，没有统一定义。应增加独立评估章，至少固定 train/validation/test 角色、token-weighted NLL 与 perplexity、分数 mask 分母、生成和检索任务格式、污染检查、多 seed、bootstrap 或置信区间，以及统一结果 JSON schema。没有这一章，消融结果无法形成可比较的证据链。

### P1：端到端数据 capstone

examples/book 明确说明五个 walkthrough 使用不同的 tiny corpus，不能拼成一条完整预训练流水线。应增加一个贯穿项目：同一批原始记录依次经过 provenance、清洗、去重、按组切分、tokenizer、packing/mask、shard/manifest、preflight、dense baseline 和评估；每一步保存输入/输出摘要、映射、规则版本和失败回滚记录。当前 manifest 也缺少完整的 document-offset sidecar，应在本章明确实现或边界。

### P1：Tokenizer 身份与模型输出头的桥接

正式 Hugging Face tokenizer 当前只记录名称，CLI 没有单独的 revision 参数，manifest 也没有完整保存 revision、特殊 token 映射、归一化规则和 tokenizer 文件摘要。应支持本地 tokenizer 包或固定 revision，并将这些身份字段写入 manifest。增加一张 raw text → tokenizer artifact → manifest → vocab size/special IDs → embedding/output head → checkpoint 的对象映射图，避免读者把词表大小和模型隐藏维度混为一谈。

### P1：数据生产与治理

获取章的教学实验很清楚，但生产路线仍缺少 streaming 下载、断点续传、分片与 offset 反查、许可证准入/拒绝、PII 与恶意内容处理、撤回删除、人工质量标注和 dataset card。应增加一个最小治理模板，并明确哪些属于本书不覆盖的法律或运营判断。

### P1：训练失败与运行溯源

train.py 的模型构造、数据流初始化、mesh/state 初始化和 checkpoint 恢复发生在主 try 之前，初始化失败时可能没有 status.json。运行元数据也没有完整记录实际 CLI、输入路径、git commit、配置摘要、manifest 摘要和 tokenizer artifact。应把初始化纳入统一失败协议，并写入阶段、异常类型和可追溯输入。

### P1/P2：多卡与正式规模教学

GPU 章节有 mesh 和命令入口，但初学者还看不到 data/model/expert parallel 的 shape、collective 通信、sharding 归属和恢复边界。官方 3B、32K/128K、完整 RULER、专家通信优化仍是外部条件缺口，应在章节中用准入表区分“代码路径存在”“tiny 功能检查”“正式质量证据”。

### P2：双语和工程附录

中英文若干章节标题骨架不同，读者难以逐段对照；应统一主要标题或增加合并映射表。English Edition 目前没有从导航进入 reproducibility、validation、industry alignment、production requirements 等工程附录；应翻译这些入口，或明确标注它们只有中文版本。官方 checkpoint 转换命令还缺少可复制的 official-config 示例模板。

### P2：维护与证据边界

validation.md 主要是历史版本证据，应在页首更醒目标注历史快照，并增加当前提交的待验收表。通用 CI 目前持续在依赖安装阶段失败，不能把旧的 70 passed 当成当前版本通过。Frontier 页面应维护“论文主张—本地实现差异—最小可证伪实验—证据链接/更新时间”四列，并定期核对论文版本。

## 推荐实施顺序

1. 先写数学工具箱和标准 Transformer baseline，补齐零基础读者从普通 LM 到 TTT 的概念阶梯。
2. 接着写统一评估章，并规定结果 schema；之后再扩充消融和 Frontier 质量比较。
3. 实现端到端数据 capstone 和 tokenizer provenance，建立从原始记录到 checkpoint 的一条可追溯链。
4. 最后补治理模板、多卡通信图、英文工程附录、官方转换模板和 CI/启动失败协议。

本轮没有运行 Python、pytest、构建、TeX 或远端命令；以上判断来自静态代码、文档和历史记录审阅。
