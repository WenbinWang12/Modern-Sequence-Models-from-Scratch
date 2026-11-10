# 清洗章第二轮整改记录

状态：作者整改与静态自查完成，等待远端执行、独立交叉复核和网页视觉验收。本文不是出版通过声明。

基线：开始整改时仓库 HEAD 为 `f544bc5`。共享工作区还有其他章节的并行修改；本轮作者只修改下列清洗相关文件，未提交。

- `docs/book/12_data_cleaning.md`
- `docs/book/en/12_data_cleaning.md`
- `examples/book/cleaning_walkthrough.py`
- `examples/book/fixtures/cleaning_records.jsonl`
- `docs/assets/figures/cleaning-views-zh.svg`
- `docs/assets/figures/cleaning-views.svg`
- 本记录

## 对照原审核意见

| 审核项 | 修改位置 | 本轮处理 | 仍需核验 |
|---|---|---|---|
| T09：NFC/NFKC 只有描述 | 双语 L23–68；脚本 `unicode_examples` | 补 UTF-8 与 Unicode 码点区别、重音组合、全角字母、圈号；列实际码点与字节；NFC 改字节、NFKC 丢兼容区别均说明 | 对照远端报告中所有 Unicode forms；独立审校措辞 |
| T09：全局 strip 破坏技术文本 | 双语 L69–85；`normalize_text`、`inspect_record` | 训练正文不 strip，计数和行比较单独建临时视图；`c02/c13` 验缩进；图中无统计视图回写箭头 | 实测 repr 保留四个空格和末尾换行 |
| T09：没有实际清洗输入和结果 | 双语 L87–140、L180–238 | 原创 13 行 fixture，覆盖正常、空白、短、长、重复、坏 schema、坏 JSON；逐条预期；真实三个输出与 report 字段说明 | 远端执行默认/改阈值两次；边界输入检查 |
| T09：拒绝原因无法对账 | 双语 L142–178；`primary_counts`、`diagnostic_counts`、`stages` | 主状态互斥、诊断允许重叠；手算 13=5+8、10 次诊断；分母 2/9 与 2/13；group 仅包含 10 条 schema 合法记录 | 独立核对计数与 scope |
| T18：章节末尾重复 tokenizer 命令 | 双语完整重写 | 主实践改成真实清洗和阈值比较；删除重复编码命令和第二次结尾 | 交叉审稿检查教学连贯性 |
| T20：图片未视觉验收 | 双语图 6.1，双语原生 SVG | 图解释训练正文/统计视图以及主原因/全部命中的区别；保留可编辑源、可访问 title/desc | 实际页面、窄屏、中文字体和导出效果仍待看图 |
| T19：双语教学内容不一致 | 两版按同一段落结构写作 | 相同 13 条输入、数字、公式、命令、字段范围、失败边界与生产缺口 | 另一作者双语复核；不是按字数判断翻译质量 |

## 新正文的概念依赖与静态审读范围

自查阅读了两版全部新增正文、表格、公式和代码块，以及完整脚本和 fixture。以下定位是写作者的整改说明，不替代独立审读。

| 范围（两版行号对应） | 本节要解决的问题 | 依赖与自查要点 |
|---|---|---|
| L3–21 | 为什么不能把“整齐”当作质量 | 从导航、代码、数学对象讲规则改变学习机会；不以保留率证明质量 |
| L23–44 | 读入失败、HTML 抽取失败、文本过滤失败有什么不同 | 先讲字节/码点/编码，再介绍 JSONL；纯文本脚本不冒充抽取器 |
| L46–68 | NFC/NFKC 的真实作用和有损边界 | 组合重音与兼容字符分开；原文另存；BOM 删除也是策略 |
| L69–85 | 计数忽略空白为何不等于修改正文 | 先给 repr 可见的缩进例，再引入训练/统计视图图 |
| L87–140 | 每条规则到底怎样落在记录上 | schema 是首次解释；metadata label 不冒充检测；重复行公式有分母与空情况 |
| L142–178 | 一条文本命中两条规则怎么算 | 互斥集合推出守恒；全部诊断不能相加当拒绝数；阶段是记账视图 |
| L180–238 | 怎样亲自验证并读懂报告 | 自带输入、独立输出路径、原文对照、字段定义、分组 scope、失败分类 |
| L240–266 | 调阈值之后如何判断 | c05 回来，c06 归因变；按语言/领域检查但实现只分 prose/code；误删率分母需有保留样本 |
| L268–272 | 例子能支持什么结论 | 内存小脚本不声称流式/恢复/抽取/质量分类已实现；自然过渡到去重 |

## 已核对的一手资料

- [Python 3.12 Unicode HOWTO](https://docs.python.org/3.12/howto/unicode.html)：编码、码点和 UTF-8 的区别。
- [Python 3.12 unicodedata](https://docs.python.org/3.12/library/unicodedata.html#unicodedata.normalize)：NFC/NFKC API 与 Unicode 数据库。
- [Python 3.12 str.strip](https://docs.python.org/3.12/library/stdtypes.html#str.strip)：去首尾空白的实际语义。
- [Unicode UAX #15，revision 58](https://www.unicode.org/reports/tr15/tr15-58.html)：规范/兼容等价与兼容变换的区别丢失。

正文引用靠近对应断言；代码、fixture、计数例、图均为本轮原创教学内容。文中 13/5/8 和 min4 的 13/6/7 明确称手算预期，不冒充运行日志。

## 交给远端执行者的命令与预期

CPU 验证仅使用用户指定的 `ssh root@10.210.3.10`，本作者没有运行本地 Python、测试或构建。以下命令在已同步仓库的远端根目录运行；目录已存在时应改用新的证据目录，不删除旧证据。

~~~bash
.venv/bin/python examples/book/cleaning_walkthrough.py --output outputs/book-cleaning-v1
.venv/bin/python examples/book/cleaning_walkthrough.py --output outputs/book-cleaning-min4 --min-prose-chars 4
~~~

默认值预期：

- 输入 13，保留 5，拒绝 8；保留 ID 为 `c01,c02,c03,c12,c13`。
- 主拒绝：`invalid_json=1, invalid_schema=2, invalid_unicode=0, empty=1, too_short=2, too_long=1, repeated_line=1`。
- 诊断：`too_short=3, repeated_line=2`，其余同主拒绝，共 10 次命中。
- stages 剩余：`13,12,10,10,9,7,6,5`；groups 输入数之和 10。
- `c01` 原文 17 码点、NFC 16；`c02` 正文仍含四个行首空格；`c03` 的全角/圈号保持原状。
- `c13` 虽有重复比例 2/3，因 code 不适用重复规则而保留。

min4 预期：保留 6，拒绝 7；`c05` 加入保留，`c06` 从 `too_short` 改为 `repeated_line`；主 `too_short=0,repeated_line=2`。

建议额外验证边界，范围应限于这次新增脚本：

- 正文有字面 `U+2028` 时，JSONL 按 LF 分记录，不误拆成多条。
- 空文件输入数和忽略空行数均为 0；尾部一个 LF 不增加空白行计数。
- 孤立代理码点正文进入 `invalid_unicode`，拒绝日志仍可写出并读回。
- `NaN/Infinity` 输入被记为坏 JSON，不生成非标准 JSON 正常记录。
- 非法 UTF-8 字节使整个运行失败且不创建输出；已有输出目录不覆盖。
- `max_duplicate_line_fraction` 越界、非正长度及最小值超过最大值时明确失败。

已执行的本地检查仅为 `git diff --check`，无空白错误；这不证明 Python 能运行，也不证明 SVG/XML 或网页视觉通过。

## 尚未完成的独立验收

1. 最新脚本（包含 LF 分记录、空文件和非标准 JSON 常量边界修复）的远端运行证据尚待主代理回填或链接。
2. 独立审稿者需核对所有新段落，尤其是“码点/感知字符/token”区别、NFC 的保守边界和误删率抽样范围。
3. SVG 需要实际渲染审阅，检查代码四个空格的呈现、英文长标签、中文字体、手机缩放及 dark mode 页面背景；源文件静态看起来合理不算视觉通过。
4. 页面数学表格和 cases 公式需要站点构建与浏览器检查；这次作者未运行构建。
5. 新读者试读仍缺，不能用作者自查或脚本通过代替。

## 第二轮交叉复核补记

独立复核记录见 [cleaning-round2-recheck.md](cleaning-round2-recheck.md)。复核者发现额外元数据中的 JSON 数字 `1e999` 会经默认浮点解析溢出为无穷，再被默认 JSON 输出写成非标准 `Infinity`。经根代理授权，复核者新增 `parse_finite_float` 拒绝超出有限浮点范围的输入，给全部 JSON 输出增加 `allow_nan=False`，并同步两版正文对输入接受范围的说明。此项限制不把 `1e999` 称为 JSON 语法错误；原始行留在拒绝日志中。另补误删率分母必须非零的数学条件。

以上修改仍待另一人复看和远端回归，不改变默认13条fixture的手算结果。应补正负／嵌套数值溢出、正常有限小数保留，以及原默认/min4的行为检查。复核者没有本地执行Python，也没有声称出版通过。
