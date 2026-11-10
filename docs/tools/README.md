# 出版图的生成方式

教材的可编辑主图是 `docs/assets/figures/*.svg`。其中三组数据章教学图有中英文版本；BPE 图另附同名 `.tex`。SVG 负责网页展示，TikZ 源便于后续排版调整。

文档发布流程安装 CairoSVG 2.8.2 与 Noto CJK / DejaVu 字体，并在 MkDocs 构建前生成：

- `assets/figures/exports/<名称>.png`：原尺寸两倍的位图。
- `assets/figures/exports/<名称>.pdf`：从 SVG 导出的矢量 PDF。
- `assets/figures/exports/manifest.json`：图源与产物摘要、库版本及实际匹配字体。

PDF 来自 SVG 转换，**不是本轮编译 TikZ 得到的文件**。新 TeX 源是否能够编译仍应单独记录。PNG/PDF 属于出版资产，生成成功也不证明模型代码、数学内容或数据处理实验通过。

生成入口由 `.github/workflows/docs.yml` 调用。需在已准备 Cairo、fontconfig 及所列字体的出版环境中执行：

~~~bash
python -m pip install -r docs/requirements.txt
python docs/tools/export_figures.py --docs-dir docs
mkdocs build --site-dir site
~~~

不要在未运行 `export_figures.py` 的干净克隆中直接执行 `mkdocs build`：`assets/figures/exports/` 被 Git 忽略，缺少生成资产时图页的下载链接不会成立。发布工作流会按上面的顺序自动生成它们。

本项目的模型与数据 CPU 测试仍使用指定远端节点；不要把本目录的文档发布流程用作替代测试环境。

图片的几何位置、颜色和计数应由图源明确给出。避免仅靠颜色区分含义，保持线型、标签和阅读顺序；不用外部字体 URL、foreignObject 或依赖浏览器滤镜的关键效果。查看实际导出的 PNG/PDF 后，再记录裁切、字号和连线的视觉审阅结果。对照 [CairoSVG 支持的转换接口](https://cairosvg.org/documentation/)。

The editable sources are SVGs, with matching TikZ sources for the BPE figure. The publishing workflow exports 2× PNGs and vector PDFs, recording source hashes, output hashes, library versions, and resolved fonts. PDFs are SVG exports, not compiled TikZ output. These are publication assets, not model or data test results. Model and data checks remain assigned to the designated remote CPU host.

Figures embedded in the book use Markdown within HTML to retain MkDocs link handling, accessible captions, keyboard-focusable scrolling, and source/download links. The [Markdown-in-HTML extension](https://python-markdown.github.io/extensions/md_in_html/) is enabled explicitly in `mkdocs.yml`. Generated exports are ignored by Git and rebuilt during publication.

Do not run `mkdocs build` from a clean clone before running `export_figures.py`: the generated `assets/figures/exports/` directory is ignored by Git, so the figure download links are not available until the export step completes. The publication workflow performs these steps in order.
