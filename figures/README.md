# Figures

这里保存书籍图表的 LaTeX/TikZ 源文件。网页正文使用 `docs/assets/figures/` 中的 SVG 版本；源文件、SVG 和正文一起版本控制，确保读者看到的图与仓库中的图一致。

编译示例：

```bash
latexmk -lualatex -shell-escape figures/ttt_training_overview.tex
```

中文正文使用带 `-zh` 后缀的中文标签 SVG，英文正文使用同名英文标签 SVG；两套图共享尺寸、颜色和布局，便于双语页面逐图对照。SVG 采用统一的深蓝、蓝、薄荷绿、金色、玫瑰和紫色系统；新增图应同时提供可访问的 `alt` 文本、编号图注和可追溯的源文件或生成说明。Mermaid 只用于快速结构图，出版级主图优先使用可导出的 SVG/TikZ。
