# Figures

这里保存书籍图表的 LaTeX/TikZ 源文件。图表的源文件和正文一起版本控制；导出的 PDF/SVG 属于构建产物，不提交到源码仓库。

编译示例：

```bash
latexmk -lualatex -shell-escape figures/ttt_training_overview.tex
```

正文使用中文说明，图内使用英文短标签，避免不同机器上的 CJK 字体差异。后续网页构建可以把 PDF 转为 SVG，再由 MkDocs 页面引用；没有图形构建环境时，正文仍然保留 Mermaid 预览和 LaTeX 源链接。
