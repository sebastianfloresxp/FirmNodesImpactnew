# Report — The Silicon Backbone

Technical report source files. Main document: `report_main.tex` (consulting report format).

## Compilation
1. Ensure all files are synced via GitHub or uploaded to Overleaf.
2. In Overleaf Settings, set compiler to **pdfLaTeX** and bibliography to **Biber**.
3. Compile `report_main.tex`.

## Structure
- `frontmatter/` — cover page, disclaimer, executive summary
- `chapters/` — Sections 1–6 (all complete and active in `report_main.tex`)
- `appendix/` — Appendices A (extended results) and B (classification codes)
- `tables/` — LaTeX tabular fragments (included via `\input`)
- `figures/` — All figure files (PNG and PDF)
- `references.bib` — Bibliography (217 entries, biblatex/authoryear/biber)
