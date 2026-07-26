# Formatting Migration TODOs

- TODO(author-kit): The official AAAI-27 author kit was not present in the workspace. `main.tex` expects `aaai27.sty`; place the official kit files in this directory before compilation.
- TODO(compiler): No LaTeX compiler (`latexmk`, `pdflatex`, `xelatex`, `lualatex`, or `tectonic`) was available in the environment, so no PDF was generated or validated.
- TODO(conclusion): The authoritative Markdown paper ends after Results and contains no conclusion section. `sections/conclusion.tex` deliberately contains no scientific text.
- TODO(figures): Existing figure assets remain in `paper/figures/`. The authoritative Markdown paper contains no figure insertion, caption, or placement instructions, so no figure environment was added.
