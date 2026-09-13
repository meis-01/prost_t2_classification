Run `python report/scripts/analyze_results.py` to regenerate CSV-derived summaries, consolidated metrics, and table rows. The generated LaTeX files are written directly into `report/overleaf/generated/`.

Run `python report/scripts/generate_tikz_data.py` to regenerate the compact data files consumed by the native PGFPlots figures in `report/figures_tikz/`.

Compile from PowerShell with `cd report/overleaf`, then `pdflatex --interaction=nonstopmode --halt-on-error "--output-directory=C:/Users/meisa/Projects/prost_t2_classification/report/build" Main.tex`, `cd ../build`, `bibtex --include-directory=../overleaf Main`, `cd ../overleaf`, and two further identical `pdflatex` commands. Keep the output-directory option enabled so auxiliary files remain under `report/build`.

Unresolved issue: author metadata and funding acknowledgments remain placeholders; IEEEtran and the bibliography toolchain must be available for compilation.
