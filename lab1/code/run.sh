#!/bin/bash
# Reproduces lab1 end-to-end: cleans raw data, re-executes the analysis
# notebook (regenerating figures), then compiles the PDF report.
set -e

cd "$(dirname "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate 215a

# Re-run the notebook from scratch, regenerating figures in ../figs/
jupyter nbconvert --to notebook --execute --inplace lab1.ipynb

# Compile the report (two passes to resolve references)
cd ../report
pdflatex -interaction=nonstopmode lab1-report.tex
pdflatex -interaction=nonstopmode lab1-report.tex

echo "Done. Report is at report/lab1-report.pdf"
