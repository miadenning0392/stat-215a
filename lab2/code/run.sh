#!/bin/bash
#reproduces lab 2's results: cleans the raw data and re-executes the analysis
#notebook, regenerating all figures in ../report/figures/ and the table in ../report/tables/
#requires the data folder at lab2/data and the conda environment in environment.yaml
set -e

cd "$(dirname "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate 215a

jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 eda_walkthrough.ipynb

# compile the report if LaTeX is installed (twice so figure/citation references resolve)
if command -v pdflatex >/dev/null 2>&1; then
    cd ../report
    pdflatex -interaction=nonstopmode lab2.tex
    pdflatex -interaction=nonstopmode lab2.tex
    echo "Done. Figures regenerated and report/lab2.pdf compiled."
else
    echo "Done. Figures regenerated in ../report/figures/. pdflatex not found: compile report/lab2.tex to produce the PDF."
fi
