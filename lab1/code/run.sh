#!/bin/bash
# Reproduces lab1's results: cleans raw data and re-executes the analysis
# notebook, regenerating all figures in ../figs/.
set -e

cd "$(dirname "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate 215a

jupyter nbconvert --to notebook --execute --inplace lab1.ipynb

echo "Done. Figures regenerated in ../figs/. Compile report/lab1-report.tex to produce the PDF."
