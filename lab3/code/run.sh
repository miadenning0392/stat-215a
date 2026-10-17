#!/bin/bash
# Reproduces lab 3's report figures and tables by re-executing the notebook,
# which writes to ../report/figures/ and ../report/tables/.
# Requires lab3/data/lingBinary.Rdata, lab3/results/S.csv, and the conda
# environment in environment.yaml. It does NOT recreate S.csv: that comes from
# running algorithm1_job.sh (which calls algorithm1.py) on the cluster.
set -e

cd "$(dirname "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate 215a

# Timeout is off because the pure-Python timing cells take a few minutes.
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 lab3.ipynb

# Compile the report if LaTeX is installed (twice so references resolve).
if command -v pdflatex >/dev/null 2>&1 && [ -f ../report/lab3.tex ]; then
    cd ../report
    pdflatex -interaction=nonstopmode lab3.tex
    pdflatex -interaction=nonstopmode lab3.tex
    echo "Done. Figures regenerated and report/lab3.pdf compiled."
else
    echo "Done. Figures regenerated in ../report/figures/. pdflatex or report/lab3.tex not found: compile report/lab3.tex to produce the PDF."
fi
