#!/bin/bash
# run.sh -- reproduces this submission end-to-end: raw data -> cleaned data ->
# notebook (figures + printed results) -> compiled PDF report.
#
# Assumes:
#   - This script lives in lab1/code/, alongside clean.py and lab1.ipynb
#   - The `stat215a` conda environment has already been created:
#       conda env create -f environment.yaml
#   - lab1/data/ contains the raw CSV (not committed to git; grader adds it)
#   - lab1/figs/ and lab1/report/ exist as siblings of lab1/code/

set -e  # stop immediately if any step fails

# Always operate relative to this script's own location, regardless of
# where it's invoked from
cd "$(dirname "$0")"

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate stat215a

# Re-execute the notebook from scratch. This reruns clean_data() from
# clean.py on the raw CSV, regenerates every figure into ../figs/, and
# refreshes all printed output/tables in the notebook in place.
jupyter nbconvert --to notebook --execute --inplace lab1.ipynb

# Build the final PDF report from the LaTeX source.
cd ../report
pdflatex -interaction=nonstopmode lab1-report.tex
pdflatex -interaction=nonstopmode lab1-report.tex  # second pass to resolve references

# If your report uses a .bib file for the bibliography, uncomment:
# bibtex lab1
# pdflatex -interaction=nonstopmode lab1-report.tex
# pdflatex -interaction=nonstopmode lab1-report.tex

echo "Done. Report is at report/lab1-report.pdf"