#!/bin/bash
mkdir -p ../report/figs
for notebook in 01_cleaning.ipynb 02_eda.ipynb 03_dimred.ipynb 04_clustering.ipynb; do
    echo "Running $notebook"
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.kernel_name=python3 \
        --ExecutePreprocessor.timeout=-1 \
        "$notebook"
done
