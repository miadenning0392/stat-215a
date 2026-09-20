#!/bin/bash
conda activate 215a

python clean.py
python figures.py
python modeling.py

conda deactivate