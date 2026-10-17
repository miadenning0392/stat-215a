#!/bin/bash
# Savio job for Algorithm 1. Submit from lab3/code:  sbatch algorithm1_job.sh
# Extra arguments go to algorithm1.py, e.g. a quick test:
#   sbatch algorithm1_job.sh --B 3 --out-dir ../results_test
#SBATCH --job-name=lab3_algorithm1
#SBATCH --account=ac_guthrielab
#SBATCH --partition=savio2
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=../results/algorithm1_%j.log

set -e

# Run from the code folder, where sbatch was called.
cd "$SLURM_SUBMIT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate 215a

# algorithm1.py reads SLURM_CPUS_PER_TASK to set the number of workers.
python algorithm1.py "$@"
