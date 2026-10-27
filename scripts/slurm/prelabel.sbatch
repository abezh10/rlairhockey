#!/bin/bash
# Run the detector over the labeling frames on CPU (scripts/prelabel.py predict); upload afterwards from the login node.
# Usage: sbatch scripts/slurm/prelabel.sbatch [--model v1]
#SBATCH --job-name=prelabel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=normal
#SBATCH --output=outputs/slurm/%x-%j.out
source env.sh
export PYTHONDONTWRITEBYTECODE=1
python -W ignore scripts/prelabel.py predict "$@"
