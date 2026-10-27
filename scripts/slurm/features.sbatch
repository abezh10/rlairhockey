#!/bin/bash
# Per-frame prep signals for each video ID given (runs them in parallel).
# Usage: sbatch scripts/slurm/features.sbatch <id> [<id> ...]
#SBATCH --job-name=features
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --partition=normal
#SBATCH --output=outputs/slurm/%x-%j.out
source env.sh
for id in "$@"; do python scripts/extract_features.py $id & done
wait
