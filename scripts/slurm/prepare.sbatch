#!/bin/bash
# Segment videos and rebuild the review page. Needs data/features/* (features.sbatch).
# Usage: sbatch scripts/slurm/prepare.sbatch [<id> ...]   (default: all in configs/videos.yaml)
#SBATCH --job-name=prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --partition=normal
#SBATCH --output=outputs/slurm/%x-%j.out
source env.sh
export PYTHONDONTWRITEBYTECODE=1
python -W ignore scripts/prepare_video.py "$@" && python -W ignore scripts/build_review.py "$@"
