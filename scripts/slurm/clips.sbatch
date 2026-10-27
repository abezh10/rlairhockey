#!/bin/bash
# Cut play segments out of the raw videos. Needs data/segments/* (prepare.sbatch).
# Usage: sbatch scripts/slurm/clips.sbatch [--force] [<id> ...]   (default: all in configs/videos.yaml)
#SBATCH --job-name=clips
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --output=outputs/slurm/%x-%j.out
source env.sh
export PYTHONDONTWRITEBYTECODE=1
python -W ignore scripts/cut_clips.py --jobs 8 "$@"
