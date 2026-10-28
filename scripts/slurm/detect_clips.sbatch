#!/bin/bash
# Run the detector over all play clips (scripts/detect_clips.py), split across an array of GPU jobs.
# Usage: sbatch --array=0-3 scripts/slurm/detect_clips.sbatch [--model v2]
#SBATCH --job-name=detect-clips
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x-%A_%a.out
source env.sh
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name --format=csv,noheader
python -W ignore scripts/detect_clips.py --shard "${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}" "$@"
