#!/bin/bash
# Run the detector over all play clips (scripts/detect_clips.py), split across an array of GPU jobs.
# Few CPUs on purpose: GPU nodes often have free GPUs but almost no free CPUs. FarmShare allows 4 GPUs per user.
# Usage: sbatch --array=0-3 scripts/slurm/detect_clips.sbatch [--model v2] [--wait-minutes 90]
#SBATCH --job-name=detect-clips
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x-%A_%a.out
source env.sh
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name --format=csv,noheader
python -W ignore scripts/detect_clips.py --shard "${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}" "$@"
