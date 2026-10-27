#!/bin/bash
# Fine-tune RF-DETR on the Roboflow labels (data/labels/detector/<name>, from scripts/roboflow_pull.py).
# Usage: sbatch scripts/slurm/train_detector.sbatch [train_detector.py args, e.g. --name v1 --epochs 80]
#SBATCH --job-name=train-detector
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x-%j.out
source env.sh
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -W ignore scripts/train_detector.py "$@"
