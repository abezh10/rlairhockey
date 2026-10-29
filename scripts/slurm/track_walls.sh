#!/bin/bash
# Track every clip, fit each framing's walls to its bounces (fit_walls.py) and track again, twice so the
# correction settles; then print what's left. Usage: sbatch ... --wrap "bash scripts/slurm/track_walls.sh colab-v2"
set -e
cd "$(dirname "$0")/../.."
source env.sh
model=${1:-colab-v2}
rm -f data/calib/walls.json
python scripts/track_clips.py --model "$model" --jobs 8 > /dev/null
for pass in 1 2; do
  python scripts/fit_walls.py --model "$model"
  python scripts/track_clips.py --model "$model" --jobs 8 > /dev/null
done
python scripts/fit_walls.py --model "$model" --check
