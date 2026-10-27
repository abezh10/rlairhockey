#!/bin/bash
# Make a 320x180 analysis proxy (same timestamps) for each video ID given.
# Usage: sbatch scripts/slurm/proxy.sbatch <id> [<id> ...]
#SBATCH --job-name=proxy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --partition=normal
#SBATCH --output=outputs/slurm/%x-%j.out
source env.sh
for id in "$@"; do
  ffmpeg -nostdin -loglevel error -y -i data/raw_videos/$id.mp4 -an -vf scale=320:180 \
    -fps_mode passthrough -c:v libx264 -preset veryfast -crf 20 -g 30 data/proxy/$id.mp4 &
done
wait
