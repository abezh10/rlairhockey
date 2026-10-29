#!/bin/bash
# Copy clips and calibration frames to Google Drive for notebooks/colab_detector.ipynb, and bring models
# and results made there back. Drive mirrors the repo layout under My Drive/rlairhockey/.
# One-time setup: an rclone remote named "gdrive" (README, "Training on Colab").
# Usage: scripts/sync_drive.sh push [<video> ...]   # data/clips/<video>/ (and changes/), data/calib/frames -> Drive (all by default)
#        scripts/sync_drive.sh pull <model>         # Drive -> here: the model's weights, detections, tracks and renders
#        scripts/sync_drive.sh push-results <model> # here -> Drive: the model's detections, tracks and renders
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE=${RCLONE_REMOTE:-gdrive}:rlairhockey
# --timeout retries a transfer that stalls (one hung on a single clip); stats go to the log at NOTICE level
RC=(rclone --timeout 2m --stats 30s --stats-one-line --stats-log-level NOTICE)
results() { echo "data/detections/$1 data/tracks/$1 outputs/videos/$1 outputs/tracks/$1"; }
case "${1:-}" in
  push)
    shift
    vids=("$@")
    [ ${#vids[@]} -gt 0 ] || vids=($(cd data/clips && ls -d */ | tr -d /))
    for v in "${vids[@]}"; do
      "${RC[@]}" copy --copy-links "data/clips/$v" "$REMOTE/data/clips/$v"
    done
    "${RC[@]}" copy data/calib/frames "$REMOTE/data/calib/frames"
    ;;
  pull)
    m=${2:?usage: sync_drive.sh pull <model>}
    "${RC[@]}" copy "$REMOTE/models/detector/$m" "models/detector/$m"
    # stats.csv is merged per run, so keep ours: Drive's covers only the clips tracked there
    for d in $(results "$m"); do "${RC[@]}" copy --exclude stats.csv "$REMOTE/$d" "$d"; done
    ;;
  push-results)
    m=${2:?usage: sync_drive.sh push-results <model>}
    for d in $(results "$m"); do if [ -d "$d" ]; then "${RC[@]}" copy --copy-links "$d" "$REMOTE/$d"; fi; done
    ;;
  *)
    sed -n 2,8p "$0"
    exit 1
    ;;
esac
