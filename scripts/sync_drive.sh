#!/bin/bash
# Copy clips and calibration frames to Google Drive for notebooks/colab_detector.ipynb, and bring models
# trained there back. Drive mirrors the repo layout under My Drive/rlairhockey/.
# One-time setup: an rclone remote named "gdrive" (README, "Training on Colab").
# Usage: scripts/sync_drive.sh push [<video> ...]   # data/clips/<video>/, data/calib/frames -> Drive (all videos by default)
#        scripts/sync_drive.sh pull <model>         # Drive models/detector/<model>/ -> models/detector/<model>/
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE=${RCLONE_REMOTE:-gdrive}:rlairhockey
# --timeout retries a transfer that stalls (one hung on a single clip); stats go to the log at NOTICE level
RC=(rclone --timeout 2m --stats 30s --stats-one-line --stats-log-level NOTICE)
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
    ;;
  *)
    sed -n 2,7p "$0"
    exit 1
    ;;
esac
