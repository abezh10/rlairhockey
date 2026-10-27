#!/usr/bin/env bash
# Download YouTube videos listed in a text file (one URL per line, # comments ok).
# Usage: scripts/download_videos.sh links.txt [extra yt-dlp args...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LINKS="${1:?usage: $0 links.txt [yt-dlp args]}"; shift
export PATH="$ROOT/tools/bin:$ROOT/.venv/bin:$PATH"

# Up to 1080p, highest fps within that, h264 preferred for easy decoding.
# The archive file skips anything already downloaded; info.json keeps the metadata.
yt-dlp \
  --batch-file "$LINKS" \
  -S "res:1080,fps,vcodec:h264,acodec:m4a" \
  --merge-output-format mp4 \
  -o "$ROOT/data/raw_videos/%(id)s.%(ext)s" \
  --write-info-json \
  --download-archive "$ROOT/data/raw_videos/archive.txt" \
  --no-playlist --ignore-errors \
  --sleep-requests 1 --min-sleep-interval 2 --max-sleep-interval 6 \
  "$@"
