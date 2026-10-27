"""Seek-based contact sheet for eyeballing a video.

Usage: python scripts/contact_sheet.py <id> <t0> <t1> <step> <out.jpg> [proxy|raw] [cols] [tile_width]
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from rlairhockey.video import frame_at, probe

ROOT = str(Path(__file__).resolve().parents[1] / "data")
vid, t0, t1, step, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
src = sys.argv[6] if len(sys.argv) > 6 else "proxy"
cols = int(sys.argv[7]) if len(sys.argv) > 7 else 8
w = int(sys.argv[8]) if len(sys.argv) > 8 else 320
path = f"{ROOT}/{'proxy' if src == 'proxy' else 'raw_videos'}/{vid}.mp4"
dur = probe(path).duration
tiles = []
for t in np.arange(t0, min(t1, dur - 0.1) + 1e-6, step):
    f = frame_at(path, float(t), width=w)
    f = cv2.resize(f, (w, w * 9 // 16)).copy()
    lab = f"{int(t // 60)}:{t % 60:04.1f}"
    cv2.putText(f, lab, (5, 22), 0, 0.7, (0, 0, 0), 4)
    cv2.putText(f, lab, (5, 22), 0, 0.7, (0, 255, 255), 2)
    tiles.append(f)
h = tiles[0].shape[0]
rows = -(-len(tiles) // cols)
g = np.zeros((rows * h, cols * w, 3), np.uint8)
for i, f in enumerate(tiles):
    g[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = f
cv2.imwrite(out, g, [cv2.IMWRITE_JPEG_QUALITY, 82])
print(out, len(tiles), "tiles")
