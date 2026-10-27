"""One clean frame per camera framing, for clicking table landmarks.

For every (video, camN) with at least --min-play minutes of play, takes the per-pixel
median of --n frames spread over its longest play segment. The puck, mallets and most
of the players' motion drop out, leaving the table edges unoccluded. Drift between the
medians of the segment's two halves (phase correlation, px) flags a moving camera.

Writes data/calib/frames/<id>_<camN>.jpg and data/calib/frames/frames.csv.
Usage: python scripts/calib_frames.py [--min-play 0.5] [--n 40] [<id> ...]
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from rlairhockey.video import iter_frames, probe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/calib/frames"


def median_frame(vid: str, angle: str, start: float, end: float, n: int) -> dict:
    src = ROOT / "data/raw_videos" / f"{vid}.mp4"
    want = np.linspace(start + 0.2, end - 0.2, n)
    frames, k = [], 0
    for t, f in iter_frames(src, width=probe(src).width, start=start, end=end):
        if k < n and t >= want[k]:
            frames.append(f)
            while k < n and t >= want[k]:
                k += 1
        if k == n:
            break
    stack = np.stack(frames)
    med = np.median(stack, axis=0).astype(np.uint8)
    h = len(stack) // 2
    a, b = (cv2.cvtColor(np.median(s, axis=0).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            for s in (stack[:h], stack[h:]))
    (dx, dy), _ = cv2.phaseCorrelate(a, b)
    name = f"{vid}_{angle}"
    cv2.imwrite(str(OUT / f"{name}.jpg"), med, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return dict(frame=name, video=vid, angle=angle, start=round(start, 2), end=round(end, 2),
                n=len(frames), drift_px=round(float(np.hypot(dx, dy)), 2), width=med.shape[1], height=med.shape[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--min-play", type=float, default=0.5, help="minutes")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/videos.yaml"))["videos"]
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for vid in a.ids or list(cfg):
        seg = pd.read_csv(ROOT / "data/segments" / f"{vid}_segments.csv", keep_default_na=False)
        play = seg[(seg.label == "play") & (seg.angle != "")]
        for angle, g in play.groupby("angle"):
            if g.dur.sum() / 60 < a.min_play:
                continue
            r = g.loc[g.dur.idxmax()]
            jobs.append((vid, angle, float(r.start), float(r.end), a.n, round(g.dur.sum() / 60, 2)))
    with ProcessPoolExecutor(min(len(jobs), len(os.sched_getaffinity(0)))) as ex:
        rows = list(ex.map(median_frame, *zip(*[j[:5] for j in jobs])))
    df = pd.DataFrame(rows)
    df.insert(3, "play_min", [j[5] for j in jobs])
    df = df.sort_values("play_min", ascending=False)
    df.to_csv(OUT / "frames.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
