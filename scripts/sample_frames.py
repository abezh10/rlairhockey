"""Pick frames from the play clips for labeling in Roboflow (puck, mallet).

Frames are spread over videos and camera framings: each (video, framing) gets a share of
--n proportional to the square root of its play time, at least --min-per-group. Within a
group, frames are drawn with probability rising with motion on the table (so fast pucks and
streaks show up often) and kept at least --min-gap seconds apart. Frames come from the raw
video at full resolution.

Writes data/frames/<set>/<id>_f<frame>.jpg and data/frames/<set>/manifest.csv.
Usage: python scripts/sample_frames.py [--n 500] [--set detector] [--seed 0]
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rlairhockey.geometry import Table, apply_h
from rlairhockey.video import frame_at

ROOT = Path(__file__).resolve().parents[1]
GRID_W, GRID_H = 32, 18


def table_mask(H: list | None) -> np.ndarray:
    """Motion-grid cells covered by the table, from a calibration homography (whole grid if none)."""
    if H is None:
        return np.ones((GRID_H, GRID_W), np.float32)
    t = Table.load()
    L, W = t.length / 2, t.width / 2
    quad = apply_h(np.array(H), [[-L, W], [L, W], [L, -W], [-L, -W]]) * [GRID_W / 1920, GRID_H / 1080]
    m = np.zeros((GRID_H * 4, GRID_W * 4), np.uint8)
    cv2.fillPoly(m, [np.round(quad * 4).astype(np.int32)], 1)
    return cv2.resize(m.astype(np.float32), (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)


def extract(job: tuple) -> None:
    src, t, dst = job
    cv2.imwrite(str(dst), frame_at(src, t), [cv2.IMWRITE_JPEG_QUALITY, 95])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--set", default="detector")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-gap", type=float, default=1.0)
    ap.add_argument("--min-per-group", type=int, default=4)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    clips = pd.read_csv(ROOT / "data/clips/clips.csv", keep_default_na=False)
    clips["group"] = clips.video + "_" + clips.angle.replace("", "nocam")
    fits_path = ROOT / "data/calib/fits.json"
    fits = json.loads(fits_path.read_text()) if fits_path.exists() else {}

    minutes = clips.groupby("group").dur.sum() / 60
    share = np.sqrt(minutes) / np.sqrt(minutes).sum() * a.n
    alloc = np.maximum(a.min_per_group, np.round(share)).astype(int)
    alloc = (alloc * a.n / alloc.sum()).round().astype(int).clip(lower=1)

    rows = []
    for vid, vclips in clips.groupby("video"):
        feat = pd.read_parquet(ROOT / "data/features" / f"{vid}.parquet", columns=["t"])
        grid = np.load(ROOT / "data/features" / f"{vid}_grid.npy", mmap_mode="r")
        t_all = feat.t.to_numpy()
        for g, gclips in vclips.groupby("group"):
            mask = table_mask(fits.get(g, {}).get("H"))
            idx = np.concatenate([np.flatnonzero((t_all >= r.start + 0.3) & (t_all < r.end - 0.3)) for r in gclips.itertuples()])
            if len(idx) == 0:
                continue
            motion = (grid[idx].astype(np.float32) / 255 * mask).sum(axis=(1, 2)) / max(mask.sum(), 1e-6)
            w = np.sqrt(motion / max(motion.mean(), 1e-6)) + 0.5
            w /= w.sum()
            chosen: list[float] = []
            for i in rng.choice(len(idx), size=min(len(idx), alloc[g] * 20), replace=False, p=w):
                t = float(t_all[idx[i]])
                if all(abs(t - c) >= a.min_gap for c in chosen):
                    chosen.append(t)
                    clip = gclips[(gclips.start <= t) & (gclips.end > t)].iloc[0]
                    rows.append(dict(video=vid, group=g, angle=clip.angle, clip=clip["clip"], t=round(t, 4),
                                     frame=int(round(t * clip.fps)), motion=round(float(motion[i]), 4)))
                if len(chosen) == alloc[g]:
                    break

    out = ROOT / "data/frames" / a.set
    out.mkdir(parents=True, exist_ok=True)
    man = pd.DataFrame(rows).sort_values(["video", "t"])
    man.insert(0, "file", [f"{r.video}_f{r.frame:06d}.jpg" for r in man.itertuples()])
    jobs = [(ROOT / "data/raw_videos" / f"{r.video}.mp4", r.t, out / r.file) for r in man.itertuples()]
    with ProcessPoolExecutor(len(os.sched_getaffinity(0))) as ex:
        list(ex.map(extract, jobs, chunksize=4))
    man.to_csv(out / "manifest.csv", index=False)
    print(man.groupby("group").size().rename("frames").to_string())
    print(f"{len(man)} frames -> {out}")


if __name__ == "__main__":
    main()
