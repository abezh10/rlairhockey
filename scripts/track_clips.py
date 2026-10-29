"""Track the puck and both mallets in every detected clip (src/rlairhockey/track.py).

Reads data/detections/<model>/<clip>.parquet and the clip's camera calibration (data/calib/fits.json);
writes data/tracks/<model>/<clip>.parquet (frame, obj, x, y, vx, vy, x_det, y_det, score, event; table
meters, in the clip camera's frame; event names the puck impact, if any, that the frame is the first one
after), data/tracks/<model>/events/<clip>.parquet (the puck's impacts: time, place, velocities in and out,
cause; track.py, smooth_puck) and data/tracks/<model>/stats.csv (per clip: boxes dropped as off the
surface or parked, the fraction of frames each object was detected on / covered after smoothing, and
impacts by cause).
Repeated frames (data/clips/changes/, find_repeats.py) and wall corrections (data/calib/walls.json,
fit_walls.py) are used when present.
Clips from framings without a calibration are listed and skipped.
Usage: python scripts/track_clips.py [--model v2] [--clips <clip> ...]
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from rlairhockey.geometry import Camera, Table
from rlairhockey.track import load_config, track_clip

ROOT = Path(__file__).resolve().parents[1]


def run(job: tuple) -> dict:
    clip, n, fps, cam, walls, src, dst = job
    ch = ROOT / "data/clips/changes" / f"{clip}.npy"
    res = track_clip(pd.read_parquet(src), n, fps, Camera.from_dict(cam), Table.load(), load_config(),
                     np.load(ch) if ch.exists() else None, None if walls is None else np.array(walls))
    res.tracks.to_parquet(dst, index=False)
    res.events.to_parquet(dst.parent / "events" / dst.name, index=False)
    return dict(clip=clip, frames=n, **res.stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v2")
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    clips = pd.read_csv(ROOT / "data/clips/clips.csv", keep_default_na=False)
    clips["group"] = clips.video + "_" + clips.angle.replace("", "nocam")
    if a.clips:
        clips = clips[clips["clip"].isin(a.clips)]
    fits = json.load(open(ROOT / "data/calib/fits.json"))
    wp = ROOT / "data/calib/walls.json"
    walls = json.load(open(wp)) if wp.exists() else {}
    src, dst = ROOT / "data/detections" / a.model, ROOT / "data/tracks" / a.model
    (dst / "events").mkdir(parents=True, exist_ok=True)
    jobs, skipped = [], []
    for c, n, fps, g in zip(clips["clip"], clips.frames, clips.fps, clips.group):
        if not (src / f"{c}.parquet").exists():
            continue
        if g not in fits:
            skipped.append(c)
            continue
        jobs.append((c, n, fps, fits[g]["camera"], walls.get(g, {}).get("A"), src / f"{c}.parquet", dst / f"{c}.parquet"))
    print(f"{len(jobs)} clips to track; {len(skipped)} detected clips have no calibration: {skipped}")
    with ProcessPoolExecutor(a.jobs) as ex:
        stats = pd.DataFrame(list(ex.map(run, jobs)))
    if len(stats):
        old = pd.read_csv(dst / "stats.csv") if (dst / "stats.csv").exists() else None
        if old is not None:
            stats = pd.concat([old[~old["clip"].isin(stats["clip"])], stats], ignore_index=True)
        stats.sort_values("clip").to_csv(dst / "stats.csv", index=False)
        print(stats[stats["clip"].isin([j[0] for j in jobs])].to_string(index=False))


if __name__ == "__main__":
    main()
