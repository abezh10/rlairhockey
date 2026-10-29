"""Measure how much each clip frame changed from the one before, to find repeated frames.

Some broadcasts were converted to a higher frame rate by showing frames twice: the 2024 final is 60 fps
with nearly every frame doubled, and the 2023 final repeats about one frame in twelve. A repeat is not a
new look at the table, so the tracker drops its detections (track.py); kept, they make the puck stop for
a frame and then jump, which reads as impossible accelerations.
Writes data/clips/changes/<clip>.npy: per frame, the number of pixels (at 480 x 270, grayscale) that
changed by more than 12 levels since the previous frame; -1 for the first. Repeats change almost none
(compression noise), new frames thousands. Finished clips are skipped.
Usage: python scripts/find_repeats.py [--clips <clip> ...] [--jobs 16]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/clips/changes"


def changes(job: tuple) -> str:
    clip, path = job
    cap = cv2.VideoCapture(str(ROOT / path))
    prev, out = None, []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(im, (480, 270), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.int16)
        out.append(-1 if prev is None else int((np.abs(g - prev) > 12).sum()))
        prev = g
    np.save(OUT / f"{clip}.npy", np.array(out, np.int32))
    return clip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    clips = pd.read_csv(ROOT / "data/clips/clips.csv", keep_default_na=False)
    if a.clips:
        clips = clips[clips["clip"].isin(a.clips)]
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(c, p) for c, p in zip(clips["clip"], clips.path) if not (OUT / f"{c}.npy").exists()]
    print(f"{len(jobs)} clips")
    with ProcessPoolExecutor(a.jobs) as ex:
        for c in ex.map(changes, jobs):
            print(c, flush=True)


if __name__ == "__main__":
    main()
