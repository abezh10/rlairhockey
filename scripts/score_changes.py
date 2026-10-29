"""Record the broadcast scoreboard's points digits through each clip, to check goal detection against.

Each match's overlay shows both players' points (configs/videos.yaml, score_roi). This crops that box at
full resolution (at most 128 px wide), about 10 times a second, so check_goals.py can find where the score
changed: in gray, or for score_channel: red (red digits on dark, the 2023 final) as redness.
Writes data/clips/score/<clip>.npz: frames (the clip frames sampled) and crops (n x h x w, uint8).
Finished clips are skipped.
Usage: python scripts/score_changes.py [--clips <clip> ...] [--jobs 16]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/clips/score"


def crop(job: tuple) -> str:
    clip, path, roi, channel = job
    cap = cv2.VideoCapture(str(ROOT / path))
    step = max(1, round(cap.get(cv2.CAP_PROP_FPS) / 10))
    frames, out, k = [], [], 0
    while cap.grab():
        if k % step == 0:
            im = cap.retrieve()[1]
            sx, sy = im.shape[1] / 1920, im.shape[0] / 1080
            x0, y0, x1, y1 = (int(round(v * s)) for v, s in zip(roi, (sx, sy, sx, sy)))
            c = im[y0:y1, x0:x1].astype(np.int16)
            g = (np.clip(c[..., 2] - (c[..., 0] + c[..., 1]) // 2, 0, 255).astype(np.uint8) if channel == "red"
                 else cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY))
            if g.shape[1] > 128:
                g = cv2.resize(g, (128, round(g.shape[0] * 128 / g.shape[1])), interpolation=cv2.INTER_AREA)
            frames.append(k)
            out.append(g)
        k += 1
    np.savez_compressed(OUT / f"{clip}.npz", frames=np.array(frames), crops=np.array(out, np.uint8))
    return clip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    videos = yaml.safe_load(open(ROOT / "configs/videos.yaml"))["videos"]
    clips = pd.read_csv(ROOT / "data/clips/clips.csv", keep_default_na=False)
    if a.clips:
        clips = clips[clips["clip"].isin(a.clips)]
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(c, p, videos[v]["score_roi"], videos[v].get("score_channel", "gray"))
            for c, p, v in zip(clips["clip"], clips.path, clips.video)
            if "score_roi" in videos[v] and not (OUT / f"{c}.npz").exists()]
    print(f"{len(jobs)} clips")
    with ProcessPoolExecutor(a.jobs) as ex:
        for c in ex.map(crop, jobs):
            print(c, flush=True)


if __name__ == "__main__":
    main()
