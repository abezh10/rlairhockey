"""At each live camera switch, check whether the two cameras' table frames agree or are rotated 180 degrees.

Each calibration names the table's ends and sides as its own camera sees them, so a camera on the other
side of the table has its frame turned 180 degrees about the center. Across a live switch (gap_before = 0)
the puck and mallets can't jump, so their positions just before and just after the cut tell which: compare
them as they are and with (x, y) -> (-x, -y) (which also swaps the two mallets). The puck is extrapolated
over the cut with its velocity.
Writes data/tracks/<model>/switches.csv (one row per switch) and prints the verdict per camera pair.
Usage: python scripts/check_switches.py [--model v2]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SWAP = {"puck": "puck", "mallet_l": "mallet_r", "mallet_r": "mallet_l"}


def ends(tr: pd.DataFrame, last: bool, n: int) -> dict:
    """Mean position (and velocity) of each object over the n frames at one end of a clip, where tracked."""
    out = {}
    for obj, t in tr.groupby("obj"):
        t = t.sort_values("frame").dropna(subset=["x"])
        t = t[t.frame >= tr.frame.max() - n + 1] if last else t[t.frame < n]
        if len(t):
            r = t.iloc[-1] if last else t.iloc[0]
            out[obj] = (np.array([r.x, r.y]), np.array([r.vx, r.vy]), int(r.frame))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v2")
    ap.add_argument("--window", type=float, default=0.1, help="look this far either side of the cut (s)")
    a = ap.parse_args()
    clips = pd.read_csv(ROOT / "data/clips/clips.csv", keep_default_na=False)
    clips["group"] = clips.video + "_" + clips.angle.replace("", "nocam")
    tdir = ROOT / "data/tracks" / a.model
    rows = []
    for vid, c in clips.groupby("video"):
        c = c.sort_values("seg").reset_index(drop=True)
        for i in range(1, len(c)):
            prev, cur = c.iloc[i - 1], c.iloc[i]
            if cur.gap_before != 0.0 or prev.group == cur.group:
                continue
            fa, fb = tdir / f"{prev['clip']}.parquet", tdir / f"{cur['clip']}.parquet"
            if not (fa.exists() and fb.exists()):
                continue
            n = max(1, int(round(a.window * cur.fps)))
            A = ends(pd.read_parquet(fa), True, n)
            B = ends(pd.read_parquet(fb), False, n)
            dt_end = (prev.frames - 1)                        # frame index of the cut in clip A
            d = {}
            for name, rot in (("same", 1), ("rotated", -1)):
                errs = []
                for obj, (pb, vb, kb) in B.items():
                    src = obj if rot == 1 else SWAP[obj]
                    if src not in A:
                        continue
                    pa, va, ka = A[src]
                    gap = (dt_end - ka + 1 + kb) / cur.fps        # seconds between the two observations
                    pred = pa + (va * gap if obj == "puck" else 0)
                    errs.append(np.linalg.norm(rot * pred - pb))
                d[name] = np.mean(errs) if errs else np.nan
                d[f"n_{name}"] = len(errs)
            rows.append(dict(video=vid, from_clip=prev["clip"], to_clip=cur["clip"], from_group=prev.group,
                             to_group=cur.group, err_same=d["same"], err_rotated=d["rotated"], objects=d["n_same"]))
    sw = pd.DataFrame(rows)
    if not len(sw):
        print("no live switches with tracks on both sides")
        return
    sw["verdict"] = np.where(sw.err_same < sw.err_rotated, "same", "rotated")
    sw.to_csv(tdir / "switches.csv", index=False)
    pair = sw.groupby(["from_group", "to_group"]).agg(
        switches=("verdict", "size"), same=("verdict", lambda v: (v == "same").sum()),
        err_same=("err_same", "median"), err_rotated=("err_rotated", "median")).round(3)
    print(pair.to_string())


if __name__ == "__main__":
    main()
