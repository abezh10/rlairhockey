"""Correct each framing's calibration so the puck bounces off the rails where the rails are.

The hand-clicked calibration is a few cm off in places, and it shows in the tracks: a framing's rail
bounces land consistently past the rail line (the puck goes into the wall) or short of it (it turns
before reaching the wall). A bounce's contact point is the puck's center touching a wall, so the bounces
off each wall trace a line one puck radius inside it. This fits those lines in the tracked table frame
(a rail's contacts as offset + slope along the rail, an end wall's as an offset; each shrunk toward no
change when there are few bounces, and left alone under MIN_PER_WALL) and the homography on the table plane that takes the four lines to
where they belong. track.py applies it after projecting each detection.
Contacts come from the current tracks (data/tracks/<model>/events, which already include any correction
in data/calib/walls.json), and the new correction composes with the old, so rerunning this after
re-tracking refines it.
Writes data/calib/walls.json: framing -> A (3x3, tracked table -> corrected table), bounce counts per wall,
and the offsets (cm, + = past the line) it removed.
--check prints what it would remove without writing (after a final re-track: what's left).
Usage: python scripts/fit_walls.py [--model v2] [--min-bounces 8] [--check]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rlairhockey.geometry import Table
from rlairhockey.track import load_config

ROOT = Path(__file__).resolve().parents[1]
PRIOR = {"offset": 10, "slope": 100}       # bounces' worth of pull toward no change
MIN_PER_WALL = 5                           # a wall with fewer bounces stays where it is
MAX_OFFSET = 0.10                          # m; a larger one is a tracking problem, not calibration


def robust_line(x: np.ndarray, off: np.ndarray, slope: bool) -> np.ndarray:
    """offset ~ a + b x (b = 0 unless slope) by least squares, refitted without points 3 MADs out."""
    keep = np.ones(len(x), bool)
    for _ in range(3):
        A = np.c_[np.ones(len(x)), x] if slope else np.ones((len(x), 1))
        c = np.linalg.lstsq(A[keep], off[keep], rcond=None)[0]
        r = off - A @ c
        mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r) <= max(3 * mad, 0.01)
    return np.r_[c, 0.0][:2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v2")
    ap.add_argument("--min-bounces", type=int, default=8, help="per framing, else it's left as it is")
    ap.add_argument("--check", action="store_true", help="print the offsets, don't write")
    a = ap.parse_args()
    table, cfg = Table.load(), load_config()
    r = cfg["impacts"]["puck_r"]
    ly, lx = table.width / 2 - r, table.length / 2 - r          # where contact points belong
    clips = pd.read_csv(ROOT / "data/clips/clips.csv", keep_default_na=False).set_index("clip")
    ev = []
    for p in sorted((ROOT / "data/tracks" / a.model / "events").glob("*.parquet")):
        e = pd.read_parquet(p)
        c = clips.loc[p.stem]
        ev.append(e.assign(group=f"{c.video}_{c.angle or 'nocam'}"))
    ev = pd.concat(ev, ignore_index=True)
    ev = ev[ev.kind.isin(["rail", "end"]) & ev.seen.astype(bool)]
    path = ROOT / "data/calib/walls.json"
    old = json.load(open(path)) if path.exists() else {}
    out, rows = {}, []
    for g, e in ev.groupby("group"):
        A_old = np.array(old.get(g, {}).get("A", np.eye(3)))
        # contacts were snapped onto the wall by constrain(); offset keeps how far past the line they were
        x, y = e.x.to_numpy(float), e.y.to_numpy(float)
        off = e.offset.to_numpy(float)
        walls = {"far": (e.kind == "rail").to_numpy() & (y > 0), "near": (e.kind == "rail").to_numpy() & (y < 0),
                 "left": (e.kind == "end").to_numpy() & (x < 0), "right": (e.kind == "end").to_numpy() & (x > 0)}
        n = {k: int(m.sum()) for k, m in walls.items()}
        if sum(n.values()) < a.min_bounces:
            if g in old:
                out[g] = old[g]
            continue
        fit = {}
        for k, m in walls.items():
            if n[k] < MIN_PER_WALL:
                fit[k] = np.zeros(2)
                continue
            along = x[m] if k in ("far", "near") else y[m]
            c = robust_line(along, off[m], slope=k in ("far", "near"))
            fit[k] = c * [n[k] / (n[k] + PRIOR["offset"]), n[k] / (n[k] + PRIOR["slope"])]
            fit[k][0] = np.clip(fit[k][0], -MAX_OFFSET, MAX_OFFSET)
        # the walls as tracked (contact lines), as functions along each wall; their corners -> true corners
        far = lambda x: ly + fit["far"][0] + fit["far"][1] * x
        near = lambda x: -ly - fit["near"][0] - fit["near"][1] * x
        left, right = -lx - fit["left"][0], lx + fit["right"][0]
        src = np.array([[left, far(left)], [right, far(right)], [right, near(right)], [left, near(left)]])
        dst = np.array([[-lx, ly], [lx, ly], [lx, -ly], [-lx, -ly]])
        D = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)).astype(float)
        A = D @ A_old
        out[g] = dict(A=(A / A[2, 2]).tolist(), bounces=n,
                      offsets_cm={k: [round(v * 100, 1) for v in fit[k]] for k in fit})
        rows.append(dict(framing=g, **{f"n_{k}": v for k, v in n.items()},
                         **{k: round(fit[k][0] * 100, 1) for k in fit},
                         corner_shift_cm=round(float(np.abs(src - dst).max()) * 100, 1)))
    for g in old:
        out.setdefault(g, old[g])
    print("offsets (cm past the rail line, + = into the wall):")
    print(pd.DataFrame(rows).to_string(index=False))
    if not a.check:
        json.dump(out, open(path, "w"), indent=1)
        print(path)


if __name__ == "__main__":
    main()
