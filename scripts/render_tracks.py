"""Draw a clip's tracks over the top-down view of its camera framing, for checking by eye.

Smoothed paths are lines (puck shaded from dark at the clip start to bright at its end; mallets in their
own colors; the puck's runs through each impact's contact point, and stretches the tracker bridged over a
gap are thin); detections the tracker used are dots, and detections it rejected are small gray crosses.
Puck impacts are rings: green for a rail or end-wall bounce, the mallet's color for a strike, magenta
for an unexplained velocity jump.
Writes outputs/tracks/<model>/<clip>.png.
Usage: python scripts/render_tracks.py <clip> [<clip> ...] [--model v2]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rlairhockey.geometry import Camera, Table, apply_h
from rlairhockey.render import path_points, topdown, topdown_transform
from rlairhockey.track import load_config, to_table

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"mallet_l": (255, 120, 0), "mallet_r": (0, 120, 255)}     # BGR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--model", default="v2")
    ap.add_argument("--ppm", type=float, default=250)
    a = ap.parse_args()
    table, cfg = Table.load(), load_config()
    fits = json.load(open(ROOT / "data/calib/fits.json"))
    clips = pd.read_csv(ROOT / "data/clips/clips.csv", keep_default_na=False).set_index("clip")
    out = ROOT / "outputs/tracks" / a.model
    out.mkdir(parents=True, exist_ok=True)
    S, _ = topdown_transform(table, a.ppm)
    for c in a.clips:
        r = clips.loc[c]
        g = f"{r.video}_{r.angle or 'nocam'}"
        cam = Camera.from_dict(fits[g]["camera"])
        img = topdown(cv2.imread(str(ROOT / "data/calib/frames" / f"{g}.jpg")), np.array(fits[g]["H"]), table, a.ppm)
        img = (img * 0.6).astype(np.uint8)
        px = lambda xy: apply_h(S, xy).round().astype(int)
        det = to_table(pd.read_parquet(ROOT / "data/detections" / a.model / f"{c}.parquet"), cam, cfg)
        det = det[det.score >= cfg["min_score"]]
        tr = pd.read_parquet(ROOT / "data/tracks" / a.model / f"{c}.parquet")
        used = set(map(tuple, tr[["x_det", "y_det"]].dropna().round(5).to_numpy()))
        for x, y in det[["x", "y"]].to_numpy():
            if (round(x, 5), round(y, 5)) not in used:
                u, v = px([[x, y]])[0]
                cv2.drawMarker(img, (int(u), int(v)), (160, 160, 160), cv2.MARKER_TILTED_CROSS, 6, 1)
        ev_path = ROOT / "data/tracks" / a.model / "events" / f"{c}.parquet"
        events = pd.read_parquet(ev_path) if ev_path.exists() else None
        for obj, t in tr.groupby("obj"):
            t = t.sort_values("frame")
            _, xy, thin = path_points(t, events if obj == "puck" else None, r.fps)
            p, ok = px(np.nan_to_num(xy)), np.isfinite(xy[:, 0])
            for i in range(1, len(p)):
                if ok[i] and ok[i - 1]:
                    f = i / len(p)
                    col = (0, int(80 + 175 * f), int(80 + 175 * f)) if obj == "puck" else COLORS[obj]
                    cv2.line(img, tuple(p[i - 1]), tuple(p[i]), col, 1 if thin[i] else 2, cv2.LINE_AA)
            for u, v in px(t[["x_det", "y_det"]].dropna().to_numpy()):
                cv2.circle(img, (int(u), int(v)), 2, (255, 255, 255) if obj == "puck" else COLORS[obj], -1)
        if events is not None:
            for e in events.itertuples():
                u, v = px([[e.x, e.y]])[0]
                col = {"rail": (80, 255, 80), "end": (80, 255, 80), "other": (255, 0, 255)}.get(e.kind) or COLORS[e.kind]
                cv2.circle(img, (int(u), int(v)), 9, col, 2, cv2.LINE_AA)
        cv2.putText(img, f"{c}  ({len(tr.frame.unique())} frames, {r.fps:.0f} fps)", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(out / f"{c}.png"), img)
        print(out / f"{c}.png")


if __name__ == "__main__":
    main()
