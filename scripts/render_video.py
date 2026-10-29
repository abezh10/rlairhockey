"""Render a clip as a video: the camera view with detections and tracks, next to the top-down table.

Left: the broadcast frame. Every detection is a thin box (colored if the tracker used it, gray if not)
and each tracked object's smoothed position is drawn back into the image as a ring. Right: the top-down
table (the framing's median frame, warped) with the puck and both mallets at true size, their trails
over the last --trail seconds, and the puck's speed. The puck's trail runs through each impact's contact
point, so a bounce between two frames still touches the rail. Each impact (a rail or end-wall bounce, a
mallet strike, or an unexplained velocity jump) shows for half a second as a ring at the contact point,
labeled with its cause (and for bounces the restitution); "(gap)" marks one inferred inside a gap the
tracker bridged. Bridged stretches of the puck's path are drawn thin, and its rings there thin too.
Writes outputs/videos/<model>/<clip>.mp4 (H.264, so browsers and Colab can play it).
Usage: python scripts/render_video.py <clip> [<clip> ...] [--model v2] [--height 540] [--seconds 40]
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rlairhockey.geometry import Camera, Table, apply_h
from rlairhockey.render import path_points, topdown, topdown_transform
from rlairhockey.track import OBJECTS, load_config, to_table

ROOT = Path(__file__).resolve().parents[1]
COLOR = {"puck": (0, 230, 255), "mallet_l": (255, 140, 0), "mallet_r": (60, 60, 255)}   # BGR
RADIUS = {"puck": 0.041, "mallet_l": 0.05, "mallet_r": 0.05}                            # m
IMPACT = {"rail": (80, 255, 80), "end": (80, 255, 80), "mallet_l": COLOR["mallet_l"], "mallet_r": COLOR["mallet_r"],
          "other": (255, 0, 255)}


def render(clip: str, model: str, height: int, trail: float, crf: int, seconds: float = 0) -> Path:
    table, cfg = Table.load(), load_config()
    fits = json.load(open(ROOT / "data/calib/fits.json"))
    r = pd.read_csv(ROOT / "data/clips/clips.csv", keep_default_na=False).set_index("clip").loc[clip]
    g = f"{r.video}_{r.angle or 'nocam'}"
    cam = Camera.from_dict(fits[g]["camera"])
    ppm = 250
    S, _ = topdown_transform(table, ppm)
    base = topdown(cv2.imread(str(ROOT / "data/calib/frames" / f"{g}.jpg")), np.array(fits[g]["H"]), table, ppm)
    k_top = height / base.shape[0]
    base = (cv2.resize(base, (round(base.shape[1] * k_top), height)) * 0.55).astype(np.uint8)
    top_px = lambda xy: (apply_h(S, xy) * k_top).round().astype(int)

    det = pd.read_parquet(ROOT / "data/detections" / model / f"{clip}.parquet")
    det = to_table(det[det.score >= cfg["min_score"]], cam, cfg)
    tr = pd.read_parquet(ROOT / "data/tracks" / model / f"{clip}.parquet")
    T = {o: t.sort_values("frame").reset_index(drop=True) for o, t in tr.groupby("obj")}
    ev_path = ROOT / "data/tracks" / model / "events" / f"{clip}.parquet"
    events = pd.read_parquet(ev_path) if ev_path.exists() else pd.DataFrame(columns=["t", "frame", "x", "y", "kind"])
    n_show = int(0.5 * r.fps)
    trails = {o: path_points(t, events if o == "puck" else None, r.fps) for o, t in T.items()}
    used = {}                                   # (frame, rounded x, y) of each detection on a path -> object
    for o, t in T.items():
        for f, x, y in t[["frame", "x_det", "y_det"]].dropna().to_numpy():
            used[(int(f), round(x, 4), round(y, 4))] = o
    det["obj"] = [used.get((f, round(x, 4), round(y, 4)), "") for f, x, y in det[["frame", "x", "y"]].to_numpy()]
    by_frame = dict(tuple(det.groupby("frame")))

    cap = cv2.VideoCapture(str(ROOT / r.path))
    w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    k_cam = height / h_src
    w_cam = round(w_src * k_cam) // 2 * 2
    W = (w_cam + base.shape[1]) // 2 * 2
    out = ROOT / "outputs/videos" / model / f"{clip}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{W}x{height}", "-r", f"{r.fps}", "-i", "-", "-c:v", "libx264", "-crf", str(crf),
                            "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                           stdin=subprocess.PIPE)
    n_trail = max(1, int(trail * r.fps))
    z = {"puck": cfg["puck"]["z"], "mallet_l": cfg["mallet"]["z"], "mallet_r": cfg["mallet"]["z"]}
    k, last = 0, int(seconds * r.fps) if seconds else r.frames
    while k < last:
        ok, im = cap.read()
        if not ok:
            break
        left = cv2.resize(im, (w_cam, height), interpolation=cv2.INTER_AREA)
        if k in by_frame:                       # obj "" = a box the tracker didn't use
            for b in by_frame[k].itertuples():
                col = COLOR[b.obj] if b.obj else (150, 150, 150)
                p0, p1 = (int(b.x0 * k_cam), int(b.y0 * k_cam)), (int(b.x1 * k_cam), int(b.y1 * k_cam))
                cv2.rectangle(left, p0, p1, col, 2 if b.obj else 1)
                cv2.putText(left, f"{b.score:.2f}", (p0[0], p0[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        right = base.copy()
        lines = []
        for o in OBJECTS:
            t = T.get(o)
            if t is None or k >= len(t):
                continue
            tt, xy, thin = trails[o]
            i0, i1 = np.searchsorted(tt, (k - n_trail) / r.fps), np.searchsorted(tt, k / r.fps + 1e-9, "right")
            pts, ok = top_px(np.nan_to_num(xy[i0:i1])), np.isfinite(xy[i0:i1, 0])
            for i in range(1, len(pts)):
                if ok[i] and ok[i - 1]:
                    fade = 0.3 + 0.7 * i / len(pts)
                    cv2.line(right, tuple(pts[i - 1]), tuple(pts[i]), tuple(int(c * fade) for c in COLOR[o]),
                             1 if thin[i0 + i] else 2, cv2.LINE_AA)
            cur = t.iloc[k]
            if np.isfinite(cur.x):
                c = tuple(top_px([[cur.x, cur.y]])[0])
                cv2.circle(right, c, max(3, int(RADIUS[o] * ppm * k_top)), COLOR[o], 2 if np.isnan(cur.score) else -1, cv2.LINE_AA)
                u, v = cam.project([[cur.x, cur.y, z[o]]])[0] * k_cam
                cv2.circle(left, (int(u), int(v)), 9, COLOR[o], 1 if cur.get("filled", False) else 2, cv2.LINE_AA)
                if o == "puck":
                    lines.append(f"puck {np.hypot(cur.vx, cur.vy):4.1f} m/s")
        for e in events[(events.frame <= k) & (events.frame > k - n_show)].itertuples():
            c = tuple(top_px([[e.x, e.y]])[0])
            fade = 1 - (k - e.frame) / n_show
            col = tuple(int(v * (0.4 + 0.6 * fade)) for v in IMPACT[e.kind])
            cv2.circle(right, c, int(RADIUS["puck"] * ppm * k_top) + 6, col, 2, cv2.LINE_AA)
            label = f"{e.kind} e={e.restitution:.2f}" if e.kind in ("rail", "end") else e.kind
            label += "" if getattr(e, "seen", True) else " (gap)"
            cv2.putText(right, label, (c[0] + 12, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        cv2.putText(right, f"{clip}  {k / r.fps:5.2f} s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for i, s in enumerate(lines):
            cv2.putText(right, s, (10, 50 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR["puck"], 2)
        frame = np.zeros((height, W, 3), np.uint8)
        frame[:, :w_cam] = left
        frame[:, w_cam:w_cam + right.shape[1]] = right[:, :W - w_cam]
        enc.stdin.write(frame.tobytes())
        k += 1
    cap.release()
    enc.stdin.close()
    enc.wait()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--model", default="v2")
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--trail", type=float, default=1.0, help="seconds of trail behind each object")
    ap.add_argument("--crf", type=int, default=26)
    ap.add_argument("--seconds", type=float, default=0, help="render only the first this many seconds (0 = all)")
    a = ap.parse_args()
    for c in a.clips:
        print(render(c, a.model, a.height, a.trail, a.crf, a.seconds))


if __name__ == "__main__":
    main()
