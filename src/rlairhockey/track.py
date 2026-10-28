"""Per-frame detections -> one puck track and one mallet track per table half, in table coordinates.

For each clip:
  1. Project each box center onto the table: its viewing ray meets the plane at that class's box-center
     height (configs/track.yaml), so tall mallets don't land too far from the camera.
  2. Drop detections off the playing surface (the goal mouths count as on).
  3. Drop parked objects: a spare puck or mallet that sits still for most of the clip while another
     candidate of its kind is in view. One that is the only candidate is never dropped, so a puck
     waiting for a faceoff stays.
  4. Pick each target's path through the remaining candidates by dynamic programming: a detection
     earns its score, a step costs its length (in meters, and relative to the farthest the object could
     have moved),
     a frame without a detection costs a gap penalty, and losing the target (to pick it up anywhere
     later) costs a fixed amount. The path follows whatever is consistently there and moves plausibly.
  5. Smooth with a constant-velocity Kalman filter and RTS pass: positions, velocities, short gaps filled.

Mallets can't cross the center line, so each half is its own single-target problem: mallet_l on x < 0,
mallet_r on x > 0. Coordinates are in the clip's camera frame (geometry.py: "left" = -x as that camera
sees the table); putting a match's cameras in one frame is the stitching step's job.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .geometry import ROOT, Camera, Table

OBJECTS = ("puck", "mallet_l", "mallet_r")


def load_config(path: str | Path = ROOT / "configs/track.yaml") -> dict:
    return yaml.safe_load(open(path))


def to_table(det: pd.DataFrame, cam: Camera, cfg: dict) -> pd.DataFrame:
    """Add table x, y for each box center, intersecting its ray with the plane at the class's height."""
    det = det.copy()
    uv = np.c_[(det.x0 + det.x1) / 2, (det.y0 + det.y1) / 2]
    xy = np.full((len(det), 2), np.nan)
    for cls in ("puck", "mallet"):
        m = (det.cls == cls).to_numpy()
        if m.any():
            xy[m] = cam.to_table(uv[m], z=cfg[cls]["z"])[:, :2]
    det["x"], det["y"] = xy[:, 0], xy[:, 1]
    return det


def on_surface(det: pd.DataFrame, table: Table, cfg: dict) -> np.ndarray:
    L, W = table.length / 2, table.width / 2
    ok = np.zeros(len(det), bool)
    for cls in ("puck", "mallet"):
        c = cfg[cls]
        m = (det.cls == cls).to_numpy()
        x, y = det.x.to_numpy()[m], det.y.to_numpy()[m]
        inside = (np.abs(x) <= L + c["margin"]) & (np.abs(y) <= W + c["margin"])
        if cls == "puck" and table.goal_width:     # into the goal mouth
            inside |= (np.abs(x) <= L + c["goal_depth"]) & (np.abs(y) <= table.goal_width / 2)
        ok[m] = inside
    return ok


def parked(det: pd.DataFrame, n_frames: int, cfg: dict) -> np.ndarray:
    """Mask of detections belonging to a parked object (det: one target's candidates)."""
    p = cfg["parked"]
    out = np.zeros(len(det), bool)
    xy, fr = det[["x", "y"]].to_numpy(), det.frame.to_numpy()
    for _ in range(3):                                   # at most a few spares per target
        live = ~out
        if live.sum() == 0:
            break
        cells = np.round(xy[live] / p["radius"]).astype(int)
        _, inv, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
        center = np.median(xy[live][inv.ravel() == counts.argmax()], axis=0)
        near = live & (np.linalg.norm(xy - center, axis=1) <= p["radius"])
        frames = np.unique(fr[near])
        if len(frames) < p["min_frac"] * n_frames:
            break
        others = np.unique(fr[live & ~near & (np.linalg.norm(xy - center, axis=1) > 5 * p["radius"])])
        if len(np.intersect1d(frames, others)) < p["coexist"] * len(frames):
            break
        out |= near
    return out


def best_path(det: pd.DataFrame, n_frames: int, fps: float, vmax: float, cfg: dict) -> np.ndarray:
    """Indices (into det) of the detections on the best path; see the module docstring, step 4."""
    c = cfg["path"]
    K = max(1, int(round(c["max_skip"] * fps)))
    xy, sc, fr = det[["x", "y"]].to_numpy(), det.score.to_numpy(), det.frame.to_numpy()
    by_frame = [[] for _ in range(n_frames)]
    for i, f in enumerate(fr):
        by_frame[f].append(i)
    INF = float("inf")
    cost = np.full(len(det), INF)
    prev = np.full(len(det), -1)                 # predecessor detection, or -1 = came from lost
    lost = np.zeros(n_frames + 1)                # lost[t + 1]: best cost of being lost at frame t
    lost_prev = np.full(n_frames + 1, -1)        # -1 = was lost the frame before too
    for t in range(n_frames):
        for j in by_frame[t]:
            best, arg = lost[t] + c["lost"] / 2, -1
            for k in range(1, min(K, t) + 1):
                for i in by_frame[t - k]:
                    d = np.hypot(*(xy[j] - xy[i]))
                    reach = vmax * k / fps + 0.05
                    if d <= reach:
                        v = cost[i] + c["w_move"] * d / reach + c["w_dist"] * d + c["gap"] * (k - 1)
                        if v < best:
                            best, arg = v, i
            cost[j], prev[j] = best - c["w_score"] * sc[j], arg
        enter = [(cost[i] + c["lost"] / 2, i) for i in (by_frame[t - 1] if t else [])]
        stay = (lost[t], -1)
        v, a = min([stay] + enter)
        lost[t + 1], lost_prev[t + 1] = v + c["gap"], a
    # end: lost at the last frame, or a detection in the last K frames with the frames after it as gaps
    ends = [(lost[n_frames], None)] + [(cost[i] + c["gap"] * (n_frames - 1 - fr[i]), i)
                                       for t in range(max(0, n_frames - K), n_frames) for i in by_frame[t]]
    _, node = min(ends, key=lambda e: e[0])
    path, t = [], n_frames - 1
    while True:
        if node is None:                          # walk back through lost frames
            while t >= 0 and lost_prev[t + 1] == -1:
                t -= 1
            if t < 0:
                break
            node = lost_prev[t + 1]
        path.append(node)
        p = prev[node]
        t = fr[node] - 1
        node = p if p >= 0 else None
        if node is None and t < 0:
            break
    return np.array(path[::-1], int)


def smooth(frames: np.ndarray, xy: np.ndarray, n_frames: int, fps: float, meas_sigma: float,
           accel_sigma: float, max_gap: float) -> np.ndarray:
    """Constant-velocity Kalman filter + RTS smoother over all frames from the first to the last
    measurement. Returns (n_frames, 4) x, y, vx, vy; NaN outside, and inside gaps longer than max_gap."""
    out = np.full((n_frames, 4), np.nan)
    if len(frames) == 0:
        return out
    dt = 1 / fps
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
    Q = G @ G.T * accel_sigma ** 2
    H = np.eye(2, 4)
    R = np.eye(2) * meas_sigma ** 2
    z = np.full((n_frames, 2), np.nan)
    z[frames] = xy
    t0, t1 = frames.min(), frames.max()
    n = t1 - t0 + 1
    xs, Ps, xp, Pp = np.zeros((n, 4)), np.zeros((n, 4, 4)), np.zeros((n, 4)), np.zeros((n, 4, 4))
    x = np.r_[z[t0], 0, 0]
    P = np.diag([meas_sigma ** 2] * 2 + [10.0 ** 2] * 2)
    for i in range(n):
        if i:
            x, P = F @ x, F @ P @ F.T + Q
        xp[i], Pp[i] = x, P
        if not np.isnan(z[t0 + i, 0]):
            S = H @ P @ H.T + R
            Kg = P @ H.T @ np.linalg.inv(S)
            x = x + Kg @ (z[t0 + i] - H @ x)
            P = (np.eye(4) - Kg @ H) @ P
        xs[i], Ps[i] = x, P
    for i in range(n - 2, -1, -1):                # RTS
        C = Ps[i] @ F.T @ np.linalg.inv(Pp[i + 1])
        xs[i] = xs[i] + C @ (xs[i + 1] - xp[i + 1])
        Ps[i] = Ps[i] + C @ (Ps[i + 1] - Pp[i + 1]) @ C.T
    out[t0:t1 + 1] = xs
    seen = frames - t0                            # blank gaps longer than max_gap
    for a, b in zip(seen[:-1], seen[1:]):
        if (b - a - 1) / fps > max_gap:
            out[t0 + a + 1:t0 + b] = np.nan
    return out


@dataclass
class ClipTracks:
    tracks: pd.DataFrame          # frame, obj, x, y, vx, vy, x_det, y_det, score (NaN when not detected)
    stats: dict


def track_clip(det: pd.DataFrame, n_frames: int, fps: float, cam: Camera, table: Table,
               cfg: dict | None = None) -> ClipTracks:
    cfg = cfg or load_config()
    det = det[det.score >= cfg["min_score"]]
    det = to_table(det, cam, cfg)
    surf = on_surface(det, table, cfg)
    stats = dict(boxes=len(det), off_surface=int((~surf).sum()))
    det = det[surf]
    half = cfg["mallet"]["center_overlap"]
    targets = {"puck": det[det.cls == "puck"],
               "mallet_l": det[(det.cls == "mallet") & (det.x < half)],
               "mallet_r": det[(det.cls == "mallet") & (det.x > -half)]}
    rows = []
    for obj, d in targets.items():
        d = d.reset_index(drop=True)
        pk = parked(d, n_frames, cfg)
        stats[f"{obj}_parked"] = int(pk.sum())
        d = d[~pk].reset_index(drop=True)
        cls = "puck" if obj == "puck" else "mallet"
        path = d.iloc[best_path(d, n_frames, fps, cfg[cls]["vmax"], cfg)] if len(d) else d
        s = smooth(path.frame.to_numpy(), path[["x", "y"]].to_numpy(), n_frames, fps,
                   cfg["smooth"]["meas_sigma"][cls], cfg["smooth"]["accel_sigma"][cls], cfg["smooth"]["max_gap"])
        raw = np.full((n_frames, 3), np.nan)
        raw[path.frame.to_numpy()] = path[["x", "y", "score"]].to_numpy()
        rows.append(pd.DataFrame(dict(frame=np.arange(n_frames, dtype=np.int32), obj=obj,
                                      x=s[:, 0], y=s[:, 1], vx=s[:, 2], vy=s[:, 3],
                                      x_det=raw[:, 0], y_det=raw[:, 1], score=raw[:, 2])))
        stats[f"{obj}_detected"] = round(len(path) / n_frames, 3)
        stats[f"{obj}_covered"] = round(float(np.isfinite(s[:, 0]).mean()), 3)
    return ClipTracks(pd.concat(rows, ignore_index=True), stats)
