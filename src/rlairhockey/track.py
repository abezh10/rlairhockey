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
     The puck's velocity jumps at impacts (a rail, an end wall, a mallet), and smoothing across one would
     round the corner and blend the velocities in and out over several frames. So the puck's path is first
     split where its motion stops being smooth (impacts()), each impact is identified by where it happened,
     and the pieces between impacts are smoothed separately and extrapolated to meet at the impact.

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


def _suffix_sse(t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Residual sum of squares of a quadratic fit of xy(t) over every suffix t[i:] (inf under 3 points)."""
    s = t - t[-1]
    pw = np.stack([s ** k for k in range(5)], 1)
    P = pw[::-1].cumsum(0)[::-1]                                  # sums of s^k over each suffix
    B = (pw[:, :3, None] * xy[:, None, :])[::-1].cumsum(0)[::-1]  # sums of s^k * (x, y)
    Q = (xy ** 2).sum(1)[::-1].cumsum()[::-1]
    sse = np.full(len(t), np.inf)
    ok = np.arange(len(t)) <= len(t) - 3
    if ok.any():
        M = np.stack([P[ok, k:k + 3] for k in range(3)], 1)       # normal matrices
        c = np.linalg.solve(M, B[ok])
        sse[ok] = np.maximum(Q[ok] - (c * B[ok]).sum((1, 2)), 0)
    return sse


def pieces(t: np.ndarray, xy: np.ndarray, cfg: dict) -> list[tuple[int, int]]:
    """Split detections into pieces of smooth motion, as [start, end) index pairs: the partition that
    minimizes each piece's quadratic-fit residual (in units of the detection noise) plus a cost per piece.
    Pieces are at least min_len detections and at most window seconds (longer motion gets split and the
    caller rejoins it, finding no impact there)."""
    m, n = cfg["min_len"], len(t)
    if n < 2 * m:
        return [(0, n)]
    F = np.full(n + 1, np.inf)                    # F[j]: best cost of detections [0, j)
    F[0] = 0
    start = np.zeros(n + 1, int)
    for j in range(m, n + 1):
        lo = min(int(np.searchsorted(t, t[j - 1] - cfg["window"])), j - m)
        cand = F[lo:j - m + 1] + _suffix_sse(t[lo:j], xy[lo:j])[:j - m - lo + 1] / cfg["sigma"] ** 2 + cfg["penalty"]
        k = int(np.argmin(cand))
        F[j], start[j] = cand[k], lo + k
    if not np.isfinite(F[n]):
        return [(0, n)]
    out, j = [], n
    while j > 0:
        out.append((start[j], j))
        j = start[j]
    return out[::-1]


def _line(t: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares (p0, v) of xy = p0 + v * t, as a (2, 2) array, and its residual sum of squares."""
    A = np.c_[np.ones_like(t), t]
    c = np.linalg.lstsq(A, xy, rcond=None)[0]
    return c, float(((A @ c - xy) ** 2).sum())


def impact(t: np.ndarray, xy: np.ndarray, a: tuple, b: tuple, mallets: dict, fps: float, table: Table,
           cfg: dict) -> tuple[dict, int] | None:
    """What happened between piece a and piece b (index ranges into t, xy), if anything, and the index
    the impact really splits them at (the pieces' boundary can be a detection or two off). The motions
    in and out are lines through the detections either side; the impact is where they meet. A mallet
    within reach then makes it a strike; otherwise a rail or end wall the puck was moving into and is
    moving away from makes it a bounce; an unexplained change only counts when it's large.
    None when the velocity barely changed."""
    w = max(int(round(cfg["fit"] * fps)), cfg["min_len"])
    best = None
    for k in range(max(a[0] + 3, b[0] - 2), min(b[1] - 3, b[0] + 2) + 1):
        (la, ra), (lb, rb) = (_line(t[i] - t[k], xy[i]) for i in (np.arange(max(a[0], k - w), k),
                                                                   np.arange(k, min(b[1], k + w))))
        if best is None or ra + rb < best[0]:
            best = (ra + rb, k, la, lb)
    if best is None:
        return None
    _, k, la, lb = best
    va, vb = la[1], lb[1]
    dv, speed = np.hypot(*(vb - va)), max(np.hypot(*va), np.hypot(*vb))
    if dv < max(cfg["min_dv"], cfg["rel_dv"] * speed):
        return None
    d0 = la[0] - lb[0]                                            # where the two lines come closest
    s = np.clip(-d0 @ (va - vb) / max((va - vb) @ (va - vb), 1e-9), t[k - 1] - t[k] - 1 / fps, 1 / fps)
    t0, c = t[k] + s, (la[0] + lb[0] + s * (va + vb)) / 2
    ev = dict(t=t0, x=c[0], y=c[1], vx_in=va[0], vy_in=va[1], vx_out=vb[0], vy_out=vb[1],
              kind="other", offset=np.nan, restitution=np.nan)
    reach = cfg["puck_r"] + cfg["mallet_r"] + cfg["mallet_tol"]
    near = {}                                     # closest each mallet came within a couple of frames
    for o, m in mallets.items():
        fs = np.arange(int(np.floor(t0 * fps)) - 2, int(np.ceil(t0 * fps)) + 3)
        d = np.hypot(*(m[fs[(fs >= 0) & (fs < len(m))], :2] - c).T)
        if np.isfinite(d).any() and np.nanmin(d) <= reach:
            near[o] = np.nanmin(d)
    if near:
        ev["kind"] = min(near, key=near.get)
        return ev, k
    walls = (("rail", 1, table.width / 2), ("end", 0, table.length / 2))
    for kind, ax, half in walls:
        s, lim = np.sign(c[ax]), half - cfg["puck_r"]
        into, out = s * va[ax], -s * vb[ax]
        mouth = kind == "end" and table.goal_width and abs(c[1]) < table.goal_width / 2 - cfg["puck_r"]
        if abs(c[ax]) > lim - cfg["rail_tol"] and into > 0 and out > 0 and not mouth:
            ev.update(kind=kind, offset=abs(c[ax]) - lim, restitution=out / into)
            return ev, k
    return (ev, k) if dv >= cfg["other_dv"] else None


def smooth_puck(frames: np.ndarray, xy: np.ndarray, n_frames: int, fps: float, mallets: dict, table: Table,
                cfg: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """smooth() for the puck, with its velocity free to jump at impacts (see impact()). Returns the
    (n_frames, 4) track and the impacts (t, frame = first frame after it, kind, contact x, y, velocities
    in and out, and for bounces the contact's distance past the rail line and the normal restitution)."""
    sm, ic = cfg["smooth"], cfg["impacts"]
    args = (sm["meas_sigma"]["puck"], sm["accel_sigma"]["puck"], sm["max_gap"])
    out = np.full((n_frames, 4), np.nan)
    events = []
    t = frames / fps
    for run in np.split(np.arange(len(frames)), np.flatnonzero(np.diff(frames) - 1 > sm["max_gap"] * fps) + 1):
        if not len(run):
            continue
        ps = [(run[0] + a, run[0] + b) for a, b in pieces(t[run], xy[run], ic)]
        joined, cuts = [ps[0]], []                # pieces with an impact between them
        for p in ps[1:]:
            hit = impact(t, xy, joined[-1], p, mallets, fps, table, ic)
            if hit is None:
                joined[-1] = (joined[-1][0], p[1])
            else:
                ev, k = hit
                joined[-1] = (joined[-1][0], k)
                joined.append((k, p[1]))
                cuts.append(ev)
        for i, (a, b) in enumerate(joined):
            s = smooth(frames[a:b], xy[a:b], n_frames, fps, *args)
            out[frames[a]:frames[b - 1] + 1] = s[frames[a]:frames[b - 1] + 1]
            if i:                                 # fill up to the impact from both sides
                ev, fa, fb = cuts[i - 1], frames[joined[i - 1][1] - 1], frames[a]
                for f in range(fa + 1, fb):
                    src = fa if f / fps < ev["t"] else fb
                    out[f, :2] = out[src, :2] + out[src, 2:] * (f - src) / fps
                    out[f, 2:] = out[src, 2:]
        events += cuts
    ev = pd.DataFrame(events, columns=["t", "x", "y", "vx_in", "vy_in", "vx_out", "vy_out", "kind", "offset",
                                       "restitution"])
    ev.insert(1, "frame", np.ceil(ev.t * fps - 1e-6).astype(int))
    return out, ev


@dataclass
class ClipTracks:
    tracks: pd.DataFrame          # frame, obj, x, y, vx, vy, x_det, y_det, score (NaN when not detected), event
    events: pd.DataFrame          # puck impacts, see smooth_puck()
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
    rows, smoothed = {}, {}
    for obj in ("mallet_l", "mallet_r", "puck"):  # mallets first: the puck's impacts look for them
        d = targets[obj].reset_index(drop=True)
        pk = parked(d, n_frames, cfg)
        stats[f"{obj}_parked"] = int(pk.sum())
        d = d[~pk].reset_index(drop=True)
        cls = "puck" if obj == "puck" else "mallet"
        path = d.iloc[best_path(d, n_frames, fps, cfg[cls]["vmax"], cfg)] if len(d) else d
        fr, xy = path.frame.to_numpy(), path[["x", "y"]].to_numpy()
        event = np.full(n_frames, "", object)
        if obj == "puck":
            s, events = smooth_puck(fr, xy, n_frames, fps, smoothed, table, cfg)
            event[events.frame.clip(0, n_frames - 1)] = events.kind
            for k, v in events.kind.value_counts().items():
                stats[f"puck_{k}"] = int(v)
        else:
            s = smooth(fr, xy, n_frames, fps, cfg["smooth"]["meas_sigma"][cls], cfg["smooth"]["accel_sigma"][cls],
                       cfg["smooth"]["max_gap"])
            smoothed[obj] = s
        raw = np.full((n_frames, 3), np.nan)
        raw[fr] = path[["x", "y", "score"]].to_numpy()
        rows[obj] = pd.DataFrame(dict(frame=np.arange(n_frames, dtype=np.int32), obj=obj,
                                      x=s[:, 0], y=s[:, 1], vx=s[:, 2], vy=s[:, 3],
                                      x_det=raw[:, 0], y_det=raw[:, 1], score=raw[:, 2], event=event))
        stats[f"{obj}_detected"] = round(len(path) / n_frames, 3)
        stats[f"{obj}_covered"] = round(float(np.isfinite(s[:, 0]).mean()), 3)
    return ClipTracks(pd.concat([rows[o] for o in OBJECTS], ignore_index=True), events, stats)
