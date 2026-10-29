"""Per-frame detections -> one puck track and one mallet track per table half, in table coordinates.

For each clip:
  0. Drop detections on repeated frames (find_repeats.py): a frame shown twice is no new look at the table,
     and keeping it makes the puck stop for a frame and then jump. In a clip made from a lower frame rate
     by showing frames two or three times (pulldown), each picture is also given the time it was really
     taken (retime()).
  1. Project each box center onto the table: its viewing ray meets the plane at that class's box-center
     height (configs/track.yaml), so tall mallets don't land too far from the camera. Then apply the
     framing's wall correction (fit_walls.py), which moves the calibration's rails to where the puck
     actually bounces.
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
     Gaps too long for the smoother (an arm over the puck, a fast puck missed for a few frames) are
     bridged up to fill.max seconds by the likeliest straight-line path: no corner, one wall bounce, or
     one impact (bridge()). Those frames are flagged `filled`. A piece no mallet comes near is free flight
     on an air cushion, nearly straight and steady, so it gets a much smaller allowance for acceleration.
  6. Keep every frame physically possible (constrain()): centers at least a radius inside the rails, and
     the puck and a mallet never overlapping.

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

from .geometry import ROOT, Camera, Table, apply_h

OBJECTS = ("puck", "mallet_l", "mallet_r")


def load_config(path: str | Path = ROOT / "configs/track.yaml") -> dict:
    return yaml.safe_load(open(path))


def to_table(det: pd.DataFrame, cam: Camera, cfg: dict, walls: np.ndarray | None = None) -> pd.DataFrame:
    """Add table x, y for each box center, intersecting its ray with the plane at the class's height, then
    applying the framing's wall correction (a homography on the table plane; fit_walls.py), if any."""
    det = det.copy()
    uv = np.c_[(det.x0 + det.x1) / 2, (det.y0 + det.y1) / 2]
    xy = np.full((len(det), 2), np.nan)
    for cls in ("puck", "mallet"):
        m = (det.cls == cls).to_numpy()
        if m.any():
            xy[m] = cam.to_table(uv[m], z=cfg[cls]["z"])[:, :2]
    if walls is not None:
        xy = apply_h(walls, xy)
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


def retime(changes: np.ndarray, cfg: dict, xy: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """When each frame's picture was taken, in frames, from how much each frame changed (find_repeats.py);
    NaN for a repeat. Also a mask of pictures that may have been taken a frame earlier than that (the first
    after a repeat that couldn't be resolved). Two kinds of clip show repeats:
    - Pulldown (most frames repeated): a lower-rate source's pictures shown two or more times each (the 2024
      final is 24 fps shown at 60, with no fixed pattern). A picture's first frame is up to a frame or two
      late, but the pictures are evenly spaced in time, so each one's time comes from a straight-line fit
      of first frame against picture number around it.
    - Dropped and doubled frames (a few repeated; the 2023 final): after a repeat, pictures run a frame late
      until one is skipped, where everything moves two steps at once. With xy (the puck's position per
      frame, NaN where not seen) the skip is found where the puck, moving steadily, takes a double step
      within repeats.catch_up frames, and the pictures between are moved back a frame. Otherwise (the
      puck too slow to tell, or a strike right at the repeat) a picture stays at its frame and the first
      one after the repeat is flagged unsure."""
    c = cfg["repeats"]
    rep = (changes >= 0) & (changes <= c["px"])
    t = np.where(rep, np.nan, np.arange(len(changes), dtype=float))
    unsure = np.zeros(len(changes), bool)
    new = np.flatnonzero(~rep)
    w = c["window"]
    if rep.mean() >= c["pulldown"]:
        if len(new) > 2 * w:
            k = np.arange(len(new))
            for i in k:
                lo, hi = max(0, min(i - w, len(new) - 2 * w - 1)), max(2 * w + 1, min(i + w + 1, len(new)))
                b, a = np.polyfit(k[lo:hi], new[lo:hi], 1)
                t[new[i]] = a + b * i
        return t, unsure
    gap = np.diff(new)
    unsure[new[1:][gap >= 2]] = True
    if xy is None or len(new) < 8:
        return t, unsure
    step = np.hypot(*np.diff(xy[new], axis=0).T)          # step[j]: new[j] -> new[j + 1]
    steady = lambda r: 0.6 <= r <= 1.4
    for j in np.flatnonzero(gap == 2):                    # one repeat between new[j] and new[j + 1]
        d = np.median(step[max(0, j - 3):j]) if j >= 3 else np.nan
        if not d > c["min_step"] or not steady(step[j] / d):
            continue                                      # too slow to tell, or not a stall
        for k in range(j + 1, len(step) - 1):
            if gap[k] != 1 or new[k] - new[j] > c["catch_up"] or not np.isfinite(step[k]):
                break
            if 1.6 <= step[k] / d <= 2.5 and steady(step[k + 1] / d):
                t[new[j + 1]:new[k] + 1] -= 1             # those pictures were a frame late
                unsure[new[j + 1]] = False
                break
            if not steady(step[k] / d):
                break
    return t, unsure


def picture_times(det: pd.DataFrame, changes: np.ndarray, n_frames: int, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """retime() for a clip, given its detections in table coordinates (the puck's steps find dropped pictures)."""
    pk = det[(det.cls == "puck") & (det.frame < n_frames)].sort_values("score").groupby("frame").tail(1)
    xy = np.full((n_frames, 2), np.nan)
    xy[pk.frame] = pk[["x", "y"]].to_numpy()
    return retime(np.r_[changes, np.full(max(0, n_frames - len(changes)), 10 ** 6)][:n_frames], cfg, xy)


def smooth(frames: np.ndarray, xy: np.ndarray, n_frames: int, fps: float, meas_sigma: float,
           accel_sigma: float, max_gap: float, offsets: np.ndarray | None = None) -> np.ndarray:
    """Constant-velocity Kalman filter + RTS smoother over all frames from the first to the last
    measurement. offsets: when each measurement was taken relative to its frame, in s (retime()). Returns
    (n_frames, 4) x, y, vx, vy; NaN outside, and inside gaps longer than max_gap."""
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
    off = np.zeros(n_frames)
    if offsets is not None:
        off[frames] = offsets
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
            H[0, 2] = H[1, 3] = off[t0 + i]           # the position at the measurement's own time
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


def _suffix_sse(t: np.ndarray, xy: np.ndarray, deg: int = 1) -> np.ndarray:
    """Residual sum of squares of a polynomial fit of xy(t) over every suffix t[i:] (inf under deg + 1 points)."""
    s, d = t - t[-1], deg + 1
    pw = np.stack([s ** k for k in range(2 * d - 1)], 1)
    P = pw[::-1].cumsum(0)[::-1]                                  # sums of s^k over each suffix
    B = (pw[:, :d, None] * xy[:, None, :])[::-1].cumsum(0)[::-1]  # sums of s^k * (x, y)
    Q = (xy ** 2).sum(1)[::-1].cumsum()[::-1]
    sse = np.full(len(t), np.inf)
    ok = np.arange(len(t)) <= len(t) - d
    if ok.any():
        M = np.stack([P[ok, k:k + d] for k in range(d)], 1)       # normal matrices
        c = np.linalg.solve(M, B[ok])
        sse[ok] = np.maximum(Q[ok] - (c * B[ok]).sum((1, 2)), 0)
    return sse


def pieces(t: np.ndarray, xy: np.ndarray, cfg: dict) -> list[tuple[int, int]]:
    """Split detections into pieces of steady motion, as [start, end) index pairs: the partition that
    minimizes each piece's straight-line residual (in units of the detection noise) plus a cost per piece.
    (A puck between impacts slides straight at a nearly constant speed.)
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
    m, best = cfg["min_len"], None
    for k in range(max(a[0] + 3, b[0] - 2), min(b[1] - 3, b[0] + 2) + 1):
        lo = max(a[0], min(int(np.searchsorted(t, t[k] - cfg["fit"])), k - m))     # fit seconds either side,
        hi = min(b[1], max(int(np.searchsorted(t, t[k] + cfg["fit"])), k + m))     # at least min_len detections
        (la, ra), (lb, rb) = (_line(t[i] - t[k], xy[i]) for i in (np.arange(lo, k), np.arange(k, hi)))
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


def hidden_bounce(t: np.ndarray, xy: np.ndarray, alt: np.ndarray, side: str, table: Table, cfg: dict) -> dict | None:
    """A wall bounce among a few detections, too few either side of it for pieces() to split (a fast shot
    off a rail is often seen only once or twice before and after). The model: a straight line out to the
    wall and back, the speed into the wall scaled by a restitution e on the way out, which is a straight
    line once the detections after the bounce are mirrored across the wall and their distance from it
    divided by e. The best wall, split and e must beat one straight line by impacts.hidden_penalty (in
    units of the detection noise squared). side "end"/"start" keeps the split within min_len of the last /
    first detection (pieces() would have found one further in); "any" allows it anywhere. Detections
    with alt set may have been taken that much earlier (retime()): each takes whichever time fits.
    Returns None, or the split (index of the first detection after the bounce), the bounce as an impact,
    the chosen times, and the line before it (p0 at t = 0, v)."""
    ic = cfg["impacts"]
    n, m, r = len(t), ic["min_len"], ic["puck_r"]
    if n < 3:
        return None
    walls = [(1, sgn, table.width / 2 - r, "rail") for sgn in (1, -1)] + \
            [(0, sgn, table.length / 2 - r, "end") for sgn in (1, -1)]
    flex = np.flatnonzero(alt != 0)[:3]
    variants = []
    for bits in range(2 ** len(flex)):
        tt = t.copy()
        for q, i in enumerate(flex):
            if bits >> q & 1:
                tt[i] += alt[i]
        if np.all(np.diff(tt) > 0):
            variants.append(tt)
    A = lambda tt: np.c_[np.ones(n), tt]
    sse0 = min(_line(tt, xy)[1] for tt in variants)
    lo, hi = {"end": (max(1, n - m + 1), n), "start": (1, min(n, m)), "any": (1, n)}[side]
    best = None
    for tt in variants:
        for ax, sgn, lim, kind in walls:
            w = sgn * lim
            for j in range(lo, hi):
                for e in (0.6, 0.75, 0.9, 1.0):
                    u = xy.copy()
                    u[j:, ax] = w + (w - xy[j:, ax]) / e      # mirrored, distance from the wall / e
                    c = np.linalg.lstsq(A(tt), u, rcond=None)[0]
                    res = A(tt) @ c - u
                    res[j:, ax] *= e                           # back to the real distance
                    sse = float((res ** 2).sum())
                    if best is not None and sse >= best[0]:
                        continue
                    p0, v = c
                    if sgn * v[ax] <= 0:
                        continue                               # not moving into that wall
                    tc = (w - p0[ax]) / v[ax]
                    if not tt[j - 1] - 0.5 * (tt[j] - tt[j - 1]) <= tc <= tt[j] + 0.5 * (tt[j] - tt[j - 1]):
                        continue
                    hit = p0 + v * tc
                    if kind == "end" and table.goal_width and abs(hit[1]) < table.goal_width / 2:
                        continue                               # that's the goal mouth
                    best = (sse, j, tt, p0, v, tc, hit, e, ax, kind)
    if best is None or best[0] / ic["sigma"] ** 2 + ic["hidden_penalty"] >= sse0 / ic["sigma"] ** 2:
        return None
    sse, j, tt, p0, v, tc, hit, e, ax, kind = best
    v_out = v.copy()
    v_out[ax] *= -e
    ev = dict(t=tc, x=hit[0], y=hit[1], vx_in=v[0], vy_in=v[1], vx_out=v_out[0], vy_out=v_out[1], kind=kind,
              offset=0.0, restitution=e)
    return dict(split=j, event=ev, times=tt, p0=p0, v=v)


def bridge(pa: np.ndarray, va: np.ndarray, pb: np.ndarray, vb: np.ndarray, T: float, table: Table,
           cfg: dict) -> tuple[list, dict | None]:
    """The puck's likeliest path across a gap of T seconds from (pa, va) to (pb, vb): a straight line, one
    bounce off a rail or end wall (a straight line once pb is mirrored across that wall), or two straight
    legs meeting where the lines in and out come closest: on the wall if that's next to one and the puck
    was moving into it and away (a bounce that lost speed), else an impact. Each is scored by how far its
    velocities in and out are from va and vb, plus a cost for the corner it adds. Returns the path as
    (time in the gap, point) knots and the bounce or impact on it (t relative to the gap start), if any."""
    c = cfg["fill"]
    cost = lambda v_in, v_out: float(np.linalg.norm(np.r_[v_in - va, v_out - vb]))
    v = (pb - pa) / T
    best = (cost(v, v), [(0.0, pa), (T, pb)], None)
    r = cfg["impacts"]["puck_r"]
    for kind, ax, half in (("rail", 1, table.width / 2), ("end", 0, table.length / 2)):
        for s in (1, -1):
            wall = s * (half - r)
            pm = pb.copy()
            pm[ax] = 2 * wall - pb[ax]                            # pb mirrored across the wall
            v_in = (pm - pa) / T
            if s * (pm[ax] - wall) <= 0 or s * (wall - pa[ax]) < 0 or s * v_in[ax] <= 0:
                continue                                          # the straight line must cross the wall
            tc = (wall - pa[ax]) / v_in[ax]
            hit = pa + v_in * tc
            if kind == "end" and table.goal_width and abs(hit[1]) < table.goal_width / 2 - r:
                continue                                          # that's the goal mouth, not a wall
            v_out = v_in.copy()
            v_out[ax] *= -1
            k = cost(v_in, v_out) + c["bounce_cost"]
            if k < best[0]:
                into, away = s * va[ax], -s * vb[ax]
                ev = dict(t=tc, x=hit[0], y=hit[1], vx_in=v_in[0], vy_in=v_in[1], vx_out=v_out[0], vy_out=v_out[1],
                          kind=kind, offset=0.0, restitution=away / into if into > 0 and away > 0 else np.nan)
                best = (k, [(0.0, pa), (tc, hit), (T, pb)], ev)
    dv = va - vb                                                  # the in and out lines' closest approach
    d0 = pa - pb + vb * T
    s = float(np.clip(-d0 @ dv / max(dv @ dv, 1e-9), 0.5 * T / 8, T - 0.5 * T / 8))
    hit = (pa + va * s + pb + vb * (s - T)) / 2
    hit = np.clip(hit, [-table.length / 2 + r, -table.width / 2 + r], [table.length / 2 - r, table.width / 2 - r])
    kind, extra, rest = "other", c["impact_cost"], np.nan
    for wall_kind, ax, half in (("rail", 1, table.width / 2), ("end", 0, table.length / 2)):
        sg, lim = np.sign(hit[ax]), half - r                     # a corner by a wall, moving into it and away:
        into, away = sg * va[ax], -sg * vb[ax]                    # a bounce that lost speed, so put it on the wall
        mouth = wall_kind == "end" and table.goal_width and abs(hit[1]) < table.goal_width / 2 - r
        if abs(hit[ax]) > lim - cfg["impacts"]["rail_tol"] and into > 0 and away > 0 and not mouth:
            hit[ax] = sg * lim
            kind, extra, rest = wall_kind, c["bounce_cost"], away / into
            break
    v_in, v_out = (hit - pa) / s, (pb - hit) / (T - s)
    k = cost(v_in, v_out) + extra
    if k < best[0]:
        ev = dict(t=s, x=hit[0], y=hit[1], vx_in=v_in[0], vy_in=v_in[1], vx_out=v_out[0], vy_out=v_out[1],
                  kind=kind, offset=0.0 if kind != "other" else np.nan, restitution=rest)
        best = (k, [(0.0, pa), (s, hit), (T, pb)], ev)
    return best[1], best[2]


def smooth_puck(frames: np.ndarray, xy: np.ndarray, n_frames: int, fps: float, mallets: dict, table: Table,
                cfg: dict, offsets: np.ndarray | None = None,
                alt: np.ndarray | None = None) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """smooth() for the puck, with its velocity free to jump at impacts (see impact()), and gaps up to
    fill.max seconds bridged (see bridge()). Returns the (n_frames, 4) track; the impacts (t, frame = first
    frame after it, kind, contact x, y, velocities in and out, for bounces the contact's distance past the
    rail line and the normal restitution, and seen = False for one inferred inside a gap); a mask of the
    frames filled in by a bridge; and each detection's time (s), with any alt it took. offsets: as for
    smooth(); alt: as for hidden_bounce()."""
    sm, ic = cfg["smooth"], cfg["impacts"]
    reach = ic["puck_r"] + ic["mallet_r"] + ic["mallet_tol"]
    edge = 0.1 * fps

    def free(a: int, b: int) -> bool:             # no mallet touching it for long, away from its ends (impacts)
        f, p = frames[a:b], xy[a:b]
        inner = (f > f[0] + edge) & (f < f[-1] - edge)
        touch = np.zeros(inner.sum(), bool)
        for m in mallets.values():
            touch |= np.hypot(*(m[f[inner], :2] - p[inner]).T) <= ic["puck_r"] + ic["mallet_r"] + ic["touch_tol"]
        return touch.sum() * (f[-1] - f[0]) / fps / max(len(f) - 1, 1) < ic["touch_min"]

    out = np.full((n_frames, 4), np.nan)
    filled = np.zeros(n_frames, bool)
    events = []
    offsets = np.zeros(len(frames)) if offsets is None else offsets.astype(float)
    alt = np.zeros(len(frames)) if alt is None else alt.astype(float)
    t = frames / fps + offsets
    ends = []                                     # (first, last) frame of each run
    for run in np.split(np.arange(len(frames)), np.flatnonzero(np.diff(frames) - 1 > sm["max_gap"] * fps) + 1):
        if not len(run):
            continue
        ends.append((frames[run[0]], frames[run[-1]]))
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
        lines, m, i = {}, ic["min_len"], 0       # lines: a piece too short to smooth, as (time, point, velocity)
        while i < len(joined):                    # bounces hidden among too few detections (hidden_bounce())
            a, b = joined[i]
            for side in (["any"] if b - a < 2 * m else ["start", "end"]):
                w = {"any": slice(a, b), "start": slice(a, a + 2 * m - 1), "end": slice(b - 2 * m + 1, b)}[side]
                hb = hidden_bounce(t[w], xy[w], alt[w], side, table, cfg)
                if hb is None:
                    continue
                t[w], alt[w] = hb["times"], 0.0
                offsets[w] = t[w] - frames[w] / fps
                k, ev = w.start + hb["split"], hb["event"]
                joined[i:i + 1] = [(a, k), (k, b)]
                cuts.insert(i, ev)
                if i and side != "end":                       # the impact before saw the bounce as part of its way out
                    cuts[i - 1].update(vx_out=ev["vx_in"], vy_out=ev["vy_in"])
                if i + 1 < len(cuts) and side != "start":
                    cuts[i + 1].update(vx_in=ev["vx_out"], vy_in=ev["vy_out"])
                lines[(a, k)] = (0.0, hb["p0"], hb["v"])
                lines[(k, b)] = (ev["t"], np.array([ev["x"], ev["y"]]), np.array([ev["vx_out"], ev["vy_out"]]))
                break
            else:
                i += 1
        for i, (a, b) in enumerate(joined):
            if (a, b) in lines and b - a < m:
                t0, p0, v = lines[(a, b)]
                f = np.arange(frames[a], frames[b - 1] + 1)
                out[f] = np.c_[p0 + np.outer(f / fps - t0, v), np.tile(v, (len(f), 1))]
            else:
                acc = sm["accel_sigma"]["puck_free" if free(a, b) else "puck"]
                s = smooth(frames[a:b], xy[a:b], n_frames, fps, sm["meas_sigma"]["puck"], acc, sm["max_gap"], offsets[a:b])
                out[frames[a]:frames[b - 1] + 1] = s[frames[a]:frames[b - 1] + 1]
            if i:                                 # between the pieces: straight to the impact and on from it
                ev, fa, fb = cuts[i - 1], frames[joined[i - 1][1] - 1], frames[a]
                tc, c = np.clip(ev["t"], fa / fps, fb / fps), np.array([ev["x"], ev["y"]])
                for f in range(fa + 1, fb):
                    (t0, p0), (t1, p1) = ((fa / fps, out[fa, :2]), (tc, c)) if f / fps < tc else \
                        ((tc, c), (fb / fps, out[fb, :2]))
                    v = (p1 - p0) / max(t1 - t0, 1e-9)
                    out[f] = np.r_[p0 + v * (f / fps - t0), v]
        events += [dict(e, seen=True) for e in cuts]
    for (_, fa), (fb, _) in zip(ends[:-1], ends[1:]):
        if (fb - fa - 1) / fps > cfg["fill"]["max"]:
            continue
        T = (fb - fa) / fps
        knots, ev = bridge(out[fa, :2], out[fa, 2:], out[fb, :2], out[fb, 2:], T, table, cfg)
        ks, kp = np.array([k[0] for k in knots]), np.array([k[1] for k in knots])
        for f in range(fa + 1, fb):
            s = (f - fa) / fps
            i = min(int(np.searchsorted(ks, s)) - 1, len(ks) - 2)
            v = (kp[i + 1] - kp[i]) / (ks[i + 1] - ks[i])
            out[f] = np.r_[kp[i] + v * (s - ks[i]), v]
        filled[fa + 1:fb] = True
        if ev is not None:
            ev["t"] += fa / fps
            if ev["kind"] == "other":                  # a mallet there at the time makes it a strike
                f = int(round(ev["t"] * fps))
                near = {o: np.hypot(*(m[f, :2] - [ev["x"], ev["y"]])) for o, m in mallets.items() if np.isfinite(m[f, 0])}
                near = {o: d for o, d in near.items() if d <= reach}
                if near:
                    ev["kind"] = min(near, key=near.get)
            events.append(dict(ev, seen=False))
    ev = pd.DataFrame(events, columns=["t", "x", "y", "vx_in", "vy_in", "vx_out", "vy_out", "kind", "offset",
                                       "restitution", "seen"]).sort_values("t", ignore_index=True)
    ev.insert(1, "frame", np.ceil(ev.t * fps - 1e-6).astype(int))
    return out, ev, filled, t


def constrain(tracks: dict[str, np.ndarray], events: pd.DataFrame, table: Table, cfg: dict) -> dict:
    """Make every frame physically possible, a last small correction for calibration and detection error:
    each object's center at least its radius inside the rails (the puck may go on into a goal mouth), and
    the puck never overlapping a mallet (the two are pushed apart along the line between them, each by a
    share of the overlap in proportion to its measurement variance, so the less certain mallet moves more).
    Bounces are put on their wall. Edits tracks (obj -> (n_frames, 4)) and events in place; returns how
    many frames each rule moved something in."""
    ic = cfg["impacts"]
    L, W, g = table.length / 2, table.width / 2, (table.goal_width or 0) / 2
    rad = {"puck": ic["puck_r"], "mallet_l": ic["mallet_r"], "mallet_r": ic["mallet_r"]}
    var = {o: cfg["smooth"]["meas_sigma"]["puck" if o == "puck" else "mallet"] ** 2 for o in rad}
    n = len(tracks["puck"])
    moved = dict(off_wall=np.zeros(n, bool), apart=np.zeros(n, bool))

    def inside(o: str, p: np.ndarray) -> np.ndarray:     # p clamped to where o's center can be
        y = np.clip(p[:, 1], -(W - rad[o]), W - rad[o])
        lx = np.where((o == "puck") & (np.abs(y) < g), L + cfg["puck"]["goal_depth"], L - rad[o])
        return np.c_[np.clip(p[:, 0], -lx, lx), y]

    def clamp() -> None:
        for o, p in tracks.items():
            q = inside(o, p[:, :2])
            moved["off_wall"] |= np.any(np.abs(q - p[:, :2]) > 1e-9, axis=1)
            p[:, :2] = q

    P = tracks["puck"]
    for _ in range(3):                            # a puck pinned to a rail by a mallet takes a few rounds
        clamp()
        for o in ("mallet_l", "mallet_r"):
            M = tracks[o]
            d = P[:, :2] - M[:, :2]
            dist = np.hypot(*d.T)
            over = np.nan_to_num(rad["puck"] + rad[o] - dist, nan=0.0)
            m = over > 1e-6
            u = d[m] / np.maximum(dist[m], 1e-9)[:, None]
            share = var["puck"] / (var["puck"] + var[o])
            P[m, :2] += (share * over[m])[:, None] * u
            M[m, :2] -= ((1 - share) * over[m])[:, None] * u
            moved["apart"] |= m
    clamp()
    if len(events):
        events[["x", "y"]] = inside("puck", events[["x", "y"]].to_numpy(float))
        for kind, c, half in (("rail", "y", W), ("end", "x", L)):
            m = events.kind == kind
            events.loc[m, c] = np.sign(events.loc[m, c]) * (half - ic["puck_r"])
    return {k: int(v.sum()) for k, v in moved.items()}


@dataclass
class ClipTracks:
    tracks: pd.DataFrame          # frame, obj, x, y, vx, vy, x_det, y_det, score (NaN when not detected), event, filled,
                                  # picture (when the frame's picture was taken, in frames; retime())
    events: pd.DataFrame          # puck impacts, see smooth_puck()
    stats: dict


def track_clip(det: pd.DataFrame, n_frames: int, fps: float, cam: Camera, table: Table,
               cfg: dict | None = None, changes: np.ndarray | None = None,
               walls: np.ndarray | None = None) -> ClipTracks:
    """changes: per frame, the pixels changed since the frame before (find_repeats.py), to drop repeats and
    retime pulldown. walls: the framing's wall correction (fit_walls.py)."""
    cfg = cfg or load_config()
    det = to_table(det[det.score >= cfg["min_score"]], cam, cfg, walls).assign(dt=0.0, alt=0.0)
    det = det.assign(shown=det.frame)
    picture = np.arange(n_frames, dtype=float)
    stats = {}
    if changes is not None:
        when, unsure = picture_times(det, changes, n_frames, cfg)
        f = det.frame.clip(0, n_frames - 1).to_numpy()
        tf, maybe = when[f], unsure[f]
        keep = np.isfinite(tf)
        det, tf, maybe = det[keep], tf[keep], maybe[keep]
        det = det.assign(shown=f[keep], frame=np.clip(np.round(tf), 0, n_frames - 1).astype(int))
        det = det.assign(dt=(tf - det.frame.to_numpy()) / fps, alt=np.where(maybe, -1 / fps, 0.0))
        picture = np.array(pd.Series(when).ffill().fillna(0), float)
        stats["repeats"] = int(np.isnan(when).sum())
        stats["retimed"] = int((np.abs(when - np.arange(n_frames)) > 0.5).sum())
    surf = on_surface(det, table, cfg)
    stats |= dict(boxes=len(det), off_surface=int((~surf).sum()))
    det = det[surf]
    half = cfg["mallet"]["center_overlap"]
    targets = {"puck": det[det.cls == "puck"],
               "mallet_l": det[(det.cls == "mallet") & (det.x < half)],
               "mallet_r": det[(det.cls == "mallet") & (det.x > -half)]}
    paths, smoothed = {}, {}
    filled = np.zeros(n_frames, bool)
    for obj in ("mallet_l", "mallet_r", "puck"):  # mallets first: the puck's impacts look for them
        d = targets[obj].reset_index(drop=True)
        pk = parked(d, n_frames, cfg)
        stats[f"{obj}_parked"] = int(pk.sum())
        d = d[~pk].reset_index(drop=True)
        cls = "puck" if obj == "puck" else "mallet"
        paths[obj] = path = d.iloc[best_path(d, n_frames, fps, cfg[cls]["vmax"], cfg)] if len(d) else d
        fr, xy, dt = path.frame.to_numpy(), path[["x", "y"]].to_numpy(), path.dt.to_numpy()
        if obj == "puck":
            smoothed[obj], events, filled, tdet = smooth_puck(fr, xy, n_frames, fps, smoothed, table, cfg, dt,
                                                              path.alt.to_numpy())
            shown = path.shown.to_numpy()                 # a picture whose time hidden_bounce() moved
            moved = np.abs(tdet * fps - (fr + dt * fps)) > 0.5
            picture[shown[moved]] = tdet[moved] * fps
            for k, v in events.kind.value_counts().items():
                stats[f"puck_{k}"] = int(v)
        else:
            sm = cfg["smooth"]
            smoothed[obj] = smooth(fr, xy, n_frames, fps, sm["meas_sigma"][cls], sm["accel_sigma"][cls], sm["max_gap"], dt)
    stats |= {f"constrained_{k}": v for k, v in constrain(smoothed, events, table, cfg).items()}
    rows = []
    for obj in OBJECTS:
        s, path = smoothed[obj], paths[obj]
        event = np.full(n_frames, "", object)
        if obj == "puck":
            event[events.frame.clip(0, n_frames - 1)] = events.kind
        raw = np.full((n_frames, 3), np.nan)
        raw[path.frame.to_numpy()] = path[["x", "y", "score"]].to_numpy()
        rows.append(pd.DataFrame(dict(frame=np.arange(n_frames, dtype=np.int32), obj=obj,
                                      x=s[:, 0], y=s[:, 1], vx=s[:, 2], vy=s[:, 3], x_det=raw[:, 0], y_det=raw[:, 1],
                                      score=raw[:, 2], event=event, filled=filled if obj == "puck" else False,
                                      picture=picture)))
        stats[f"{obj}_detected"] = round(len(path) / n_frames, 3)
        stats[f"{obj}_covered"] = round(float(np.isfinite(s[:, 0]).mean()), 3)
    return ClipTracks(pd.concat(rows, ignore_index=True), events, stats)
