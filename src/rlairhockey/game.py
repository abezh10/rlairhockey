"""Game events and phases from a clip's tracks (track.py): goals, the puck picked up, serves, shots,
center-line crossings; and per frame what state play is in.

The puck's track breaks into runs, stretches where it is on the table and seen (gaps up to fill.max are
already bridged, so a run ends where the puck is gone for longer). How a run ends says what happened:
  - goal: the puck's last motion carries it, bouncing off the rails, into a goal mouth within
    game.goal_horizon seconds, with no mallet in the way; or it was last seen in the mouth.
  - pickup: the puck was slow when it went (in a hand: a timeout, or retrieving it after it stopped).
  - cut: the clip ends there.
  - lost: anything else (a fast puck the detector lost; unknown).
How a run starts: placed (the puck appears slow: put down for a serve) or found (it appears moving).
A serve is the first mallet strike after a placement. A shot is a strike whose path, with up to two rail
bounces, reaches the other player's goal mouth; its outcome is the next thing that happens to the puck
(a goal, a save by the other mallet, or a miss). A cross is the puck crossing the center line in play.

Phases, per frame: live (from a serve, or a run that starts moving, to the run's end), setup (from a
placement to its serve), dead (after a goal or pickup until the puck is put down again), unseen (the
puck lost, or the clip starting without it).

Sides are as the clip's camera sees the table (-x = left); a goal in the right goal is a point for the
left player (mallet_l).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import Table

COLUMNS = ["t", "frame", "kind", "side", "by", "x", "y", "detail"]


def travel(p: np.ndarray, v: np.ndarray, horizon: float, table: Table, cfg: dict, bounces: int = 2,
           step: float = 0.004) -> tuple[np.ndarray, np.ndarray, str | None, float]:
    """Slide the puck from p at velocity v for up to horizon seconds, bouncing off the rails and the end
    walls beside the goals (restitution game.restitution on the normal speed, up to `bounces` times).
    Returns the times and points along the way, which goal it enters ("left"/"right", or None), and when."""
    r, e = cfg["impacts"]["puck_r"], cfg["game"]["restitution"]
    L, W, g = table.length / 2, table.width / 2, (table.goal_width or 0) / 2
    p, v = np.array(p, float), np.array(v, float)
    ts, ps, n = [0.0], [p.copy()], 0
    for k in range(1, int(horizon / step) + 1):
        p = p + v * step
        if abs(p[0]) > L - r and abs(p[1]) < g:             # the center over the end line inside the mouth
            if abs(p[0]) >= L:
                return np.array(ts + [k * step]), np.array(ps + [p]), "right" if p[0] > 0 else "left", k * step
        elif abs(p[0]) > L - r:
            if n == bounces:
                break
            p[0], v[0], n = np.sign(p[0]) * (L - r), -e * v[0], n + 1
        if abs(p[1]) > W - r:
            if n == bounces:
                break
            p[1], v[1], n = np.sign(p[1]) * (W - r), -e * v[1], n + 1
        ts.append(k * step)
        ps.append(p.copy())
    return np.array(ts), np.array(ps), None, np.nan


def blocked(ts: np.ndarray, ps: np.ndarray, t0: float, tracks: dict, fps: float, cfg: dict) -> str | None:
    """The mallet the puck's path from time t0 (s) runs well into (game.block_overlap past touching), if
    any, by the mallets' tracks then."""
    reach = cfg["impacts"]["puck_r"] + cfg["impacts"]["mallet_r"] - cfg["game"]["block_overlap"]
    for o in ("mallet_l", "mallet_r"):
        m = tracks[o]
        f = np.clip(np.round((t0 + ts) * fps).astype(int), 0, len(m) - 1)
        d = np.hypot(*(m[f, :2] - ps).T)
        if (d[1:] < reach).any():
            return o
    return None


def game_events(tracks: pd.DataFrame, impacts: pd.DataFrame, fps: float, table: Table,
                cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """The clip's game events (columns COLUMNS: time, first frame at or after it, kind, side of the table,
    the mallet responsible, where, and a detail) and its phase per frame."""
    c = cfg["game"]
    T = {o: t.sort_values("frame")[["x", "y", "vx", "vy"]].to_numpy(float) for o, t in tracks.groupby("obj")}
    puck = T["puck"]
    n = len(puck)
    L, g = table.length / 2, (table.goal_width or 0) / 2
    speed = np.hypot(puck[:, 2], puck[:, 3])
    seen = np.isfinite(puck[:, 0])
    idx = np.flatnonzero(seen)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(idx) else []
    edge = int(round(c["edge"] * fps))
    w = max(1, int(round(c["slow_window"] * fps)))
    events = []
    add = lambda f, kind, side="", by="", xy=(np.nan, np.nan), detail="", t=None: events.append(
        dict(t=f / fps if t is None else t, frame=int(f), kind=kind, side=side, by=by, x=float(xy[0]), y=float(xy[1]),
             detail=detail))
    phase = np.full(n, "unseen", object)
    strikes = impacts[impacts.kind.isin(["mallet_l", "mallet_r"])] if len(impacts) else impacts
    ended = ""                                    # how the previous run ended
    for i, run in enumerate(runs):
        a, b = run[0], run[-1]
        # how it starts: put down for a serve (slow, or back after a goal or pickup: carried in a hand), or found
        start_live = a
        if np.nanmedian(speed[a:a + w]) < c["slow_speed"] or ended in ("goal", "pickup"):
            if a > edge:
                add(a, "placed", "left" if puck[a, 0] < 0 else "right", xy=puck[a, :2])
            first = strikes[(strikes.frame > a) & (strikes.frame <= b)] if len(strikes) else strikes
            if len(first):
                s = first.iloc[0]
                add(s.frame, "serve", "left" if s.x < 0 else "right", s.kind, (s.x, s.y), t=s.t)
                phase[a:s.frame] = "setup"
                start_live = s.frame
            else:
                phase[a:b + 1] = "setup"
                start_live = b + 1
        elif a > edge:
            add(a, "found", xy=puck[a, :2])
        phase[start_live:b + 1] = "live"
        # how it ends: a goal, if the puck's last motion takes it into a mouth unblocked
        p, v = puck[b, :2], puck[b, 2:]
        ts, ps, goal, tg = travel(p, v, c["goal_horizon"], table, cfg)
        if goal is None and abs(p[0]) > L - c["mouth_depth"] and abs(p[1]) < g:
            goal, tg, ts, ps = ("right" if p[0] > 0 else "left"), 0.0, ts[:1], ps[:1]
        by = blocked(ts, ps, b / fps, T, fps, cfg) if goal and c["block"] else None
        if goal and by is None and (speed[b] >= c["goal_min_speed"] or tg == 0.0):
            add(b, "goal", goal, "mallet_l" if goal == "right" else "mallet_r", ps[-1], f"{tg:.2f} s after last seen",
                t=b / fps + tg)
            phase[b + 1:], ended = "dead", "goal"
        elif b >= n - 1 - edge:
            add(b, "cut", xy=p)
            ended = "cut"
        elif np.nanmedian(speed[max(a, b - w + 1):b + 1]) < c["slow_speed"]:
            add(b, "pickup", "left" if p[0] < 0 else "right", xy=p)
            phase[b + 1:], ended = "dead", "pickup"
        else:
            add(b, "lost", xy=p, detail=f"{speed[b]:.1f} m/s")
            phase[b + 1:], ended = "unseen", "lost"
        # center-line crossings in play
        live = run[phase[run] == "live"]
        sx = np.sign(puck[live, 0])
        for k in np.flatnonzero(sx[1:] * sx[:-1] < 0):
            f = live[k + 1]
            add(f, "cross", "right" if sx[k + 1] > 0 else "left", xy=puck[f, :2])
    # shots: strikes whose path reaches the other goal's mouth; outcome = what happens next
    ev = pd.DataFrame(events, columns=COLUMNS)
    for s in strikes.itertuples() if len(strikes) else []:
        vout = np.array([s.vx_out, s.vy_out], float)
        if not getattr(s, "seen", True) or np.hypot(*vout) < c["shot_speed"]:
            continue
        ts, ps, goal, tg = travel(np.array([s.x, s.y], float), vout, c["shot_horizon"], table, cfg)
        target = "right" if s.kind == "mallet_l" else "left"
        if goal != target:
            continue
        nxt_goal = ev[(ev.kind == "goal") & (ev.t > s.t) & (ev.t <= s.t + c["shot_horizon"] + 0.5)]
        nxt_strike = strikes[(strikes.t > s.t + 0.02) & (strikes.t <= s.t + tg + 0.3)]
        if len(nxt_goal) and (not len(nxt_strike) or nxt_goal.t.iloc[0] < nxt_strike.t.iloc[0]):
            outcome = "goal" if nxt_goal.side.iloc[0] == target else "miss"
        elif len(nxt_strike):
            outcome = "save" if nxt_strike.kind.iloc[0] != s.kind else "miss"
        else:
            outcome = "miss"
        events.append(dict(t=float(s.t), frame=int(s.frame), kind="shot", side=target, by=s.kind, x=float(s.x),
                           y=float(s.y), detail=outcome))
    ev = pd.DataFrame(events, columns=COLUMNS).sort_values(["t", "kind"], ignore_index=True)
    return ev, phase
