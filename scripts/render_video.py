"""Render a clip as a video of four views in a 2 x 2 grid.

Top left, the broadcast frame: every detection is a thin box (colored if the tracker used it, gray if not)
and each tracked object's position is drawn back into the image as a ring.
Top right, the top-down table (the framing's median frame, warped) with the puck and both mallets at true
size and their trails over the last --trail seconds. The puck's trail runs through each impact's contact
point, so a bounce between two frames still touches the rail. Each impact (a rail or end-wall bounce, a
mallet strike, or an unexplained velocity jump) shows for half a second as a ring at the contact point,
labeled with its cause (and for bounces the restitution); "(gap)" marks one inferred inside a gap the
tracker bridged. Bridged stretches of the puck's path are drawn thin, and its rings there thin too.
Bottom left, the same table as a plain diagram: only the puck and mallets where the tracker puts them,
with short trails and the puck's velocity as an arrow (where it will be in 0.1 s).
Bottom right, a live graph of the last --window seconds: each object's position along and across the
table, and the puck's speed with its impacts as ticks; above them a band colored by the phase of play
(game.py) with the game events marked.
Every view shows the moment the footage shows: a repeated frame holds them all still, and in a clip made
by pulldown the picture's own time (track.retime()) is used, not the frame's.
Game events (goals, the puck picked up, serves, shots) are announced across the top of the footage.
Writes outputs/videos/<model>/<clip>.mp4 (H.264, so browsers and Colab can play it).
Usage: python scripts/render_video.py <clip> [<clip> ...] [--model v2] [--height 450] [--seconds 40]
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rlairhockey.geometry import Table, apply_h, load_framing
from rlairhockey.render import STYLE, diagram, path_points, table_lines, table_view
from rlairhockey.track import OBJECTS, load_config, to_table

ROOT = Path(__file__).resolve().parents[1]
COLOR = {"puck": (0, 230, 255), "mallet_l": (255, 140, 0), "mallet_r": (60, 60, 255)}   # BGR
NAME = {"puck": "puck", "mallet_l": "mallet L", "mallet_r": "mallet R"}
RADIUS = {"puck": 0.041, "mallet_l": 0.05, "mallet_r": 0.05}                            # m
IMPACT = {"rail": (80, 255, 80), "end": (80, 255, 80), "mallet_l": COLOR["mallet_l"], "mallet_r": COLOR["mallet_r"],
          "other": (255, 0, 255)}
PLAYER = {"mallet_l": "left", "mallet_r": "right"}
GAME = {"goal": ((0, 140, 255), 2.5, "GOAL  in the {side} goal"), "pickup": ((255, 150, 220), 2.0, "PUCK PICKED UP"),
        "serve": ((255, 255, 0), 1.2, "SERVE  {player} player"), "shot": ((255, 255, 255), 1.2, "SHOT  {player} player: {detail}"),
        "lost": ((160, 160, 160), 1.0, "PUCK LOST"), "placed": ((200, 200, 200), 1.0, "PUCK PUT DOWN")}   # BGR, s shown
PHASE = {"live": (80, 190, 80), "setup": (0, 190, 255), "dead": (110, 110, 110), "unseen": (55, 55, 55)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(img: np.ndarray, s: str, org: tuple, col=(235, 235, 235), scale: float = 0.45) -> None:
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, col, 1, cv2.LINE_AA)


def segments(pts: np.ndarray, ok: np.ndarray, thin: np.ndarray):
    """Runs of consecutive drawable points with the same thickness, as (points, thin) for cv2.polylines."""
    i, n = 0, len(pts)
    while i < n - 1:
        if not (ok[i] and ok[i + 1]):
            i += 1
            continue
        j = i + 1
        while j + 1 < n and ok[j + 1] and thin[j + 1] == thin[i + 1]:
            j += 1
        yield pts[i:j + 1], bool(thin[i + 1])
        i = j


class Graph:
    """The live graph: the last `window` seconds of each object's x (along the table) and y (across), and
    the puck's speed with ticks at its impacts, scrolling so that now is the right edge. Above them, a band
    colored by phase, with goals and pickups as lines through every strip and serves and shots as ticks."""

    def __init__(self, T: dict, events: pd.DataFrame, game: pd.DataFrame, phase: np.ndarray, fps: float,
                 table: Table, size: tuple, window: float):
        self.fps, self.events, self.n_win = fps, events, max(2, int(round(window * fps)))
        self.game, self.phase = game[game.kind.isin(["goal", "pickup", "serve", "shot"])], phase
        w, h = size
        self.x0, self.x1, top, bottom = 52, w - 12, 46, h - 20
        self.band = (31, 38)
        L, W = table.length / 2, table.width / 2
        puck = T.get("puck")
        speed = np.hypot(puck.vx, puck.vy).to_numpy() if puck is not None else np.zeros(1)
        vtop = max(5.0, 5 * np.ceil(np.nanquantile(speed, 0.995) / 5)) if np.isfinite(speed).any() else 5.0
        strips = [("along (m)", -L, L, [-L, 0, L], "x"), ("across (m)", -W, W, [-W, 0, W], "y"),
                  ("puck speed (m/s)", 0, vtop, [0, vtop / 2, vtop], "speed")]
        weights, gap = np.array([1.0, 0.62, 0.8]), 14
        hs = (bottom - top - gap * (len(strips) - 1)) * weights / weights.sum()
        self.base = np.full((h, w, 3), (24, 20, 18), np.uint8)
        self.strips, y = [], top
        for (name, lo, hi, guides, key), sh in zip(strips, hs):
            ya, yb = int(y), int(y + sh)
            v = lambda val, lo=lo, hi=hi, ya=ya, yb=yb: yb - (np.asarray(val, float) - lo) / (hi - lo) * (yb - ya)
            cv2.rectangle(self.base, (self.x0, ya), (self.x1, yb), (60, 60, 60), 1)
            for g in guides:
                gy = int(round(float(v(g))))
                cv2.line(self.base, (self.x0, gy), (self.x1, gy), (52, 52, 52), 1)
                text = f"{g:.0f}" if key == "speed" or g == 0 else f"{g:+.1f}"
                label(self.base, text, (6, int(np.clip(gy + 4, ya + 10, yb - 1))), (150, 150, 150), 0.38)
            label(self.base, name, (self.x0 + 6, ya + 14), (200, 200, 200), 0.42)
            if key == "speed":                                    # what the ticks along its bottom mean
                xs = self.x0 + 150
                for word, col in (("impacts:", (170, 170, 170)), ("bounce", IMPACT["rail"]),
                                  ("strike", (235, 235, 235)), ("other", IMPACT["other"])):
                    label(self.base, word, (xs, ya + 14), col, 0.38)
                    xs += 8 + cv2.getTextSize(word, FONT, 0.38, 1)[0][0]
            series = {}
            for o, t in T.items():
                if key == "speed" and o != "puck":
                    continue
                val = speed if key == "speed" else t[key].to_numpy(float)
                series[o] = (v(np.clip(val, lo, hi)), t["filled"].to_numpy(bool) if o == "puck" else np.zeros(len(t), bool))
            self.strips.append((ya, yb, key, series))
            y += sh + gap
        for s in range(int(window) + 1):                          # time axis: seconds before now
            u = int(round(self.x1 - s * fps * (self.x1 - self.x0) / self.n_win))
            cv2.line(self.base, (u, top), (u, bottom), (44, 44, 44), 1)
            label(self.base, "now" if s == 0 else f"-{s}s", (u - (24 if s == 0 else 12), h - 5), (150, 150, 150), 0.38)
        self.bottom = bottom
        xs = self.x0 + 4
        for o in OBJECTS:
            label(self.base, NAME[o], (xs, 16), COLOR[o], 0.42)
            xs += 12 + cv2.getTextSize(NAME[o], FONT, 0.42, 1)[0][0]
        xs += 14
        for name, col in PHASE.items():
            label(self.base, name, (xs, 16), col, 0.38)
            xs += 8 + cv2.getTextSize(name, FONT, 0.38, 1)[0][0]

    def draw(self, now: float) -> np.ndarray:
        """The graph at frame time `now` (fractional: a picture's own time)."""
        img = self.base.copy()
        k = int(np.floor(now))
        f0 = now - self.n_win
        u = lambda f: self.x0 + (np.asarray(f, float) - f0) * (self.x1 - self.x0) / self.n_win
        a, b = max(0, int(np.ceil(f0))), min(k + 1, len(self.phase))
        if b > a:                                                 # the phase band
            us = np.round(u(np.r_[a:b, b])).astype(int)
            for i, f in enumerate(range(a, b)):
                cv2.rectangle(img, (us[i], self.band[0]), (max(us[i + 1] - 1, us[i]), self.band[1]),
                              PHASE.get(self.phase[f], PHASE["unseen"]), -1)
        for e in self.game[(self.game.t * self.fps > f0) & (self.game.t * self.fps <= now)].itertuples():
            ue = int(round(float(u(e.t * self.fps))))
            col = GAME[e.kind][0]
            if e.kind in ("goal", "pickup"):
                cv2.line(img, (ue, self.band[0] - 4), (ue, self.bottom), col, 2)
            else:
                cv2.line(img, (ue, self.band[0] - 4), (ue, self.band[1] + 4), col, 2)
            label(img, e.kind, (ue + 3, self.band[0] - 3), col, 0.36)
        for ya, yb, key, series in self.strips:
            for o, (val, thin) in series.items():
                a, b = max(0, int(np.ceil(f0))), min(k + 1, len(val))
                if b - a < 2:
                    continue
                pts = np.c_[u(np.arange(a, b)), val[a:b]]
                ok = np.isfinite(pts[:, 1])
                for seg, th in segments(np.round(np.nan_to_num(pts) * 4).astype(np.int32), ok, thin[a:b]):
                    cv2.polylines(img, [seg], False, COLOR[o], 1 if th else 2, cv2.LINE_AA, shift=2)
                if ok[-1] and b == k + 1:
                    cv2.circle(img, (int(pts[-1, 0]), int(pts[-1, 1])), 3, COLOR[o], -1, cv2.LINE_AA)
            if key == "speed":
                for e in self.events[(self.events.t * self.fps > f0) & (self.events.t * self.fps <= now)].itertuples():
                    ue = int(round(float(u(e.t * self.fps))))
                    cv2.line(img, (ue, yb - 10), (ue, yb), IMPACT[e.kind], 2)
        return img


def render(clip: str, model: str, height: int, trail: float, crf: int, seconds: float = 0, window: float = 4.0) -> Path:
    table, cfg = Table.load(), load_config()
    r = pd.read_csv(ROOT / "data/clips/clips.csv", keep_default_na=False).set_index("clip").loc[clip]
    g = f"{r.video}_{r.angle or 'nocam'}"
    cam, H, A = load_framing(g)
    Ainv = np.linalg.inv(A)
    ch, cw = height // 2 * 2, round(height * 16 / 9) // 2 * 2       # each view's size
    T_top, ppm = table_view(table, (cw, ch))
    frame0 = cv2.imread(str(ROOT / "data/calib/frames" / f"{g}.jpg"))
    warped = cv2.warpPerspective(frame0, H @ Ainv @ np.linalg.inv(T_top), (cw, ch),
                                 flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    warped = (warped * 0.55).astype(np.uint8)
    for pts, kind in table_lines(table):
        if kind != "grid":
            color, th = STYLE[kind]
            cv2.polylines(warped, [np.round(apply_h(T_top, pts) * 4).astype(np.int32)], False, color, th, cv2.LINE_AA, shift=2)
    plain = diagram(table, T_top, ppm, (cw, ch))
    px = lambda xy: apply_h(T_top, xy).round().astype(int)

    det = pd.read_parquet(ROOT / "data/detections" / model / f"{clip}.parquet")
    det = to_table(det[det.score >= cfg["min_score"]], cam, cfg, A)
    tr = pd.read_parquet(ROOT / "data/tracks" / model / f"{clip}.parquet")
    T = {o: t.sort_values("frame").reset_index(drop=True) for o, t in tr.groupby("obj")}
    S = {o: (t[["x", "y", "vx", "vy"]].to_numpy(float), t.score.to_numpy(float), t["filled"].to_numpy(bool))
         for o, t in T.items()}
    ev_path = ROOT / "data/tracks" / model / "events" / f"{clip}.parquet"
    events = pd.read_parquet(ev_path) if ev_path.exists() else pd.DataFrame(columns=["t", "frame", "x", "y", "kind"])
    game_path = ROOT / "data/tracks" / model / "game" / f"{clip}.parquet"
    game = pd.read_parquet(game_path) if game_path.exists() else pd.DataFrame(columns=["t", "frame", "kind", "side", "by", "x", "y", "detail"])
    puck = T["puck"]
    phase = puck.phase.to_numpy() if "phase" in puck else np.full(len(puck), "unseen", object)
    # every view shows the moment the footage shows: each frame's picture time (track.retime())
    shown = puck.picture.to_numpy(float) if "picture" in puck else np.arange(r.frames, dtype=float)
    trails = {o: path_points(t, events if o == "puck" else None, r.fps) for o, t in T.items()}
    used = {}                                   # rounded (x, y) of each detection on a path -> object
    for o, t in T.items():
        for x, y in t[["x_det", "y_det"]].dropna().to_numpy():
            used[(round(x, 4), round(y, 4))] = o
    det["obj"] = [used.get((round(x, 4), round(y, 4)), "") for x, y in det[["x", "y"]].to_numpy()]
    by_frame = dict(tuple(det.groupby("frame")))
    graph = Graph(T, events, game, phase, r.fps, table, (cw, ch), window)

    def state(o: str, s: float):
        """o's position and velocity at frame time s, between the frames either side; whether it was detected
        and whether it was filled in, at the nearer frame."""
        X, score, filled = S[o]
        f0 = int(np.clip(np.floor(s), 0, len(X) - 1))
        f1, a = min(f0 + 1, len(X) - 1), s - np.floor(s)
        p = (1 - a) * X[f0] + a * X[f1]
        near = f1 if a >= 0.5 else f0
        if not np.isfinite(p).all():
            p = X[near]
        return p, np.isfinite(score[near]), bool(filled[near])

    cap = cv2.VideoCapture(str(ROOT / r.path))
    w_src, h_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    k_cam = min(cw / w_src, ch / h_src)
    w_cam, h_cam = round(w_src * k_cam), round(h_src * k_cam)
    ox, oy = (cw - w_cam) // 2, (ch - h_cam) // 2
    out = ROOT / "outputs/videos" / model / f"{clip}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{2 * cw}x{2 * ch}", "-r", f"{r.fps}", "-i", "-", "-c:v", "libx264", "-crf", str(crf),
                            "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                           stdin=subprocess.PIPE)
    n_trail, n_short = max(1, int(trail * r.fps)), max(1, int(0.4 * r.fps))
    z = {"puck": cfg["puck"]["z"], "mallet_l": cfg["mallet"]["z"], "mallet_r": cfg["mallet"]["z"]}
    L, g = table.length / 2, (table.goal_width or 0) / 2
    k, last = 0, int(seconds * r.fps) if seconds else r.frames
    while k < last:
        ok, im = cap.read()
        if not ok:
            break
        s = shown[min(k, len(shown) - 1)]
        now = s / r.fps
        left = np.zeros((ch, cw, 3), np.uint8)
        left[oy:oy + h_cam, ox:ox + w_cam] = cv2.resize(im, (w_cam, h_cam), interpolation=cv2.INTER_AREA)
        if k in by_frame:                       # obj "" = a box the tracker didn't use
            for b in by_frame[k].itertuples():
                col = COLOR[b.obj] if b.obj else (150, 150, 150)
                p0 = (int(ox + b.x0 * k_cam), int(oy + b.y0 * k_cam))
                p1 = (int(ox + b.x1 * k_cam), int(oy + b.y1 * k_cam))
                cv2.rectangle(left, p0, p1, col, 2 if b.obj else 1)
                cv2.putText(left, f"{b.score:.2f}", (p0[0], p0[1] - 3), FONT, 0.35, col, 1)
        right, bottom = warped.copy(), plain.copy()
        for e in game[(game.kind == "goal") & (game.t <= now) & (game.t > now - GAME["goal"][1])].itertuples():
            sx = 1 if e.side == "right" else -1           # light up the goal the puck went in
            for view in (right, bottom):
                cv2.line(view, tuple(px([[sx * L, g]])[0]), tuple(px([[sx * L, -g]])[0]), GAME["goal"][0], 8, cv2.LINE_AA)
        for o in OBJECTS:
            if o not in T:
                continue
            p, detected, inferred = state(o, s)
            tt, xy, thin = trails[o]
            for view, n_back, dim in ((right, n_trail, 1.0), (bottom, n_short, 0.6)):
                i0, i1 = np.searchsorted(tt, now - n_back / r.fps), np.searchsorted(tt, now + 1e-9, "right")
                pts = np.r_[xy[i0:i1], [p[:2]]]                   # the trail up to where the object is now
                okp, th = np.isfinite(pts[:, 0]), np.r_[thin[i0:i1], inferred]
                pts = px(np.nan_to_num(pts))
                for i in range(1, len(pts)):
                    if okp[i] and okp[i - 1]:
                        fade = dim * (0.3 + 0.7 * i / len(pts))
                        cv2.line(view, tuple(pts[i - 1]), tuple(pts[i]), tuple(int(c * fade) for c in COLOR[o]),
                                 1 if th[i] else 2, cv2.LINE_AA)
            if not np.isfinite(p[0]):
                continue
            c = tuple(px([p[:2]])[0])
            rad = max(3, int(round(RADIUS[o] * ppm)))
            cv2.circle(right, c, rad, COLOR[o], -1 if detected else 2, cv2.LINE_AA)
            cv2.circle(bottom, c, rad, COLOR[o], 2 if inferred else -1, cv2.LINE_AA)
            if o != "puck":
                cv2.circle(bottom, c, max(2, rad // 2), tuple(int(v * 0.55) for v in COLOR[o]), -1, cv2.LINE_AA)
            elif np.hypot(p[2], p[3]) > 0.3:
                tip = tuple(px([p[:2] + 0.1 * p[2:]])[0])
                cv2.arrowedLine(bottom, c, tip, (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.25)
                label(right, f"puck {np.hypot(p[2], p[3]):4.1f} m/s", (10, 44), COLOR["puck"], 0.5)
            u, v = cam.project(np.c_[apply_h(Ainv, [p[:2]]), z[o]])[0] * k_cam + (ox, oy)
            cv2.circle(left, (int(u), int(v)), 9, COLOR[o], 1 if inferred else 2, cv2.LINE_AA)
        for e in events[(events.t <= now) & (events.t > now - 0.5)].itertuples():
            c = tuple(px([[e.x, e.y]])[0])
            fade = 1 - (now - e.t) / 0.5
            col = tuple(int(v * (0.4 + 0.6 * fade)) for v in IMPACT[e.kind])
            seen = getattr(e, "seen", True)
            cv2.circle(right, c, int(RADIUS["puck"] * ppm) + 6, col, 2 if seen else 1, cv2.LINE_AA)
            text = f"{e.kind} e={e.restitution:.2f}" if e.kind in ("rail", "end") and np.isfinite(e.restitution) else e.kind
            label(right, text + ("" if seen else " (gap)"), (c[0] + 12, c[1] - 8), col, 0.42)
        recent = game[game.kind.isin(list(GAME)) & (game.t <= now)]
        recent = recent[(now - recent.t).to_numpy() < np.array([GAME[kd][1] for kd in recent.kind], float)]
        if len(recent):                          # the latest game event, across the top of the footage
            e = recent.iloc[-1]
            col, _, fmt = GAME[e.kind]
            text = fmt.format(side=e.side, player=PLAYER.get(e.by, ""), detail=e.detail)
            (tw, th_), _ = cv2.getTextSize(text, FONT, 0.8, 2)
            x0 = (cw - tw) // 2
            cv2.rectangle(left, (x0 - 12, 30), (x0 + tw + 12, 40 + th_ + 10), (20, 20, 20), -1)
            cv2.rectangle(left, (x0 - 12, 30), (x0 + tw + 12, 40 + th_ + 10), col, 2)
            cv2.putText(left, text, (x0, 40 + th_), FONT, 0.8, col, 2, cv2.LINE_AA)
        ph = phase[min(int(round(s)), len(phase) - 1)]
        label(left, f"{clip}  {now:5.2f} s", (10, 20), scale=0.5)
        label(right, "table (warped footage)", (10, 20))
        label(bottom, "positions", (10, 20))
        label(bottom, ph, (10 + cv2.getTextSize("positions", FONT, 0.45, 1)[0][0] + 12, 20), PHASE.get(ph, PHASE["unseen"]), 0.5)
        frame = np.zeros((2 * ch, 2 * cw, 3), np.uint8)
        frame[:ch, :cw], frame[:ch, cw:], frame[ch:, :cw], frame[ch:, cw:] = left, right, bottom, graph.draw(s)
        cv2.line(frame, (cw, 0), (cw, 2 * ch), (90, 90, 90), 1)
        cv2.line(frame, (0, ch), (2 * cw, ch), (90, 90, 90), 1)
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
    ap.add_argument("--height", type=int, default=450, help="of each view; the video is twice this")
    ap.add_argument("--trail", type=float, default=1.0, help="seconds of trail behind each object")
    ap.add_argument("--window", type=float, default=4.0, help="seconds of history in the live graph")
    ap.add_argument("--crf", type=int, default=26)
    ap.add_argument("--seconds", type=float, default=0, help="render only the first this many seconds (0 = all)")
    a = ap.parse_args()
    for c in a.clips:
        print(render(c, a.model, a.height, a.trail, a.crf, a.seconds, a.window))


if __name__ == "__main__":
    main()
