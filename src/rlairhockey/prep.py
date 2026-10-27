"""Video preparation: per-frame signals -> shots -> camera groups -> labeled segments.

Model-free apart from zero-shot CLIP, so it works across broadcast productions
(table color, graphics and camera placement all differ between events).

Per frame (computed on the 320x180 proxy by `frame_features`):
  cut        content-change score vs previous frame
  luma       mean brightness (black frames, fades)
  right_bar, bottom_bar   brightness of the strips that go black in the
             picture-in-picture replay layout (2024 Worlds style)
  tiny       16x9 Lab thumbnail (camera grouping, replay matching)
  grid       32x18 fraction of pixels that changed vs previous frame (motion)

Shot boundaries: hard cuts (cut score) plus soft cuts/dissolves (thumbnail jump between
the 0.25 s before and after a frame).
Per shot: CLIP zero-shot type (table / booth / crowd / interview / graphic); replay
(re-shows earlier footage, or sits between two replay stinger graphics).
Per camera group: median background frame -> table region by flood fill from the
center -> motion measured inside it.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .video import detect_cuts, frame_at, iter_frames

TINY_W, TINY_H = 16, 9
GRID_W, GRID_H = 32, 18


def frame_features(path: str | Path, start: float = 0.0, end: float | None = None
                   ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """One pass over a proxy video -> (per-frame table, tiny Lab thumbs, motion grids)."""
    rows, tinies, grids = [], [], []
    prev_hsv = prev_gray = None
    for t, f in iter_frames(path, width=320, start=start, end=end):
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        cut = float(cv2.absdiff(prev_hsv, hsv).mean()) if prev_hsv is not None else 0.0
        if prev_gray is not None:
            moved = (cv2.absdiff(prev_gray, gray) > 25).astype(np.float32)
            grid = cv2.resize(moved, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)
        else:
            grid = np.zeros((GRID_H, GRID_W), np.float32)
        grids.append((grid * 255).astype(np.uint8))
        tinies.append(cv2.cvtColor(cv2.resize(f, (TINY_W, TINY_H), interpolation=cv2.INTER_AREA),
                                   cv2.COLOR_BGR2LAB))
        rows.append((t, cut, float(gray.mean()),
                     float(gray[: int(0.5 * h), int(0.94 * w):].mean()),
                     float(gray[int(0.94 * h):, : int(0.5 * w)].mean())))
        prev_hsv, prev_gray = hsv, gray
    df = pd.DataFrame(rows, columns=["t", "cut", "luma", "right_bar", "bottom_bar"])
    return df, np.stack(tinies), np.stack(grids)


# ---------------------------------------------------------------- parameters
DEFAULTS = dict(
    cut_ratio=3.0, cut_min=30.0,     # hard cuts: well above neighbors and above an absolute floor
    jump_w_s=0.25, jump_min=15.0,    # soft cuts/dissolves: before vs after window thumbnails differ this much,
    jump_sharp_min=12.0, cut_sharp=12.0,  # ... or this much with a cut score this high within 2 frames
    jump_sep_s=0.5,                  # ... and are at least this far from another boundary
    group_dist=9.0,                  # shots whose median thumbnails differ less than this share a camera
    shot_jitter=14.0,                # shots moving more than this inside themselves are handheld/pans
    group_min_s=20.0,                # a camera group needs this much screen time to count
    not_table_max=0.4,               # a table camera's mean CLIP booth+interview+graphic stays below this
    table_min=0.1,                   # ... its mean CLIP "table" probability is at least this
    region_min=0.05,                 # ... and a table region covering this much of the frame is found
    act_on=0.006, idle_s=6.0,        # table motion below act_on for idle_s seconds = idle
    break_min_s=10.0,                # off-table/idle stretches at least this long are breaks
    break_gap_s=8.0,                 # shorter non-replay stretches between two breaks join them
    replay_exact=1.0,                # this close to earlier footage = the same frames shown again
    replay_dist=3.5, replay_rho=0.6,  # close, in-order matches to earlier footage ...
    replay_slope=(0.15, 1.3),        # ... at real speed or slow motion = replay
    replay_window_s=240.0,           # how far back to look for the original footage
    stinger_win_s=0.75,              # replay intro/outro graphics: frames this close to a boundary ...
    stinger_dist=1.5, stinger_gap_s=5.0,  # ... matching frames at another boundary elsewhere ...
    stinger_motion=2.0,              # ... while animating (change vs 1/30 s earlier) ...
    stinger_frames=10,               # ... this many of them
    stinger_merge_s=2.0,             # stinger boundaries this close are one occurrence
    stinger_span_s=60.0,             # two occurrences this close bracket one replay
    lone_replay_s=15.0,              # an unpaired one covers up to this much following non-camera table footage
    absorb_s=2.0,                    # shorter cutaway pieces touching a replay are part of it
)

CLIP_PROMPTS = {
    "table": "a wide view of an air hockey table with two players playing",
    "booth": "commentators sitting at a desk with microphones",
    "crowd": "a crowd of people sitting in an audience",
    "interview": "people standing and talking, an interview or trophy presentation",
    "graphic": "a logo or title graphic on a plain background",
}


def _runs(flag: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where flag is True."""
    d = np.diff(np.concatenate([[0], flag.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _tdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean absolute Lab difference between tiny thumbnails (broadcast over leading dims)."""
    return np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=(-3, -2, -1))


def window_jump(tiny: np.ndarray, w: int) -> np.ndarray:
    """Per frame i: mean abs Lab difference between the average thumbnail of frames
    [i-w, i) and of [i, i+w). Peaks at hard cuts and in the middle of dissolves/wipes;
    stays low inside a shot from a fixed camera."""
    f = tiny.reshape(len(tiny), -1)
    c = np.zeros((len(f) + 1, f.shape[1]), np.int32)
    np.cumsum(f, axis=0, dtype=np.int32, out=c[1:])
    d = np.zeros(len(f), np.float32)
    i = np.arange(w, len(f) - w + 1)
    for s in range(0, len(i), 20000):
        j = i[s:s + 20000]
        d[j] = np.abs((c[j + w] - c[j]) - (c[j] - c[j - w])).mean(axis=1) / w
    return d


def _peaks(d: np.ndarray, t: np.ndarray, thr: float, sep_s: float, avoid: list[int]) -> list[int]:
    """Local maxima of d above thr, strongest first, at least sep_s from `avoid` and each other."""
    cand = np.flatnonzero((d[1:-1] >= thr) & (d[1:-1] >= d[:-2]) & (d[1:-1] > d[2:])) + 1
    taken = list(t[avoid]) if len(avoid) else []
    out = []
    for i in cand[np.argsort(-d[cand])]:
        if not taken or np.min(np.abs(np.array(taken) - t[i])) >= sep_s:
            taken.append(t[i]); out.append(int(i))
    return sorted(out)


# ---------------------------------------------------------------- shots
def find_shots(df: pd.DataFrame, tiny: np.ndarray, p: dict) -> pd.DataFrame:
    t = df.t.to_numpy()
    dt = float(np.median(np.diff(t)))
    pip = ((df.right_bar < 15) & (df.bottom_bar < 15) & (df.luma > 30)).to_numpy()
    black = (df.luma < 12).to_numpy()
    hard = detect_cuts(df.cut.to_numpy(), t, p["cut_ratio"], p["cut_min"])
    jump = window_jump(tiny, max(2, int(round(p["jump_w_s"] / dt))))
    # quick dissolves between similar views (two side cameras) jump less but change sharply
    sharp = df.cut.rolling(5, center=True, min_periods=1).max().to_numpy() >= p["cut_sharp"]
    jump = np.where((jump >= p["jump_min"]) | sharp, jump, 0.0)
    cuts = set(hard) | set(_peaks(jump, t, p["jump_sharp_min"], p["jump_sep_s"], hard))
    for flag in (pip, black):          # layout switches and fades are boundaries too
        for s, e in _runs(flag):
            cuts.update(i for i in (s, e) if 0 < i < len(df))
    bounds = [0] + sorted(cuts) + [len(df)]
    rows = []
    for k, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        idx = np.linspace(s, e - 1, min(e - s, 9)).astype(int)
        med = np.median(tiny[idx].astype(np.float32), axis=0)
        rows.append(dict(shot=k, i0=s, i1=e, start=float(t[s]), end=float(t[e - 1] + dt),
                         jitter=float(_tdist(tiny[idx], med[None]).mean()),
                         pip=float(pip[s:e].mean()), black=float(black[s:e].mean()), _med=med))
    return pd.DataFrame(rows)


def clip_shot_types(proxy: Path, shots: pd.DataFrame, cache: Path | None = None) -> pd.DataFrame:
    """CLIP zero-shot probabilities per shot (mean over up to 3 frames). Cached by shot start."""
    if cache is not None and cache.exists():
        c = pd.read_parquet(cache)
        if len(c) == len(shots) and np.allclose(c.start.to_numpy(), shots.start.to_numpy()):
            return c
    import torch
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-base-patch32"
    model, proc = CLIPModel.from_pretrained(name).eval(), CLIPProcessor.from_pretrained(name)
    txt = proc(text=list(CLIP_PROMPTS.values()), return_tensors="pt", padding=True)
    # one sequential decode, grabbing the frames each shot needs
    want = []
    for r in shots.itertuples():
        d = r.end - r.start
        for q in ((0.5,) if d < 1.0 else (0.2, 0.5, 0.8)):
            want.append((r.start + q * d, r.shot))
    want.sort()
    imgs, owner, k = [], [], 0
    for t, f in iter_frames(proxy, width=320):
        while k < len(want) and t >= want[k][0]:
            imgs.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)); owner.append(want[k][1]); k += 1
        if k == len(want):
            break
    probs = []
    with torch.no_grad():
        for i in range(0, len(imgs), 64):
            px = proc(images=imgs[i:i + 64], return_tensors="pt")["pixel_values"]
            out = model(input_ids=txt["input_ids"], attention_mask=txt["attention_mask"], pixel_values=px)
            probs.append(out.logits_per_image.softmax(-1).numpy())
    pr = pd.DataFrame(np.concatenate(probs), columns=list(CLIP_PROMPTS)).assign(shot=owner)
    out = pr.groupby("shot").mean().reindex(shots.shot).reset_index()
    out.insert(1, "start", shots.start.to_numpy())
    if cache is not None:
        out.to_parquet(cache)
    return out


def group_cameras(shots: pd.DataFrame, p: dict) -> pd.Series:
    """Greedy grouping of steady shots by median-thumbnail distance, longest shots first.
    Returns a group id per shot (-1 = no group: handheld, too short, black, PiP)."""
    cand = shots[(shots.jitter <= p["shot_jitter"]) & (shots.black < 0.5) & (shots.pip < 0.5)
                 & (shots.end - shots.start >= 0.5)]
    cand = cand.assign(dur=cand.end - cand.start).sort_values("dur", ascending=False)
    centers, gid = [], pd.Series(-1, index=shots.index)
    for i, r in cand.iterrows():
        d = [float(_tdist(r._med, c)) for c in centers]
        if d and min(d) <= p["group_dist"]:
            gid[i] = int(np.argmin(d))
        else:
            centers.append(r._med); gid[i] = len(centers) - 1
    return gid


def table_region(proxy: Path, shots: pd.DataFrame, n: int = 25) -> np.ndarray:
    """Table mask (GRID_H x GRID_W float) for one camera group: median of frames sampled
    across the group's shots (players average out), flood-filled from the dominant
    color near the frame center."""
    spans = [(r.start, r.end) for r in shots.itertuples()]
    total = sum(e - s for s, e in spans)
    ts = []
    for q in (np.arange(n) + 0.5) / n * total:
        acc = 0.0
        for s, e in spans:
            if q <= acc + (e - s):
                ts.append(s + q - acc); break
            acc += e - s
    med = np.median(np.stack([frame_at(proxy, t) for t in ts]), axis=0).astype(np.uint8)
    lab = cv2.cvtColor(cv2.medianBlur(med, 5), cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]
    code = (lab // 8).astype(np.int32) @ np.array([1024, 32, 1])
    box = code[int(0.35 * h):int(0.65 * h), int(0.3 * w):int(0.7 * w)].ravel()
    ys, xs = np.nonzero(code == np.bincount(box).argmax())
    j = np.argmin((ys - h / 2) ** 2 + (xs - w / 2) ** 2)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(lab.copy(), mask, (int(xs[j]), int(ys[j])), 0, (10, 6, 6), (10, 6, 6),
                  cv2.FLOODFILL_MASK_ONLY | 4 | (255 << 8))
    m = cv2.morphologyEx(mask[1:-1, 1:-1], cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return cv2.resize(m.astype(np.float32) / 255, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)


def replay_score(tiny: np.ndarray, t: np.ndarray, shots: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Per shot: does it re-show earlier footage in order? For up to 20 sampled frames, find
    the closest frame in the previous `replay_window_s` seconds (before this shot).
    A replay matches closely (dist), its matches advance with it (rho, rank correlation)
    at real or slow-motion speed (slope = source seconds per shown second).
    Live play from a fixed camera also matches closely, but to scattered moments."""
    from scipy.stats import spearmanr
    flat = tiny.reshape(len(tiny), -1).astype(np.float32)
    rows = []
    for r in shots.itertuples():
        out = dict(replay_dist=np.inf, replay_rho=0.0, replay_slope=0.0)
        lo, hi = np.searchsorted(t, r.start - p["replay_window_s"]), int(r.i0) - 1
        if r.black <= 0.5 and hi - lo >= 10:
            idx = np.unique(np.linspace(r.i0, r.i1 - 1, min(int(r.i1 - r.i0), 20)).astype(int))
            d = np.abs(flat[idx][:, None, :] - flat[lo:hi][None]).mean(-1)
            src = t[lo + d.argmin(1)]
            out["replay_dist"] = float(np.median(d.min(1)))
            if len(idx) >= 3 and np.ptp(src) > 0:
                out["replay_rho"] = float(spearmanr(idx, src).correlation)
                out["replay_slope"] = float(np.polyfit(t[idx], src, 1)[0])
        rows.append(out)
    return pd.DataFrame(rows, index=shots.index)


def is_replay(r, p: dict) -> bool:
    if r.pip > 0.5:
        return True
    if r.replay_dist <= p["replay_exact"]:            # frame-for-frame repeat (stingers, short re-shows)
        return True
    return (r.replay_dist <= p["replay_dist"] and r.replay_rho >= p["replay_rho"]
            and p["replay_slope"][0] <= r.replay_slope <= p["replay_slope"][1])


def find_stingers(tiny: np.ndarray, t: np.ndarray, bounds: np.ndarray, p: dict) -> list[tuple[int, int]]:
    """Replay intro/outro graphics ("INSTANT REPLAY", "OFF SPEED REPLAY" wipes): an
    animation played at a shot boundary whose frames reappear exactly at another boundary.
    Animated frames only, so a static booth or stats screen shown twice does not count.
    Returns [start, end) frame spans, one per stinger occurrence, in time order."""
    dt = float(np.median(np.diff(t)))
    k, step, lag = (max(1, int(round(s / dt))) for s in (p["stinger_win_s"], 1 / 30, 1 / 30))
    idx = [np.arange(max(lag, i - k), min(len(t), i + k), step) for i in bounds]
    owner = np.concatenate([np.full(len(a), b) for b, a in enumerate(idx)])
    idx = np.concatenate(idx)
    x = tiny[idx].reshape(len(idx), -1).astype(np.float32)
    moving = np.abs(x - tiny[idx - lag].reshape(len(idx), -1)).mean(1) >= p["stinger_motion"]
    bt = t[bounds][owner]
    hit = np.zeros(len(x), bool)
    for s in range(0, len(x), 32):
        d = np.abs(x[s:s + 32, None] - x[None]).mean(-1)
        ok = (d <= p["stinger_dist"]) & moving[None] & (np.abs(bt[s:s + 32, None] - bt[None]) > p["stinger_gap_s"])
        hit[s:s + 32] = ok.any(1) & moving[s:s + 32]
    is_st = np.bincount(owner, weights=hit, minlength=len(bounds)) >= p["stinger_frames"]
    events: list[list[int]] = []
    for b in np.flatnonzero(is_st):            # neighboring stinger boundaries = one occurrence
        f = idx[(owner == b) & hit]
        if events and t[f[0]] - t[events[-1][1] - 1] <= p["stinger_merge_s"]:
            events[-1][1] = max(events[-1][1], int(f[-1]) + 1)
        else:
            events.append([int(f[0]), int(f[-1]) + 1])
    return [(a, b) for a, b in events]


# ---------------------------------------------------------------- labeling
def segment_video(df: pd.DataFrame, tiny: np.ndarray, grid: np.ndarray, proxy: Path,
                  game_start: float | None = None, game_end: float | None = None,
                  clip_cache: Path | None = None, **overrides) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (segments, shots). Segment labels:
    intro, play, idle, replay, cutaway (short off-table stretch), break, outro."""
    p = {**DEFAULTS, **overrides}
    t = df.t.to_numpy()
    dt = float(np.median(np.diff(t)))
    n = len(df)
    shots = find_shots(df, tiny, p)
    clip = clip_shot_types(proxy, shots, clip_cache)
    for c in CLIP_PROMPTS:
        shots[c] = clip[c].fillna(0.0).to_numpy()
    shots["kind"] = shots[list(CLIP_PROMPTS)].idxmax(axis=1)
    shots["group"] = group_cameras(shots, p)
    shots = shots.join(replay_score(tiny, t, shots, p))
    shots["replay"] = [is_replay(r, p) for r in shots.itertuples()]
    # replay packages: stinger in, replay, stinger out. Pair occurrences in time order;
    # an unpaired one (replay straight into a break) is extended once cameras are known.
    stingers = find_stingers(tiny, t, shots.i0.to_numpy()[1:], p)
    package = np.zeros(n, bool)
    lone, k = [], 0
    while k < len(stingers):
        a, b = stingers[k]
        if k + 1 < len(stingers) and t[stingers[k + 1][0]] - t[b - 1] <= p["stinger_span_s"]:
            b = stingers[k + 1][1]; k += 1
        else:
            lone.append(b)
        package[a:b] = True; k += 1
    shots["package"] = [float(package[int(r.i0):int(r.i1)].mean()) for r in shots.itertuples()]

    # camera groups: big, steady, not booth/interview/graphic per CLIP, with a findable table
    shots["angle"] = ""
    dur = shots.end - shots.start
    g = shots[(shots.group >= 0) & (shots.package < 0.5)].assign(  # replay-only cameras do not count
        dur=dur, nt=shots.booth + shots.interview + shots.graphic)
    stats = g.groupby("group").apply(lambda x: pd.Series(dict(
        dur=x.dur.sum(), table=np.average(x.table, weights=x.dur), not_table=np.average(x.nt, weights=x.dur))))
    stats = stats[(stats.dur >= p["group_min_s"]) & (stats.not_table <= p["not_table_max"])
                  & (stats.table >= p["table_min"])].sort_values("dur", ascending=False)
    regions, rank = {}, 0
    for gid in stats.index:
        m = table_region(proxy, shots[shots.group == gid])
        if m.mean() < p["region_min"]:
            continue
        rank += 1
        shots.loc[shots.group == gid, "angle"] = f"cam{rank}"
        regions[f"cam{rank}"] = m
    # after an unpaired stinger: the replay is the table footage that no live camera claims
    for b in lone:
        total = 0.0
        for r in shots[shots.i1 > b + int(round(0.5 / dt))].itertuples():
            total += r.end - r.start
            if r.angle or r.kind != "table" or r.black > 0.5 or total > p["lone_replay_s"]:
                break
            package[int(r.i0):int(r.i1)] = True
    shots["package"] = [float(package[int(r.i0):int(r.i1)].mean()) for r in shots.itertuples()]

    # per-frame motion inside the camera's table region
    act = np.zeros(n, np.float32)
    lab = np.full(n, "other", dtype=object)
    ang = np.full(n, "", dtype=object)
    for r in shots.itertuples():
        sl = slice(int(r.i0), int(r.i1))
        if r.replay:
            lab[sl] = "replay"
        elif r.black > 0.5:
            lab[sl] = "black"
        elif r.angle:
            lab[sl], ang[sl] = "play", r.angle
            m = regions[r.angle]
            act[sl] = (grid[sl].astype(np.float32) / 255 * m).sum(axis=(1, 2)) / max(m.sum(), 1e-6)
        elif r.kind == "table":            # table footage from a rare or mixed angle: no camera
            lab[sl] = "play"
    lab[package], ang[package] = "replay", ""
    w = max(1, int(round(1.0 / dt)))
    act_s = pd.Series(act).rolling(w, center=True, min_periods=1).mean().to_numpy()
    for s, e in _runs((lab == "play") & (ang != "") & (act_s < p["act_on"])):
        if (e - s) * dt >= p["idle_s"]:
            lab[s:e] = "idle"

    # game window: config override, else first/last sustained play
    play_runs = [(s, e) for s, e in _runs(lab == "play") if (e - s) * dt >= 5.0]
    g0 = game_start if game_start is not None else (t[play_runs[0][0]] if play_runs else 0.0)
    g1 = game_end if game_end is not None else (t[play_runs[-1][1] - 1] if play_runs else t[-1])
    lab[t < g0] = "intro"
    lab[t > g1] = "outro"
    # inside the game: long off-table/idle stretches are breaks, short off-table ones cutaways
    off = np.isin(lab, ["other", "black", "idle"])
    for s, e in _runs(off):
        if (e - s) * dt >= p["break_min_s"]:
            lab[s:e] = "break"; ang[s:e] = ""
        else:
            sub = lab[s:e]
            sub[np.isin(sub, ["other", "black"])] = "cutaway"
    # a few seconds of table between two breaks (people walking past) is still the break
    brk = _runs(lab == "break")
    for (_, s), (e, _) in zip(brk[:-1], brk[1:]):
        if (e - s) * dt < p["break_gap_s"] and not (lab[s:e] == "replay").any():
            lab[s:e] = "break"; ang[s:e] = ""
    # leftover pieces of a replay package (wipe frames, fades) join the replay they touch
    for s, e in _runs((lab == "cutaway") | ((lab == "play") & (ang == ""))):
        if (e - s) * dt < p["absorb_s"] and ((s > 0 and lab[s - 1] == "replay") or (e < n and lab[e] == "replay")):
            lab[s:e] = "replay"

    change = np.flatnonzero((lab[1:] != lab[:-1]) | (ang[1:] != ang[:-1])) + 1
    edges = [0, *change.tolist(), n]
    segs = [dict(start=round(float(t[s]), 3), end=round(float(t[e - 1] + dt), 3), label=lab[s],
                 angle=ang[s], activity=round(float(act_s[s:e].mean()), 4))
            for s, e in zip(edges[:-1], edges[1:])]
    seg = pd.DataFrame(segs)
    seg["dur"] = (seg.end - seg.start).round(3)
    return seg, shots.drop(columns="_med")
