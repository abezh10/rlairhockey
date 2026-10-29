"""Check detected goals (game.py) against the broadcast scoreboard (score_changes.py).

The scoreboard's points digits (configs/videos.yaml, score_roi and score_cells) are read by shape: each
digit's strokes are thresholded out and grouped with the same shapes seen elsewhere in the match. In time
order across a video's clips, a point is one player's digit changing to a new one that lasts. The operator
updates the score a few seconds after a goal, sometimes after the clip has cut away, so a change is matched
to the last detected goal in the --window seconds before it (in source-video time).
Prints, per match, the score changes, goals, and matches; writes data/tracks/<model>/goal_check.csv (one
row per score change and per unmatched goal).
Usage: python scripts/check_goals.py [--model v2] [--window 12]
"""
import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SAME, MIN_COUNT, KEEP = 0.8, 10, 1.0          # IoU for the same digit; samples a digit needs; s a new score must last
SHOWN = 0.5                                   # correlation with the usual overlay (outside the digits) when it's on screen
RESET_GAP = 20.0                              # s; a reset to 0-0 with no point this long before it is the game's last point


def digit(cell: np.ndarray) -> np.ndarray:
    """The digit's strokes in one cell of the crop: the smaller side of an Otsu threshold."""
    _, m = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = m > 0
    return m if m.mean() < 0.5 else ~m


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return (a & b).sum() / max((a | b).sum(), 1)


def readings(clips: pd.DataFrame, cells: list) -> pd.DataFrame:
    """For each sampled frame of one video's clips (in time order): its time in the source video, and per
    cell which digit it shows, as an id (digits are grouped by shape; one seen fewer than MIN_COUNT times
    is a transition: -1; so is a frame where the overlay is hidden, which doesn't look like it usually does
    outside the digits)."""
    rows, masks, crops = [], [[] for _ in cells], []
    for r in clips.itertuples():
        p = ROOT / "data/clips/score" / f"{r.clip}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        h, w = z["crops"].shape[1:]
        for f, crop in zip(z["frames"], z["crops"]):
            rows.append(dict(t=r.start + f / r.fps, clip=r.clip, clip_t=f / r.fps))
            crops.append(crop.astype(np.float32))
            for i, (x0, y0, x1, y1) in enumerate(cells):
                masks[i].append(digit(crop[int(y0 * h):int(np.ceil(y1 * h)), int(x0 * w):int(np.ceil(x1 * w))]))
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    crops = np.array(crops)
    h, w = crops.shape[1:]
    rest = np.ones((h, w), bool)                                  # the overlay outside the digit cells
    for x0, y0, x1, y1 in cells:
        rest[int(y0 * h):int(np.ceil(y1 * h)), int(x0 * w):int(np.ceil(x1 * w))] = False
    usual = np.median(crops, axis=0)[rest]
    a = crops[:, rest] - crops[:, rest].mean(1, keepdims=True)
    b = usual - usual.mean()
    shown = (a @ b) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b), 1e-6) >= SHOWN
    for i, ms in enumerate(masks):
        protos, counts, ids = [], [], []
        for m in ms:
            k = next((j for j, q in enumerate(protos) if iou(m, q) >= SAME), None)
            if k is None:
                protos.append(m)
                counts.append(0)
                k = len(protos) - 1
            counts[k] += 1
            ids.append(k)
        out[f"d{i}"] = [k if counts[k] >= MIN_COUNT and ok else -1 for k, ok in zip(ids, shown)]
    return out


def score_changes(read: pd.DataFrame) -> list[dict]:
    """Moments one player's points read differently from before, for at least KEEP seconds (time-ordered
    readings). Both changing at once is the reset to 0-0 for a new game: the game's last point if the board
    didn't show it first (no point in the RESET_GAP seconds before)."""
    cols = [c for c in read if c.startswith("d")]
    ok = (read[cols] >= 0).all(axis=1).to_numpy()
    r = read[ok].reset_index(drop=True)
    key = list(map(tuple, r[cols].to_numpy()))
    changes, cur, i = [], None, 0
    while i < len(r):
        j = i
        while j + 1 < len(r) and key[j + 1] == key[i]:
            j += 1
        if r.t[j] - r.t[i] >= KEEP or j == len(r) - 1:
            n_changed = 0 if cur is None else sum(a != b for a, b in zip(key[i], cur))
            last = changes[-1]["t"] if changes else -np.inf
            if n_changed == 1 or (n_changed > 1 and r.t[i] - last > RESET_GAP):
                changes.append(dict(t=r.t[i], clip=r["clip"][i], clip_t=r.clip_t[i], before=cur, after=key[i],
                                    reset=n_changed > 1))
            cur = key[i]
        i = j + 1
    return changes


def video_changes(vid: str, cv: pd.DataFrame, cells: list) -> list[dict]:
    """score_changes() for one video, cached in data/clips/score/changes_<video>.pkl until a crop or the
    cells change."""
    cache = ROOT / "data/clips/score" / f"changes_{vid}.pkl"
    crops = [ROOT / "data/clips/score" / f"{c}.npz" for c in cv["clip"]]
    newest = max((p.stat().st_mtime for p in crops if p.exists()), default=0)
    if cache.exists() and cache.stat().st_mtime > newest:
        cached = pickle.load(open(cache, "rb"))
        if cached["cells"] == cells:
            return cached["changes"]
    changes = score_changes(readings(cv, cells))
    pickle.dump(dict(cells=cells, changes=changes), open(cache, "wb"))
    return changes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v2")
    ap.add_argument("--window", type=float, default=12.0)
    a = ap.parse_args()
    clips = pd.read_csv(ROOT / "data/clips/clips.csv", keep_default_na=False).sort_values(["video", "start"])
    videos = yaml.safe_load(open(ROOT / "configs/videos.yaml"))["videos"]
    rows = []
    for vid, cv in clips.groupby("video"):
        if "score_cells" not in videos[vid]:
            continue
        changes = video_changes(vid, cv, videos[vid]["score_cells"])
        goals = []
        for r in cv.itertuples():
            p = ROOT / "data/tracks" / a.model / "game" / f"{r.clip}.parquet"
            if p.exists():
                g = pd.read_parquet(p)
                goals += [dict(t=r.start + e.t, clip=r.clip, clip_t=e.t, side=e.side) for e in g[g.kind == "goal"].itertuples()]
        used = set()
        for c in changes:
            # the last goal before the change, within the window (and after the previous score change)
            cand = [i for i, gl in enumerate(goals) if c["t"] - a.window <= gl["t"] <= c["t"] + 1 and i not in used]
            i = cand[-1] if cand else None
            if i is not None:
                used.add(i)
            rows.append(dict(video=vid, kind="score change", t=round(c["t"], 2), clip=c["clip"], clip_t=round(c["clip_t"], 2),
                             goal_clip=goals[i]["clip"] if i is not None else "", goal_t=round(goals[i]["clip_t"], 2) if i is not None else np.nan,
                             lag=round(c["t"] - goals[i]["t"], 2) if i is not None else np.nan))
        for i, gl in enumerate(goals):
            if i not in used:
                rows.append(dict(video=vid, kind="goal, no score change", t=round(gl["t"], 2), clip="", clip_t=np.nan,
                                 goal_clip=gl["clip"], goal_t=round(gl["clip_t"], 2), lag=np.nan))
        n_match = len(used)
        print(f"{vid}: {len(changes)} score changes, {len(goals)} goals detected, {n_match} matched "
              f"(precision {n_match / max(len(goals), 1):.0%}, recall {n_match / max(len(changes), 1):.0%})")
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data/tracks" / a.model / "goal_check.csv", index=False)
    print(ROOT / "data/tracks" / a.model / "goal_check.csv")


if __name__ == "__main__":
    main()
