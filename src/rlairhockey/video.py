"""Video decoding with real timestamps, plus per-frame signals for shot splitting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: Path
    fps: float
    duration: float
    width: int
    height: int
    n_frames_est: int


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or s.guessed_rate)
        dur = float(c.duration / av.time_base) if c.duration else float(s.duration * s.time_base)
        return VideoInfo(path, fps, dur, s.width, s.height, int(round(dur * fps)))


def iter_frames(path: str | Path, width: int = 320, start: float = 0.0,
                end: float | None = None) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (pts_seconds, BGR frame resized to `width`) in decode order.

    Timestamps come from the container PTS, not frame index, so dropped or
    duplicated frames and variable frame rate are handled.
    """
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        height = int(round(s.height * width / s.width / 2)) * 2
        if start > 0:
            c.seek(int(start / s.time_base), stream=s, backward=True)
        for f in c.decode(s):
            if f.pts is None:
                continue
            t = float(f.pts * s.time_base)
            if t < start:
                continue
            if end is not None and t >= end:
                break
            yield t, f.to_ndarray(width=width, height=height, format="bgr24")


def frame_at(path: str | Path, t: float, width: int | None = None) -> np.ndarray:
    """Full- or reduced-resolution BGR frame at the first PTS >= t."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        w = width or s.width
        h = int(round(s.height * w / s.width / 2)) * 2
        c.seek(int(t / s.time_base), stream=s, backward=True)
        for f in c.decode(s):
            if f.pts is not None and float(f.pts * s.time_base) >= t - 1e-6:
                return f.to_ndarray(width=w, height=h, format="bgr24")
    raise ValueError(f"no frame at t={t}")


def content_change(prev_hsv: np.ndarray, hsv: np.ndarray) -> float:
    """Mean absolute HSV difference (PySceneDetect ContentDetector-style score)."""
    d = cv2.absdiff(prev_hsv, hsv).reshape(-1, 3).mean(axis=0)
    return float(d.mean())


def detect_cuts(scores: np.ndarray, times: np.ndarray, ratio: float = 3.0,
                min_score: float = 15.0, window: int = 4, min_shot_s: float = 0.4) -> list[int]:
    """Adaptive cut detection: a frame is a cut when its change score is well above
    the local neighborhood (rejects fast pans and puck motion, catches hard cuts).
    Returns indices of the first frame of each new shot."""
    n = len(scores)
    cuts: list[int] = []
    for i in range(1, n):
        lo, hi = max(1, i - window), min(n, i + window + 1)
        nb = np.concatenate([scores[lo:i], scores[i + 1:hi]])
        base = float(np.mean(nb)) if len(nb) else 0.0
        if scores[i] >= min_score and scores[i] >= ratio * max(base, 1.0):
            if not cuts or times[i] - times[cuts[-1]] >= min_shot_s:
                cuts.append(i)
    return cuts
