"""Debug drawings: the table model over a camera image, and the image warped to a top-down view."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from rlairhockey.geometry import Table, apply_h

YELLOW, CYAN, MAGENTA, GREEN = (0, 220, 255), (255, 200, 0), (255, 0, 255), (0, 200, 0)


def table_lines(table: Table, grid: float = 0.25) -> list[tuple[np.ndarray, str]]:
    """Polylines in table coordinates: ("outline" | "center" | "goal" | "grid")."""
    L, W = table.length / 2, table.width / 2
    out = [(np.array([[-L, W], [L, W], [L, -W], [-L, -W], [-L, W]]), "outline"),
           (np.array([[0, W], [0, -W]]), "center")]
    for x in np.arange(grid, L, grid):
        out += [(np.array([[s * x, W], [s * x, -W]]), "grid") for s in (-1, 1)]
    for y in np.arange(-W + grid, W - 1e-6, grid):
        out.append((np.array([[-L, y], [L, y]]), "grid"))
    if table.goal_width:
        g = table.goal_width / 2
        out += [(np.array([[s * L, g], [s * L, -g]]), "goal") for s in (-1, 1)]
    return out


STYLE = {"outline": (YELLOW, 2), "center": (YELLOW, 2), "goal": (MAGENTA, 4), "grid": (CYAN, 1)}


def overlay(img: np.ndarray, H: np.ndarray, table: Table, clicked: dict | None = None,
            fitted: dict | None = None) -> np.ndarray:
    """Table model projected through H (table -> image). clicked/fitted map landmark name -> (u, v):
    clicks are drawn as circles, the model's reprojection of the same landmark as a cross."""
    grid = img.copy()
    lines = table_lines(table)
    for pts, kind in lines:
        if kind == "grid":
            cv2.polylines(grid, [np.round(apply_h(H, pts) * 4).astype(np.int32)], False, CYAN, 1, cv2.LINE_AA, shift=2)
    out = cv2.addWeighted(grid, 0.5, img, 0.5, 0)
    for pts, kind in lines:
        if kind != "grid":
            color, th = STYLE[kind]
            cv2.polylines(out, [np.round(apply_h(H, pts) * 4).astype(np.int32)], False, color, th, cv2.LINE_AA, shift=2)
    for name, p in (clicked or {}).items():
        cv2.circle(out, tuple(np.round(p).astype(int)), 7, GREEN, 2, cv2.LINE_AA)
    for name, p in (fitted or {}).items():
        c = tuple(np.round(p).astype(int))
        cv2.drawMarker(out, c, MAGENTA, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
    return out


def topdown_transform(table: Table, ppm: float = 250, margin: float = 0.12) -> tuple[np.ndarray, tuple[int, int]]:
    """Table (x, y) -> top-down pixel, far side up, plus the output size."""
    L, W = table.length / 2 + margin, table.width / 2 + margin
    S = np.array([[ppm, 0, L * ppm], [0, -ppm, W * ppm], [0, 0, 1.0]])
    return S, (int(round(2 * L * ppm)), int(round(2 * W * ppm)))


def topdown(img: np.ndarray, H: np.ndarray, table: Table, ppm: float = 250, margin: float = 0.12,
            lines: bool = True) -> np.ndarray:
    """Warp the camera image onto the table plane (table -> image homography H)."""
    S, size = topdown_transform(table, ppm, margin)
    out = cv2.warpPerspective(img, H @ np.linalg.inv(S), size, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    if lines:
        for pts, kind in table_lines(table):
            if kind == "grid":
                continue
            color, th = STYLE[kind]
            cv2.polylines(out, [np.round(apply_h(S, pts) * 4).astype(np.int32)], False, color, th, cv2.LINE_AA, shift=2)
    return out


def table_view(table: Table, size: tuple[int, int], margin: float = 0.12) -> tuple[np.ndarray, float]:
    """Table (x, y) -> pixel for a top-down view filling a (w, h) image, centered, far side up; and its
    pixels per meter."""
    w, h = size
    ppm = min(w / (table.length + 2 * margin), h / (table.width + 2 * margin))
    S, (sw, sh) = topdown_transform(table, ppm, margin)
    return np.array([[1, 0, (w - sw) / 2], [0, 1, (h - sh) / 2], [0, 0, 1.0]]) @ S, ppm


def diagram(table: Table, T: np.ndarray, ppm: float, size: tuple[int, int]) -> np.ndarray:
    """A plain drawing of the table (no footage) in the view T: surface, rails, center line, goal mouths."""
    w, h = size
    img = np.full((h, w, 3), (34, 28, 24), np.uint8)
    L, W = table.length / 2, table.width / 2
    px = lambda pts: np.round(apply_h(T, pts) * 4).astype(np.int32)
    corners = px([[-L, W], [L, W], [L, -W], [-L, -W]])
    cv2.fillPoly(img, [corners], (78, 62, 46), cv2.LINE_AA, shift=2)
    rail = max(3, int(round(0.025 * ppm)))
    cv2.polylines(img, [corners], True, (205, 205, 205), rail, cv2.LINE_AA, shift=2)
    cv2.polylines(img, [px([[0, W], [0, -W]])], False, (140, 140, 140), 2, cv2.LINE_AA, shift=2)
    for s in (-1, 1):
        cv2.ellipse(img, tuple(int(v) for v in apply_h(T, [[s * L, 0]])[0]), (int(0.2 * ppm), int(0.2 * ppm)),
                    0, 90 if s > 0 else -90, 270 if s > 0 else 90, (120, 120, 120), 1, cv2.LINE_AA)
        if table.goal_width:
            g = table.goal_width / 2
            cv2.polylines(img, [px([[s * L, g], [s * L, -g]])], False, (20, 20, 20), rail + 2, cv2.LINE_AA, shift=2)
    return img


def path_points(t: pd.DataFrame, events: pd.DataFrame | None, fps: float) -> tuple:
    """An object's path as (times, xy, thin): one point per frame, plus each impact's contact point at its
    own time (between frames), so lines drawn through them touch the rail where the puck did. thin marks
    points whose segment from the previous point touches a bridged (inferred) stretch."""
    tt, xy = t.frame.to_numpy() / fps, t[["x", "y"]].to_numpy(float)
    thin = np.array(t["filled"], bool) if "filled" in t else np.zeros(len(t), bool)
    if events is not None and len(events):
        seen = events["seen"].to_numpy(bool) if "seen" in events else np.ones(len(events), bool)
        tt, xy, thin = np.r_[tt, events.t], np.r_[xy, events[["x", "y"]].to_numpy(float)], np.r_[thin, ~seen]
        order = np.argsort(tt, kind="stable")
        tt, xy, thin = tt[order], xy[order], thin[order]
    thin[1:] |= thin[:-1]                                 # a segment touching an inferred point is inferred
    return tt, xy, thin
