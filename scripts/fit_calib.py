"""Fit each camera framing from the landmark clicks: homography, camera pose, goal width.

Reads the click docs exported from the landmark page (one JSON per framing, under
data/calib/clicks/) and the median frames from scripts/calib_frames.py. Rail clicks are
snapped to the image edge between the outermost clicks; center line clicks are used as
clicked (it is a painted stripe, not an edge). Writes data/calib/fits.json plus overlay and
top-down images under data/calib/fits/, then prints a summary.
Usage: python scripts/fit_calib.py
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np

from rlairhockey.geometry import (Table, _line_through, apply_h, edge_gradients, fit_camera,
                                  fit_homography_lines, snap_edge, surface_score)
from rlairhockey.render import overlay, topdown

ROOT = Path(__file__).resolve().parents[1]
CLICKS = Path(os.environ.get("CALIB_CLICKS", ROOT / "data/calib/clicks"))
OUT = ROOT / "data/calib/fits"
RAILS = ("far", "near", "left", "right")
SEARCH = float(os.environ.get("CALIB_SEARCH", 6.0))


def load_docs() -> dict[str, dict]:
    docs = {}
    for f in sorted(CLICKS.rglob("*.json")):
        d = json.loads(f.read_text())
        d = d.get("data", d)
        if "frame" in d:
            docs[d["frame"]] = d
    return docs


def line_points(doc: dict, img: np.ndarray) -> tuple[dict, dict]:
    """Points per line for the fit, and how each rail was obtained. Rails snap to where the
    surface color ends (surface_score), on the side away from the table's interior."""
    every = np.array([[p["u"], p["v"]] for ps in doc["lines"].values() for p in ps], np.float64).reshape(-1, 2)
    if len(every) < 3:
        return {}, {}
    hull = cv2.convexHull(every.astype(np.float32)).reshape(-1, 2)
    inside = every.mean(0)
    grads = edge_gradients(surface_score(img, inside + 0.6 * (hull - inside)))
    pts, how = {}, {}
    for k, clicks in doc["lines"].items():
        c = np.array([[p["u"], p["v"]] for p in clicks], np.float64).reshape(-1, 2)
        if len(c) < 2:
            continue
        pts[k], how[k] = c, "clicked"
        if k not in RAILS:
            continue
        ln = _line_through(c)
        d = np.array([-ln[1], ln[0]])
        s = c @ d
        p, q = c[s.argmin()], c[s.argmax()]
        snap = snap_edge(grads, p, q, SEARCH, inside=inside)
        if snap is not None and np.abs(c @ snap["line"][:2] + snap["line"][2]).max() <= SEARCH:
            pts[k] = snap["pts"]
            how[k] = f"snapped ({len(snap['pts'])} edge pts, support {snap['support']}, rms {snap['rms']} px)"
    return pts, how


def main() -> None:
    table = Table.load()
    L, W = table.length / 2, table.width / 2
    OUT.mkdir(parents=True, exist_ok=True)
    fits, goal_widths = {}, []
    for name, doc in load_docs().items():
        if doc.get("status") == "skip":
            print(f"{name}: marked can't tell, skipped"); continue
        img = cv2.imread(str(ROOT / "data/calib/frames" / f"{name}.jpg"))
        pts, how = line_points(doc, img)
        res = fit_homography_lines(table, pts)
        if res is None:
            print(f"{name}: not enough edges ({', '.join(pts) or 'none'})"); continue
        H, rms = res
        Hi = np.linalg.inv(H)
        click_rms = {}
        for k, clicks in doc["lines"].items():
            if len(clicks) >= 1 and k in rms:
                c = np.array([[p["u"], p["v"]] for p in clicks])
                world = apply_h(Hi, c)
                axis, val = (1, W if k == "far" else -W) if k in ("far", "near") else (0, {"left": -L, "right": L, "center": 0.0}[k])
                # distance in px: move each click's table point onto the line and reproject
                on = world.copy(); on[:, axis] = val
                click_rms[k] = round(float(np.sqrt(np.mean(np.sum((apply_h(H, on) - c) ** 2, 1)))), 2)
        anchors = np.array([[-L, W], [L, W], [L, -W], [-L, -W], [0, W], [0, -W]])
        cam = fit_camera(anchors, apply_h(H, anchors), (img.shape[1], img.shape[0]))
        goals = {}
        for side in ("l", "r"):
            a, b = doc["goals"].get(f"{side}_far"), doc["goals"].get(f"{side}_near")
            if a and b:
                wa, wb = apply_h(Hi, [[a["u"], a["v"]], [b["u"], b["v"]]])
                goals[side] = dict(width=round(float(abs(wa[1] - wb[1])), 4),
                                   center_y=round(float((wa[1] + wb[1]) / 2), 4),
                                   off_end=round(float(np.mean(np.abs([wa[0], wb[0]])) - L), 4))
                goal_widths.append(goals[side]["width"])
        clicked = {f"{k}{i}": (p["u"], p["v"]) for k, ps in doc["lines"].items() for i, p in enumerate(ps)}
        clicked |= {f"goal_{k}": (p["u"], p["v"]) for k, p in doc["goals"].items() if p}
        ov = overlay(img, H, table, clicked=clicked, fitted=dict(zip(map(str, range(6)), apply_h(H, anchors))))
        ov = cv2.resize(ov, (1280, 720), interpolation=cv2.INTER_AREA)
        top = topdown(img, H, table, ppm=250)
        paths = {k: OUT / f"{name}_{k}.jpg" for k in ("overlay", "top")}
        cv2.imwrite(str(paths["overlay"]), ov, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(paths["top"]), top, [cv2.IMWRITE_JPEG_QUALITY, 85])
        all_res = np.concatenate([np.full(len(pts[k]), rms[k]) for k in rms])
        fits[name] = dict(video=doc["video"], angle=doc["angle"], H=H.tolist(), line_rms_px={k: round(v, 2) for k, v in rms.items()},
                          click_rms_px=click_rms, edge_rms_px=round(float(np.sqrt(np.mean(all_res ** 2))), 2), lines=how,
                          camera=cam.to_dict() if cam else None,
                          camera_center=[round(float(x), 3) for x in cam.center] if cam else None,
                          goals=goals, goal_width=round(float(np.mean([g["width"] for g in goals.values()])), 4) if goals else None,
                          clicks_updated=doc.get("updated"), check=doc.get("check", ""),
                          overlay=str(paths["overlay"].relative_to(ROOT)), top=str(paths["top"].relative_to(ROOT)))
        f = fits[name]
        print(f"{name}: edge rms {f['edge_rms_px']} px | per line {f['line_rms_px']} | clicks vs fit {click_rms} | "
              f"f {cam.K[0, 0]:.0f} px, camera at {f['camera_center']} m | goals {goals}" if cam else f"{name}: no pose")
    (ROOT / "data/calib/fits.json").write_text(json.dumps(fits, indent=1))
    print(f"{len(fits)} framings fitted -> data/calib/fits.json")
    if goal_widths:
        print(f"goal width from {len(goal_widths)} goals: median {np.median(goal_widths):.4f} m, "
              f"range {min(goal_widths):.4f}-{max(goal_widths):.4f}")


if __name__ == "__main__":
    main()
