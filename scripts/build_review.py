"""Build the segment review page: outputs/review/index.html + thumbs/<id>.jpg sprites.

Each segment gets three thumbnails (just after start, middle, just before end) from the
320x180 proxy, packed into one sprite sheet per video.
Usage: python scripts/build_review.py [<id> ...]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from rlairhockey.video import iter_frames, probe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/review"
TW, TH, COLS = 160, 90, 30


def sprite_for(vid: str, times: np.ndarray) -> np.ndarray:
    """Grab the first proxy frame at or after each requested time, in one sequential pass."""
    order = np.argsort(times)
    tiles: list[np.ndarray | None] = [None] * len(times)
    k = 0
    last = None
    for t, f in iter_frames(ROOT / "data/proxy" / f"{vid}.mp4", width=320):
        last = f
        while k < len(order) and t >= times[order[k]]:
            tiles[order[k]] = cv2.resize(f, (TW, TH), interpolation=cv2.INTER_AREA)
            k += 1
        if k == len(order):
            break
    for i in range(len(tiles)):
        if tiles[i] is None:
            tiles[i] = cv2.resize(last, (TW, TH), interpolation=cv2.INTER_AREA)
    rows = -(-len(tiles) // COLS)
    sheet = np.zeros((rows * TH, COLS * TW, 3), np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, COLS)
        sheet[r * TH:(r + 1) * TH, c * TW:(c + 1) * TW] = tile
    return sheet


def main(ids: list[str]) -> None:
    cfg = yaml.safe_load(open(ROOT / "configs/videos.yaml"))["videos"]
    (OUT / "thumbs").mkdir(parents=True, exist_ok=True)
    data_path = OUT / "data.json"
    data = json.loads(data_path.read_text()) if data_path.exists() else {"videos": []}
    keep = {v["id"]: v for v in data["videos"]}
    for vid in ids or list(cfg):
        f = ROOT / "data/segments" / f"{vid}_segments.csv"
        if not f.exists():
            print(f"{vid}: no segments yet, skipping"); continue
        seg = pd.read_csv(f)
        info = probe(ROOT / "data/raw_videos" / f"{vid}.mp4")
        times = []
        for s, e in zip(seg.start, seg.end):
            pad = min(0.3, (e - s) / 4)
            times += [s + pad, (s + e) / 2, e - pad]
        sheet = sprite_for(vid, np.array(times))
        cv2.imwrite(str(OUT / "thumbs" / f"{vid}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 72])
        c = cfg.get(vid, {})
        keep[vid] = dict(
            id=vid, name=c.get("name", vid), fps=round(info.fps, 2), duration=round(info.duration, 1),
            game_start_cfg=c.get("game_start"), game_end_cfg=c.get("game_end"), sprite=f"thumbs/{vid}.jpg", cols=COLS, tw=TW, th=TH,
            segs=[dict(i=int(r.seg), s=round(r.start, 2), e=round(r.end, 2), label=r.label,
                       angle=r.angle if isinstance(r.angle, str) else "", act=round(float(r.activity), 4))
                  for r in seg.itertuples()],
        )
        print(f"{vid}: {len(seg)} segments, sprite {sheet.shape[1]}x{sheet.shape[0]}")
    data["videos"] = [keep[v] for v in cfg if v in keep]
    data_path.write_text(json.dumps(data, separators=(",", ":")))
    page = (ROOT / "scripts/review_template.html").read_text()
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    (OUT / "index.html").write_text(page.replace("/*DATA*/", blob))
    print("wrote", OUT / "index.html")


if __name__ == "__main__":
    main(sys.argv[1:])
