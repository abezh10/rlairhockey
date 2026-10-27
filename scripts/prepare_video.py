"""Label every stretch of a match video: intro, play (+camera angle), idle, replay,
cutaway, break, outro. Reads data/features/<id>.* (scripts/extract_features.py).

Writes data/segments/<id>_segments.csv and <id>_shots.csv.
Usage: python scripts/prepare_video.py [<id> ...]   (default: all in configs/videos.yaml)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rlairhockey.prep import segment_video

ROOT = Path(__file__).resolve().parents[1]


def fmt(s: float) -> str:
    s = round(s, 1)
    return f"{int(s // 60)}:{s % 60:04.1f}"


def main(ids: list[str]) -> None:
    cfg = yaml.safe_load(open(ROOT / "configs/videos.yaml"))["videos"]
    out = ROOT / "data/segments"
    out.mkdir(parents=True, exist_ok=True)
    for vid in ids or list(cfg):
        feat = ROOT / "data/features" / f"{vid}.parquet"
        if not feat.exists():
            print(f"{vid}: no features yet, skipping"); continue
        df = pd.read_parquet(feat)
        tiny = np.load(ROOT / "data/features" / f"{vid}_tiny.npy")
        grid = np.load(ROOT / "data/features" / f"{vid}_grid.npy")
        c = cfg.get(vid, {})
        seg, shots = segment_video(df, tiny, grid, ROOT / "data/proxy" / f"{vid}.mp4",
                                   c.get("game_start"), c.get("game_end"),
                                   clip_cache=out / f"{vid}_clip.parquet")
        seg.insert(0, "video", vid)
        seg["start_hms"] = seg.start.map(fmt)
        seg["end_hms"] = seg.end.map(fmt)
        seg.to_csv(out / f"{vid}_segments.csv", index_label="seg")
        shots.to_csv(out / f"{vid}_shots.csv", index=False)
        summ = seg.groupby("label").dur.sum().div(60).round(1).to_dict()
        angles = seg[seg.label == "play"].groupby("angle").dur.sum().div(60).round(1).to_dict()
        print(f"{vid} ({c.get('name','')}): {len(seg)} segments, {len(shots)} shots | minutes {summ} | play by angle {angles}")


if __name__ == "__main__":
    main(sys.argv[1:])
