"""Cut every play segment out of the raw match video into its own clip.

Reads data/segments/<id>_segments.csv (scripts/prepare_video.py) and writes
data/clips/<id>/<id>_s<seg>_<angle>.mp4 plus data/clips/clips.csv, one row per clip.
Segment bounds are snapped to the source frame grid (the raw videos are constant
frame rate, starting at 0) and each clip is re-encoded to hold exactly those frames:
clip frame k is source frame k0 + k, at source time start + k / fps.
gap_before = seconds since the previous play clip ended (0 means a live camera
switch, so tracks can be stitched across it).
Usage: python scripts/cut_clips.py [--force] [--jobs N] [<id> ...]   (default: all in configs/videos.yaml)
"""
import argparse
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

from rlairhockey.video import probe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/clips"
CRF, PRESET = 16, "fast"


def frame_index(t: float, fps: float) -> int:
    """First source frame at or after time t."""
    return int(math.ceil(t * fps - 1e-3))


def cut(src: Path, dst: Path, k0: int, n: int, fps: float, threads: int) -> None:
    # Seek half a frame early so frame k0 is kept; -frames:v stops after exactly n frames.
    tmp = dst.with_suffix(".part.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(k0 - 0.5, 0) / fps:.6f}", "-i", str(src),
                    "-frames:v", str(n), "-t", f"{(n + 0.5) / fps:.6f}", "-map", "0:v:0", "-map", "0:a:0?",
                    "-c:v", "libx264", "-crf", str(CRF), "-preset", PRESET, "-pix_fmt", "yuv420p",
                    "-fps_mode", "passthrough", "-threads", str(threads),
                    "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(tmp)], check=True)
    tmp.rename(dst)


def n_frames(path: Path) -> int:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                        "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return int(r.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, len(os.sched_getaffinity(0)) // 4))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/videos.yaml"))["videos"]
    threads = max(1, len(os.sched_getaffinity(0)) // a.jobs)

    rows, tasks = [], []
    for vid in a.ids or list(cfg):
        seg_csv = ROOT / "data/segments" / f"{vid}_segments.csv"
        if not seg_csv.exists():
            print(f"{vid}: no segments yet, skipping"); continue
        src = ROOT / "data/raw_videos" / f"{vid}.mp4"
        fps = probe(src).fps
        seg = pd.read_csv(seg_csv, keep_default_na=False)
        play = seg[seg.label == "play"]
        (OUT / vid).mkdir(parents=True, exist_ok=True)
        prev_end = None
        for r in play.itertuples():
            k0, k1 = frame_index(r.start, fps), frame_index(r.end, fps)
            dst = OUT / vid / f"{vid}_s{r.seg:03d}_{r.angle or 'nocam'}.mp4"
            rows.append(dict(clip=dst.stem, video=vid, seg=r.seg, angle=r.angle, fps=fps,
                             k0=k0, n=k1 - k0, start=round(k0 / fps, 4), end=round(k1 / fps, 4),
                             dur=round((k1 - k0) / fps, 3),
                             gap_before=None if prev_end is None else round((k0 - prev_end) / fps, 3),
                             path=str(dst.relative_to(ROOT))))
            prev_end = k1
            if a.force or not dst.exists():
                tasks.append((src, dst, k0, k1 - k0, fps))

    print(f"{len(rows)} play clips, {len(tasks)} to cut ({a.jobs} jobs x {threads} threads)", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        for i, _ in enumerate(ex.map(lambda t: cut(*t, threads), tasks), 1):
            if i % 20 == 0 or i == len(tasks):
                print(f"  cut {i}/{len(tasks)}", flush=True)
        frames = list(ex.map(n_frames, [ROOT / r["path"] for r in rows]))

    man = pd.DataFrame(rows)
    man["frames"] = frames
    old = OUT / "clips.csv"
    if old.exists() and a.ids:  # keep rows for videos not processed this run
        prev = pd.read_csv(old, keep_default_na=False)
        man = pd.concat([prev[~prev.video.isin(man.video)], man], ignore_index=True)
    man.to_csv(old, index=False)
    print(man.groupby("video").agg(clips=("clip", "size"), minutes=("dur", lambda d: round(d.sum() / 60, 1))).to_string())
    off = man[man.frames != man.n]
    print(f"clips whose frame count != n: {len(off)}")
    if len(off):
        print(off[["clip", "n", "frames"]].to_string(index=False))


if __name__ == "__main__":
    main()
