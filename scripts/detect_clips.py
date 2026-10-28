"""Run the detector over every play clip and save every box above a low score for the tracker.

Writes data/detections/<model>/<clip>.parquet, one row per box: frame (clip frame k, i.e. source frame
k0 + k), cls (mallet / puck), score, x0, y0, x1, y1 in source pixels. The score floor is low on purpose:
the tracker decides what is real (on the playing surface, moving, one puck, one mallet per side).
Finished clips are skipped, so reruns resume; --shard i/n splits the clips across GPU jobs.
Usage: python scripts/detect_clips.py [--model v2] [--shard 0/4] [--clips <clip> ...]
"""
import argparse
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def frames(path: Path, batch: int):
    """Yield lists of RGB frames, decoding in a background thread so the GPU doesn't wait on ffmpeg."""
    q: queue.Queue = queue.Queue(maxsize=4)

    def read() -> None:
        cap = cv2.VideoCapture(str(path))
        buf = []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            buf.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            if len(buf) == batch:
                q.put(buf)
                buf = []
        if buf:
            q.put(buf)
        q.put(None)
        cap.release()

    threading.Thread(target=read, daemon=True).start()
    while (b := q.get()) is not None:
        yield b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v2", help="models/detector/<model>/checkpoint_best_total.pth")
    ap.add_argument("--resolution", type=int, default=1280)
    ap.add_argument("--keep", type=float, default=0.1, help="lowest score saved")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--shard", default="0/1", help="i/n: this job takes every n-th clip starting at i")
    ap.add_argument("--clips", nargs="*", help="only these clips")
    ap.add_argument("--fp32", action="store_true", help="skip the float16 cast on GPU")
    a = ap.parse_args()

    import torch
    from rfdetr import RFDETRMedium
    clips = pd.read_csv(ROOT / "data/clips/clips.csv", keep_default_na=False)
    if a.clips:
        clips = clips[clips["clip"].isin(a.clips)]
    i, n = map(int, a.shard.split("/"))
    clips = clips.iloc[i::n]
    out = ROOT / "data/detections" / a.model
    out.mkdir(parents=True, exist_ok=True)
    todo = [r for r in clips.itertuples() if not (out / f"{r[clips.columns.get_loc('clip') + 1]}.parquet").exists()]
    print(f"shard {a.shard}: {len(clips)} clips, {len(todo)} to do", flush=True)
    if not todo:
        return

    model = RFDETRMedium(pretrain_weights=str(ROOT / "models/detector" / a.model / "checkpoint_best_total.pth"),
                         resolution=a.resolution)
    if torch.cuda.is_available():
        model.inference(compile=False, dtype=torch.float32 if a.fp32 else torch.float16)
    names = np.array(model.class_names)   # predicted class_id indexes this list
    for r in todo:
        clip = r[clips.columns.get_loc("clip") + 1]       # itertuples can't name a column "clip" (DataFrame.clip)
        t0, rows, k = time.time(), [], 0
        for b in frames(ROOT / r.path, a.batch):
            dets = model.predict(b, threshold=a.keep, include_source_image=False)
            for d in (dets if isinstance(dets, list) else [dets]):
                if len(d):
                    rows.append(pd.DataFrame(dict(frame=k, cls=names[d.class_id], score=d.confidence.astype(np.float32),
                                                  x0=d.xyxy[:, 0], y0=d.xyxy[:, 1], x1=d.xyxy[:, 2], y1=d.xyxy[:, 3])))
                k += 1
        if k != r.frames:
            print(f"  {clip}: decoded {k} frames, clips.csv says {r.frames}", flush=True)
        df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["frame", "cls", "score", "x0", "y0", "x1", "y1"])
        df = df.astype(dict(frame="int32", x0="float32", y0="float32", x1="float32", y1="float32"))
        tmp = out / f"{clip}.part.parquet"
        df.to_parquet(tmp, index=False)
        tmp.rename(out / f"{clip}.parquet")
        dt = time.time() - t0
        print(f"  {clip}: {k} frames, {len(df)} boxes, {k / dt:.1f} fps", flush=True)


if __name__ == "__main__":
    main()
