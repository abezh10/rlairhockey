"""Score a trained detector on a dataset's held-out frames: per-class precision, recall and box-center error.

A prediction is a hit when it overlaps a not-yet-matched label of its class with IoU >= 0.5 (highest score
first). Reported at several score thresholds, so models (v1, v2, a Colab run) can be compared on the same
frames and thresholds chosen. Also saved to models/detector/<model>/eval_<dataset>_<split>[_<tag>].csv.
Labels that began as a model's pre-labels favor that model, so --frames can restrict scoring to a list:
data/labels/detector/benchmark_frames.txt holds frames labeled from scratch and held out by v1 and v2.
Usage: python scripts/eval_detector.py [--model v2] [--data data/labels/detector/v2] [--split valid] [--frames FILE]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box against many."""
    ix = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - i)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v2")
    ap.add_argument("--data", default="data/labels/detector/v2")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--resolution", type=int, default=1280)
    ap.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6")
    ap.add_argument("--frames", help="only these frames: a text file of names, or another split's directory")
    a = ap.parse_args()

    import torch
    from rfdetr import RFDETRMedium
    d = ROOT / a.data / a.split
    coco = json.load(open(d / "_annotations.coco.json"))
    names = {c["id"]: c["name"] for c in coco["categories"]}
    if a.frames:
        src = ROOT / a.frames
        keep = ({im["file_name"] for im in json.load(open(src / "_annotations.coco.json"))["images"]} if src.is_dir()
                else {ln.strip() for ln in open(src) if ln.strip() and not ln.startswith("#")})
        coco["images"] = [im for im in coco["images"] if im["file_name"] in keep]
    gt = {im["id"]: [] for im in coco["images"]}
    for an in coco["annotations"]:
        if an["image_id"] not in gt:
            continue
        x, y, w, h = an["bbox"]
        gt[an["image_id"]].append((names[an["category_id"]], [x, y, x + w, y + h]))
    model = RFDETRMedium(pretrain_weights=str(ROOT / "models/detector" / a.model / "checkpoint_best_total.pth"),
                         resolution=a.resolution)
    if torch.cuda.is_available():
        model.inference(compile=False, dtype=torch.float16)
    classes = list(model.class_names)
    preds = {}
    ims = coco["images"]
    for i in range(0, len(ims), 8):
        batch = ims[i:i + 8]
        dets = model.predict([Image.open(d / im["file_name"]).convert("RGB") for im in batch], threshold=0.1,
                             include_source_image=False)
        for im, det in zip(batch, dets if isinstance(dets, list) else [dets]):
            preds[im["id"]] = [(classes[int(c)], float(s), b) for b, c, s in zip(det.xyxy, det.class_id, det.confidence)]

    rows = []
    for thr in map(float, a.thresholds.split(",")):
        for cls in sorted({c for v in gt.values() for c, _ in v}):
            tp = fp = fn = 0
            err = []
            for k, labels in gt.items():
                g = np.array([b for c, b in labels if c == cls]).reshape(-1, 4)
                used = np.zeros(len(g), bool)
                for c, s, b in sorted(preds[k], key=lambda p: -p[1]):
                    if c != cls or s < thr:
                        continue
                    ov = iou(b, g) if len(g) else np.zeros(0)
                    ov[used] = 0
                    if len(ov) and ov.max() >= 0.5:
                        j = int(ov.argmax())
                        used[j] = True
                        tp += 1
                        err.append(np.hypot((b[0] + b[2] - g[j, 0] - g[j, 2]) / 2, (b[1] + b[3] - g[j, 1] - g[j, 3]) / 2))
                    else:
                        fp += 1
                fn += int((~used).sum())
            rows.append(dict(threshold=thr, cls=cls, labels=tp + fn, tp=tp, fp=fp, fn=fn,
                             precision=round(tp / max(tp + fp, 1), 3), recall=round(tp / max(tp + fn, 1), 3),
                             center_err_px=round(float(np.median(err)), 1) if err else np.nan))
    df = pd.DataFrame(rows)
    tag = f"_{Path(a.frames).stem}" if a.frames else ""
    out = ROOT / "models/detector" / a.model / f"eval_{Path(a.data).name}_{a.split}{tag}.csv"
    df.to_csv(out, index=False)
    print(f"{a.model} on {a.data}/{a.split}{' (' + a.frames + ')' if a.frames else ''}, {len(ims)} frames:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
