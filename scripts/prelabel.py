"""Pre-label frames with a trained detector and put the boxes in Roboflow for correction.

  predict  runs the model on every frame in data/frames/<set>/ (compute node; no network) and caches
           every box above --keep in data/labels/<set>/prelabel_<model>.json, with scores.
  upload   (login node) lists the Roboflow project and, for each image that still has no annotation, tags
           it --tag and then saves the boxes scoring at least that class's --threshold. It never overwrites: it
           skips images that already have boxes (checked per image, since Roboflow's search index lags), and if
           you label one between that check and the save, Roboflow refuses with "already annotated" and the tag
           comes back off.
           Images stay in their labeling batch, so they don't join the Dataset by themselves.

Review in Roboflow: filter images by the tag, fix the boxes, then remove the tag. scripts/roboflow_pull.py
skips anything still tagged, so unreviewed model boxes never become training labels.
Usage: python scripts/prelabel.py predict [--model v1]
       python scripts/prelabel.py upload [--model v1] [--threshold puck=0.2,mallet=0.3,hand=0.5] [--dry-run]
"""
import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def predict(a: argparse.Namespace, frames: Path, out: Path) -> None:
    from PIL import Image
    from rfdetr import RFDETRMedium
    cache = json.loads(out.read_text()) if out.exists() else {}
    todo = [f for f in pd.read_csv(frames / "manifest.csv").file if f not in cache]
    print(f"{len(cache)} frames cached, {len(todo)} to predict", flush=True)
    model = RFDETRMedium(pretrain_weights=str(ROOT / "models/detector" / a.model / "checkpoint_best_total.pth"),
                         resolution=a.resolution)
    names = model.class_names       # predicted class_id indexes this list (COCO category id - 1)
    for i in range(0, len(todo), a.batch):
        files = todo[i:i + a.batch]
        images = [Image.open(frames / f).convert("RGB") for f in files]
        dets = model.predict(images, threshold=a.keep)
        for f, im, d in zip(files, images, dets if isinstance(dets, list) else [dets]):
            cache[f] = dict(width=im.width, height=im.height, boxes=[
                dict(label=names[int(c)], score=round(float(s), 3), xyxy=[round(float(v), 1) for v in b])
                for b, c, s in zip(d.xyxy, d.class_id, d.confidence)])
        out.write_text(json.dumps(cache))
        print(f"  {min(i + a.batch, len(todo))}/{len(todo)}", flush=True)
    print(f"-> {out}")


def voc(name: str, width: int, height: int, boxes: list[dict]) -> str:
    objs = "".join(
        f"<object><name>{b['label']}</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult>"
        f"<bndbox><xmin>{b['xyxy'][0]}</xmin><ymin>{b['xyxy'][1]}</ymin><xmax>{b['xyxy'][2]}</xmax><ymax>{b['xyxy'][3]}</ymax>"
        f"</bndbox></object>" for b in boxes)
    return (f"<annotation><filename>{name}</filename><size><width>{width}</width><height>{height}</height>"
            f"<depth>3</depth></size>{objs}</annotation>")


def upload(a: argparse.Namespace, out: Path) -> None:
    from roboflow import Roboflow
    from roboflow.adapters import rfapi
    load_dotenv(ROOT / ".env")
    key, ws = os.environ["ROBOFLOW_API_KEY"], os.environ["ROBOFLOW_WORKSPACE"]
    preds = json.loads(out.read_text())
    project = Roboflow(api_key=key).workspace(ws).project(a.project)
    listed, off = [], 0
    while True:
        page = project.search(offset=off, limit=125, fields=["id", "name", "annotations", "tags"])
        listed += page
        off += len(page)
        if len(page) < 125:
            break
    blank = [x for x in listed if not (isinstance(x.get("annotations"), dict) and x["annotations"].get("count"))]
    todo = [x for x in blank if x["name"] in preds]
    if a.limit:
        todo = todo[:a.limit]
    thr = {k: float(v) for k, v in (kv.split("=") for kv in a.threshold.split(","))}
    boxes = {x["name"]: [b for b in preds[x["name"]]["boxes"] if b["score"] >= thr[b["label"]]] for x in todo}
    n_box = Counter(b["label"] for bs in boxes.values() for b in bs)
    print(f"{len(listed)} images in {a.project}, {len(blank)} without labels, {len(todo)} to pre-label "
          f"with boxes {dict(n_box)} at score >= {thr}")
    if a.dry_run:
        return

    def call(fn, *args, **kw):
        # Roboflow answers bursts with "Too Many Requests"; back off and retry.
        for i in range(7):
            try:
                return fn(*args, **kw)
            except Exception as e:
                if i == 6 or not ("Too Many Requests" in str(e) or "429" in str(e)):
                    raise
                time.sleep(2 ** i)

    def detail(image_id: str) -> dict:
        r = requests.get(f"https://api.roboflow.com/{ws}/{a.project}/images/{image_id}", params={"api_key": key}, timeout=60)
        if r.status_code == 429:
            raise RuntimeError("Too Many Requests")
        r.raise_for_status()
        return r.json()["image"]

    def send(x: dict) -> tuple[str, str]:
        try:
            # The search index lags behind, so check the image itself: skip anything with boxes,
            # whether you labeled it or an earlier run pre-labeled it.
            im = call(detail, x["id"])
            if (im.get("annotation") or {}).get("boxes"):
                return x["name"], "has boxes"
            if a.tag not in (im.get("tags") or []):
                call(rfapi.update_image_metadata, key, ws, x["id"], add_tags=[a.tag])   # tag first: never an untagged pre-label
            p = preds[x["name"]]
            res, _, _ = call(project.save_annotation, annotation_path=dict(
                name=Path(x["name"]).stem + ".xml", rawText=voc(x["name"], p["width"], p["height"], boxes[x["name"]])),
                image_id=x["id"], annotation_overwrite=False, add_to_dataset=False)
            if res.get("warn"):     # "already annotated": you labeled it since the check, so it isn't ours to tag
                call(rfapi.update_image_metadata, key, ws, x["id"], remove_tags=[a.tag])
                return x["name"], "has boxes"
            return x["name"], "ok"
        except Exception as e:      # a tag left on a still-blank image is harmless; a rerun finishes it
            return x["name"], f"failed ({e})"

    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(send, todo))
    for f, r in results:
        if r.startswith("failed"):
            print(f"  {f}: {r}")
    print(f"{dict(Counter(r.split(' (')[0] for _, r in results))}; pre-labeled images are tagged '{a.tag}'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["predict", "upload"])
    ap.add_argument("--set", default="detector")
    ap.add_argument("--project", default="airhockey-objects")
    ap.add_argument("--model", default="v1", help="models/detector/<model>/checkpoint_best_total.pth")
    ap.add_argument("--resolution", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--keep", type=float, default=0.2, help="predict: cache boxes scoring at least this")
    # v1 on the 22 valid frames: puck 0.2 finds one more puck at no extra false boxes, mallet is clean from
    # 0.3, and half of the hands under 0.5 are hands that weren't labeled.
    ap.add_argument("--threshold", default="puck=0.2,mallet=0.3,hand=0.5", help="upload: per-class minimum score")
    ap.add_argument("--tag", default="prelabel")
    ap.add_argument("--limit", type=int, default=0, help="upload: only this many images (0 = all)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = ROOT / "data/labels" / a.set / f"prelabel_{a.model}.json"
    if a.step == "predict":
        predict(a, ROOT / "data/frames" / a.set, out)
    else:
        upload(a, out)


if __name__ == "__main__":
    main()
