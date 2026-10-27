"""Pull labels from Roboflow into a COCO dataset for RF-DETR, without needing a Roboflow version.

Reads each labeled image's annotation through the API, keeps only the classes we train on
(puck, mallet, hand; a stray "puck-streak" becomes puck), and splits by camera framing: about
--valid of each framing's frames go to valid, chosen by a hash of the file name so the split
stays put as more frames are labeled. test/ mirrors valid/ (RF-DETR expects all three).
Images are symlinks to data/frames/<set>/.

Writes data/labels/<set>/roboflow_raw.json (the raw pull) and data/labels/<set>/<name>/{train,valid,test}/.
Usage: python scripts/roboflow_pull.py [--set detector] [--name v1] [--valid 0.2]
"""
import argparse
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["hand", "mallet", "puck"]          # Roboflow's alphabetical order; id 0 is the group
ALIASES = {"puck-streak": "puck", "puck_streak": "puck"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="detector")
    ap.add_argument("--project", default="airhockey-objects")
    ap.add_argument("--name", default="v1")
    ap.add_argument("--valid", type=float, default=0.2)
    a = ap.parse_args()
    load_dotenv(ROOT / ".env")
    key, ws = os.environ["ROBOFLOW_API_KEY"], os.environ["ROBOFLOW_WORKSPACE"]

    from roboflow import Roboflow
    project = Roboflow(api_key=key).workspace(ws).project(a.project)
    listed, off = [], 0
    while True:
        page = project.search(offset=off, limit=125, fields=["id", "name", "annotations"])
        listed += page
        off += len(page)
        if len(page) < 125:
            break
    labeled = [x for x in listed if isinstance(x.get("annotations"), dict) and x["annotations"].get("count")]

    def detail(x: dict) -> dict:
        r = requests.get(f"https://api.roboflow.com/{ws}/{a.project}/images/{x['id']}", params={"api_key": key}, timeout=60)
        r.raise_for_status()
        return r.json()["image"]

    with ThreadPoolExecutor(8) as ex:
        raw = list(ex.map(detail, labeled))
    out = ROOT / "data/labels" / a.set
    out.mkdir(parents=True, exist_ok=True)
    (out / "roboflow_raw.json").write_text(json.dumps(raw))

    man = pd.read_csv(ROOT / "data/frames" / a.set / "manifest.csv").set_index("file")
    split = {}
    for group, files in man.loc[[x["name"] for x in raw]].groupby("group").groups.items():
        ranked = sorted(files, key=lambda f: hashlib.md5(f.encode()).hexdigest())
        n_val = round(len(ranked) * a.valid) if len(ranked) >= 3 else 0
        split |= {f: ("valid" if i < n_val else "train") for i, f in enumerate(ranked)}

    cats = [dict(id=0, name="objects", supercategory="none")] + [
        dict(id=i + 1, name=c, supercategory="objects") for i, c in enumerate(CLASSES)]
    cat_id = {c: i + 1 for i, c in enumerate(CLASSES)}
    coco = {s: dict(images=[], annotations=[], categories=cats) for s in ("train", "valid")}
    dropped, counts = Counter(), {s: Counter() for s in coco}
    for img_id, im in enumerate(raw):
        s = split[im["name"]]
        ann = im["annotation"]
        coco[s]["images"].append(dict(id=img_id, file_name=im["name"], width=ann["width"], height=ann["height"]))
        for b in ann.get("boxes", []):
            label = ALIASES.get(b["label"], b["label"])
            if label not in cat_id:
                dropped[b["label"]] += 1
                continue
            cx, cy, w, h = (float(b[k]) for k in ("x", "y", "width", "height"))
            seg = [[float(v) for pt in b["points"] for v in pt]] if b.get("points") else []
            coco[s]["annotations"].append(dict(id=len(coco[s]["annotations"]), image_id=img_id, category_id=cat_id[label],
                                               bbox=[cx - w / 2, cy - h / 2, w, h], area=w * h, iscrowd=0, segmentation=seg))
            counts[s][label] += 1

    ds = out / a.name
    frames = ROOT / "data/frames" / a.set
    for s, dirs in (("train", ["train"]), ("valid", ["valid", "test"])):
        for d in dirs:
            (ds / d).mkdir(parents=True, exist_ok=True)
            for im in coco[s]["images"]:
                link = ds / d / im["file_name"]
                if not link.exists():
                    link.symlink_to(frames / im["file_name"])
            (ds / d / "_annotations.coco.json").write_text(json.dumps(coco[s]))
    for s in coco:
        print(f"{s}: {len(coco[s]['images'])} images, boxes {dict(counts[s])}")
    if dropped:
        print(f"dropped boxes with other labels: {dict(dropped)}")
    print(f"-> {ds}")


if __name__ == "__main__":
    main()
