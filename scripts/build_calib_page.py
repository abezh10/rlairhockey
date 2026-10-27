"""Build the landmark click page: outputs/calib/index.html + frames/, thumbs/ and results/ images.

Frames come from scripts/calib_frames.py; fit results (overlay and top-down images, error
stats) from scripts/fit_calib.py, shown under each framing when present.
Usage: python scripts/build_calib_page.py
"""
import json
import shutil
from pathlib import Path

import cv2
import pandas as pd
import yaml

from rlairhockey.geometry import Table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/calib"


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs/videos.yaml"))["videos"]
    frames = pd.read_csv(ROOT / "data/calib/frames/frames.csv")
    for d in ("frames", "thumbs"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    rows = []
    for r in frames.itertuples():
        img = cv2.imread(str(ROOT / "data/calib/frames" / f"{r.frame}.jpg"))
        cv2.imwrite(str(OUT / "frames" / f"{r.frame}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(str(OUT / "thumbs" / f"{r.frame}.jpg"), cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 75])
        rows.append(dict(name=r.frame, video=r.video, video_name=cfg[r.video]["name"], angle=r.angle,
                         play_min=float(r.play_min), w=int(r.width), h=int(r.height),
                         src=f"frames/{r.frame}.jpg", thumb=f"thumbs/{r.frame}.jpg"))
    results = {}
    fits = ROOT / "data/calib/fits.json"
    if fits.exists():
        (OUT / "results").mkdir(exist_ok=True)
        for name, f in json.loads(fits.read_text()).items():
            for k in ("overlay", "top"):
                shutil.copy(ROOT / f[k], OUT / "results" / Path(f[k]).name)
            cam = f.get("camera")
            results[name] = dict(rms_px=f["edge_rms_px"], f_px=cam["K"][0][0] if cam else None,
                                 cam=f.get("camera_center"), goal_width=f.get("goal_width"),
                                 clicks_updated=f.get("clicks_updated"),
                                 overlay=f"results/{Path(f['overlay']).name}", top=f"results/{Path(f['top']).name}")
    t = Table.load()
    data = dict(table=dict(length=t.length, width=t.width), frames=rows, results=results)
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    (OUT / "index.html").write_text((ROOT / "scripts/calib_template.html").read_text().replace("/*DATA*/", blob))
    print(f"wrote {OUT / 'index.html'}: {len(rows)} framings, {len(results)} fits")


if __name__ == "__main__":
    main()
