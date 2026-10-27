"""Upload sampled frames (scripts/sample_frames.py) to a Roboflow project for labeling.

Credentials come from .env (see .env.example): ROBOFLOW_API_KEY (the private key) and
ROBOFLOW_WORKSPACE. Creates the project if it doesn't exist. Each image is tagged with its
video id and camera framing, so you can filter by them in Roboflow. Uploaded files are
recorded in data/frames/<set>/uploaded.json, so a rerun only sends what's missing.
Usage: python scripts/roboflow_upload.py [--set detector] [--project airhockey-objects] [--dry-run]
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="detector")
    ap.add_argument("--project", default="airhockey-objects")
    ap.add_argument("--type", default="object-detection")
    ap.add_argument("--annotation", default="puck, puck_streak, mallet, hand")
    ap.add_argument("--batch", default=None, help="Roboflow batch name (default: the frame set)")
    ap.add_argument("--license", default="Private", help='project license; free (public) workspaces need e.g. "CC BY 4.0"')
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="check files and credentials, upload nothing")
    a = ap.parse_args()

    load_dotenv(ROOT / ".env")
    key, ws_name = os.environ.get("ROBOFLOW_API_KEY", ""), os.environ.get("ROBOFLOW_WORKSPACE", "")
    if not key or not ws_name:
        raise SystemExit("Set ROBOFLOW_API_KEY (the private key) and ROBOFLOW_WORKSPACE in .env")

    frames = ROOT / "data/frames" / a.set
    man = pd.read_csv(frames / "manifest.csv")
    done_path = frames / "uploaded.json"
    done = json.loads(done_path.read_text()) if done_path.exists() else {}
    todo = man[~man.file.isin(done)]
    missing = [f for f in todo.file if not (frames / f).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} frames in the manifest are missing on disk, e.g. {missing[0]}")
    print(f"{len(man)} frames in {frames}, {len(done)} already uploaded, {len(todo)} to go")

    from roboflow import Roboflow
    ws = Roboflow(api_key=key).workspace(ws_name)
    existing = {p.split("/")[-1] for p in ws.projects()}   # ids look like "<workspace>/<project>"
    print(f"workspace {ws_name}: projects {sorted(existing) or 'none'}")
    if a.dry_run:
        return
    if a.project in existing:
        project = ws.project(a.project)
    else:
        project = ws.create_project(project_name=a.project, project_type=a.type, project_license=a.license,
                                    annotation=a.annotation)
        print(f"created project {project.id}")

    def send(r) -> tuple[str, str | None]:
        try:
            project.upload(image_path=str(frames / r.file), batch_name=a.batch or a.set, split="train",
                           tag_names=[r.video, r.group], num_retry_uploads=2)
            return r.file, None
        except Exception as e:  # keep going; failures are retried on the next run
            return r.file, str(e)

    errors = 0
    with ThreadPoolExecutor(a.jobs) as ex:
        for i, (f, err) in enumerate(ex.map(send, todo.itertuples()), 1):
            if err:
                errors += 1
                print(f"  {f}: {err}")
            else:
                done[f] = True
            if i % 50 == 0 or i == len(todo):
                done_path.write_text(json.dumps(done, indent=0))
                print(f"  {i}/{len(todo)} sent, {errors} failed", flush=True)
    print(f"{len(done)}/{len(man)} frames uploaded to {ws_name}/{a.project}" + (f"; rerun to retry {errors}" if errors else ""))


if __name__ == "__main__":
    main()
