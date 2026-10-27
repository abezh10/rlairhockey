# rlairhockey

Edited air hockey match videos -> top-down puck/mallet trajectories -> imitation learning / RL.
Full plan: [docs/PLAN.md](docs/PLAN.md).

## Setup
```bash
source env.sh                     # venv + ffmpeg on PATH
# reinstall from scratch: ~/.local/bin/uv venv --python 3.12 .venv && ~/.local/bin/uv pip install -e ".[dev]"
```
Login node has no GPU. Train/infer on the Slurm `gpu` partition (NVIDIA L40S, 46 GB):
`srun -p gpu --gres=gpu:1 -t 2:00:00 --pty bash`

The home quota is small, so bulky data that can be regenerated lives in `/scratch/users/$USER/rlairhockey` and is
symlinked into place: `data/raw_videos`, `data/clips`, `data/proxy`, `models/detector`. On a fresh checkout, create
those directories in scratch and link them the same way before downloading or training.

## Download footage
Put one YouTube URL per line in a text file, then:
```bash
scripts/download_videos.sh links.txt
```
Saves to `data/raw_videos/<id>.mp4` + `<id>.info.json`, capped at 1080p with the highest fps available.
`data/raw_videos/archive.txt` makes reruns skip finished videos.

## Layout
| Path | Contents |
|---|---|
| `src/rlairhockey/` | package: geometry, video, detect, track, stitch, phases, dataset, render |
| `scripts/` | CLI entry points |
| `configs/` | table geometry, model and pipeline settings |
| `data/raw_videos` | downloaded matches |
| `data/frames`, `data/labels/{detector,keypoints}` | frames sampled for labeling and their COCO labels |
| `data/calib`, `data/tracks`, `data/segments`, `data/datasets` | per-frame homographies, Kalman tracks, segment/QC table, final episodes |
| `models/` | RF-DETR checkpoints |
| `outputs/renders`, `outputs/qc` | top-down debug videos, QC reports |

## Video prep (intro / play / breaks / replays / camera angles)
```bash
sbatch scripts/slurm/proxy.sbatch <id> ...      # 320x180 analysis copies -> data/proxy/
sbatch scripts/slurm/features.sbatch <id> ...   # per-frame signals -> data/features/
sbatch scripts/slurm/prepare.sbatch [<id> ...]  # segments -> data/segments/<id>_segments.csv, review page -> outputs/review/
```
Game start (faceoff release) and end (final goal) per video live in `configs/videos.yaml`; they were
found by eye with `scripts/contact_sheet.py <id> <t0> <t1> <step> <out.jpg> [proxy|raw]` (30 s pass, then finer).
Logic: `src/rlairhockey/prep.py`: hard cuts plus soft cuts/dissolves, camera groups by appearance
("camN" = one fixed framing, so a zoomed version of a camera is its own group), CLIP zero-shot shot type,
table region per camera, replay = earlier footage re-shown in order or a package bracketed by a repeated
replay graphic ("INSTANT REPLAY", "OFF SPEED REPLAY"). Play with no camera = table footage from a rare angle.

## Play clips
```bash
sbatch scripts/slurm/clips.sbatch [--force] [<id> ...]   # -> data/clips/<id>/<id>_s<seg>_<camN>.mp4 + data/clips/clips.csv
```
One clip per play segment, re-encoded (x264 CRF 16) so it holds exactly the segment's source frames:
clip frame k = source frame `k0 + k`, at source time `start + k / fps`. `gap_before` = 0 marks a live
camera switch (tracks continue across it); a positive gap means the edit skipped time.

## Table geometry (plan step 1)
Table frame: origin at the center, x along the long axis, y across, z up, meters (`configs/table.yaml`).
Each camera framing is calibrated once, since the broadcast cameras are fixed:
```bash
python scripts/calib_frames.py        # median frame per framing (players blur out) -> data/calib/frames/
python scripts/build_calib_page.py    # click page -> outputs/calib/ (published as an artifact, clicks stored in its db)
python scripts/fit_calib.py           # clicks exported to data/calib/clicks/ -> data/calib/fits.json + overlay/top-down images
```
On the click page you mark points along each rail edge and the center line, plus the goal posts. Left, right and
far are as that camera sees the table. Rail clicks snap to where the surface color ends. The homography is fitted to
every point's distance from its line, so corners can be off-frame. Camera pose (`geometry.fit_camera`) assumes square
pixels and a centered principal point; `Camera.to_table(uv, z=h)` intersects rays with the plane at mallet height.

## Frames for labeling (plan step 2)
```bash
python scripts/sample_frames.py --n 500   # -> data/frames/detector/*.jpg + manifest.csv
```
Spread over videos and framings (square root of play time each), weighted toward motion on the table so fast pucks show up.

Upload to Roboflow (credentials in `.env`, git-ignored; copy `.env.example`, and `env.sh` exports them):
```bash
python scripts/roboflow_upload.py --dry-run   # checks files and credentials
python scripts/roboflow_upload.py             # -> project airhockey-objects, tagged by video and framing; resumable
```
