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
