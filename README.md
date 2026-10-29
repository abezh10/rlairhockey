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

Pre-label the rest with a trained model, then review in Roboflow:
```bash
sbatch scripts/slurm/prelabel.sbatch            # predictions on CPU -> data/labels/detector/prelabel_<model>.json
python scripts/prelabel.py upload               # boxes into Roboflow, each image tagged "prelabel"
```
Review: filter by the tag, fix the boxes, remove the tag. Frames with nothing visible can be skipped.

## Detector (plan step 2b)
Classes: `puck` (blurred pucks too) and `mallet`. Hands and puck streaks were dropped (2026-10-27).
```bash
python scripts/roboflow_pull.py --name v2                      # reviewed labels -> data/labels/detector/v2 (COCO)
sbatch scripts/slurm/train_detector.sbatch --data data/labels/detector/v2 --name v2 --epochs 50
```
RF-DETR Medium at 1280 px (pucks are 15-50 px wide). v1 (112 frames, 12 min on one L40S): AP50:95 puck 0.81,
mallet 0.81 on 22 held-out frames.

## Detection and tracking (plan steps 4-5)
```bash
sbatch --array=0-3 scripts/slurm/detect_clips.sbatch --model v2   # every box >= 0.1 -> data/detections/v2/<clip>.parquet
python scripts/find_repeats.py                                    # repeated frames -> data/clips/changes/<clip>.npy
python scripts/track_clips.py --model v2                          # -> data/tracks/v2/<clip>.parquet, events/<clip>.parquet, stats.csv
python scripts/fit_walls.py --model v2                            # rails to where the puck bounces -> data/calib/walls.json
python scripts/render_tracks.py <clip> --model v2                 # top-down picture -> outputs/tracks/v2/<clip>.png
python scripts/render_video.py <clip> --model v2 --seconds 60     # 2 x 2 video -> outputs/videos/v2/<clip>.mp4
python scripts/score_changes.py                                   # scoreboard digits -> data/clips/score/<clip>.npz
python scripts/check_goals.py --model v2                          # detected goals vs the scoreboard
```
`scripts/slurm/track_walls.sh <model>` runs tracking and the wall fit together (track, fit, track, fit, track).

The tracker (`src/rlairhockey/track.py`, settings in `configs/track.yaml`) keeps one puck and one mallet per
table half. It drops detections off the playing surface, then parked spares: a puck or mallet that sits
still for most of a clip while another candidate of its kind is in view. A lone resting puck is never dropped.
Each target's path is then picked by dynamic programming over detection scores, step lengths and gaps, and
smoothed with a constant-velocity Kalman/RTS pass. Tracks are in each clip's camera frame for now.

The puck's velocity jumps at impacts, and smoothing straight through one rounds the corner and blends the
velocities in and out over several frames (about 30% of the puck's speed off within two frames of a bounce). So
the puck's path is first split where its motion stops being smooth, each split is checked for a cause (a mallet
within reach, else a rail or end wall the puck moved into and away from, else a large unexplained change), and the
pieces between impacts are smoothed separately and extrapolated to meet at the impact. Each impact (time, contact
point, velocities in and out, cause, and for bounces the restitution and the contact's distance past the rail
line) goes to `data/tracks/<model>/events/<clip>.parquet`, and the track's `event` column marks the first frame
after it.

Gaps in the puck's path that are too long for the smoother (an arm over the puck, a fast puck missed for a few
frames) are bridged up to `fill.max` (1 s) by the straight-line path that best matches its motion on both sides:
no corner, one bounce on a wall, or one impact. Bridged frames have `filled` set, and an impact inferred inside a
gap has `seen` false, so a dataset can mask or down-weight them.

Some broadcasts show frames twice. The 2024 final is 24 fps footage in a 60 fps file, with no fixed pattern of
which pictures are doubled, and the 2023 final now and then repeats a frame and then skips one. Taken at face value
this makes the puck stop for a frame and then jump, and in the 2024 final its speed see-saw by a factor of two.
`find_repeats.py` measures how much each frame changed, the tracker drops repeats, and `retime()` gives each
picture the time it was taken: evenly spaced for pulldown, and moved back a frame between a repeat and the
skip that makes up for it (found from the puck's steps) for the other kind.

The tracks are then held to what a puck can do:
- Between impacts the puck slides straight at a nearly steady speed, so the path is split into straight-line
  pieces (any real change of velocity starts a new one), and a piece no mallet touches is smoothed with a small
  allowance for acceleration (`smooth.accel_sigma.puck_free`). A piece a mallet is pushing gets the usual one.
- The pieces meet at each impact's contact point, so the path never jumps.
- `fit_walls.py` corrects each camera framing's calibration from its bounces: a bounce off a wall puts the puck's
  center one radius from it, so the contacts off each wall trace a line, and a homography on the table plane
  moves those lines to where they belong (offsets of up to 5 cm before; under 1 cm after two passes).
- Last, `constrain()` keeps every center at least a radius inside the rails (the puck may go on into a goal mouth)
  and pushes the puck and a mallet apart when they overlap, the mallet (the less certain of the two) by more.
  The mallet's box-center height is 3 cm: at 4 cm, puck-mallet distances at strikes depended on which side of
  the mallet the camera saw the puck.
Over all 186 tracked clips this took the puck past a rail from 4,226 frames to 51, overlapping a mallet from 6,922
to 16, accelerating hard (over 10 m/s^2) in free flight from 15,728 frames to 973, and unexplained impacts from
1,505 to 738.

`render_video.py` shows four views: the footage with every box and each track drawn back in; the top-down table
(warped footage) with trails and impacts; a plain diagram of the table with only the puck (and its velocity) and
the mallets; and a live graph of the last 4 s of positions along and across the table and the puck's speed, with
its impacts as ticks.

A fast shot off a rail is often seen only once or twice either side of the bounce, too few to split the path
there, so the tracker also tries a single hidden bounce in short stretches (`hidden_bounce()`: a straight line
out to the wall and back), and where a repeated frame leaves a picture's time uncertain it tries both times.

### Game events
`track_clips.py` also runs `src/rlairhockey/game.py` on each clip: data/tracks/<model>/game/<clip>.parquet lists
goals, pickups (the puck taken off the table without a goal: a timeout, or retrieving a stopped puck), serves,
shots (with their outcome: goal, save or miss), center-line crossings, and where the puck was lost, put down or
found; the tracks' `phase` column says, per frame, whether play is live, in setup (puck down, not yet served),
dead (after a goal or pickup) or unseen. A goal is a puck whose last motion carries it, off the rails, into a
goal mouth with no mallet in the way. `check_goals.py` checks goals against the broadcast's own scoreboard:
`score_changes.py` crops each match's points digits (`configs/videos.yaml`), and a point is one player's digit
changing shape for good. Against it, goals are 96% / 94% precise in the 2024 and 2022 finals, and about 80% of
points are found (the misses are mostly a fast puck the detector lost on its way in, or a point scored on a
camera angle that isn't calibrated). In the two 2022 USAA broadcasts the board skips some game-winning points
and doesn't count shots between games, so their precision reads lower (74%) than it is. The 2023 final's
red seven-segment digits don't read reliably this way yet.

`python scripts/check_switches.py --model v2` compares positions just before and after each live camera switch
(-> `data/tracks/v2/switches.csv`): it confirms the cameras share one table frame (none is rotated 180 degrees)
and how well their positions agree across the cut.

## Training on Colab
`notebooks/colab_detector.ipynb` trains the detector, evaluates it, and detects, tracks and renders clips. Training
needs only the Roboflow keys: labels and frames come from Roboflow. Clips come from Google Drive, laid out like the
repo under `My Drive/rlairhockey/` (`data/clips/<video>/`, `data/calib/frames/`).

One-time rclone setup on the cluster (rclone is installed; it needs your browser once). From your laptop:
```bash
ssh -L 53682:localhost:53682 <sunet>@login.farmshare.stanford.edu    # forwards rclone's sign-in page
rclone config create gdrive drive scope=drive                        # open the printed 127.0.0.1:53682 link, sign in
```
Then:
```bash
scripts/sync_drive.sh push          # clips (~4.6 GB) and calibration frames -> Drive
scripts/sync_drive.sh pull <model>  # a model trained on Colab, with any detections, tracks and renders made there
scripts/sync_drive.sh push-results <model>  # detections, tracks and renders made here -> Drive
```
In Colab: open the notebook from GitHub (File → Open notebook → GitHub → `abezh10/rlairhockey`), pick a GPU runtime,
add the secrets `ROBOFLOW_API_KEY` and `ROBOFLOW_WORKSPACE`, and run all cells.
