# Air hockey vision pipeline: plan

Goal: turn edited match videos into clean top-down puck and mallet trajectories
for imitation learning and RL.

## Implementation map

| Build step | Module(s) | Done when |
|---|---|---|
| 1. Geometry + top-down render | `geometry.py` (table model, homography fit, full pose via `cv2.solvePnP`, ray-plane intersection at mallet height h), `render.py`, `scripts/click_landmarks.py` | Hand-clicked landmarks on ~10 frames from different videos give an overlay that lines up and a warped top-down view that looks right; reprojection error is logged |
| 2a. Keypoint model | `RFDETRKeypointPreview`, `scripts/sample_frames.py`, `scripts/train_keypoints.py` | 100-300 frames across many camera setups; homographies fitted from predictions match hand-clicked ones |
| 2b. Detector | `RFDETRSmall/Medium` (puck, mallet), `scripts/train_detector.py` | ~500 frames labeled, then train, auto-label, correct, retrain until puck recall holds up on blurred and far-end frames |
| 2c. Fast-puck detector | `fastpuck.py`: (a) a `puck_streak` object in the keypoint model with its two streak endpoints labeled, so one detection gives position plus the line the puck moved along during the exposure (the Kalman filter settles which way); (b) a model-free fallback: difference consecutive frames warped to top-down, mask out a widened box around each mallet (the hand holding it is inside), fit a line to elongated blobs | Fast shots that the plain `puck` class misses are recovered; streak direction agrees with the smoothed track |
| 3. Shots, tracking, stitching | `video.py` (PyAV PTS timestamps, PySceneDetect cuts, calibration gate), `track.py` (constant-velocity Kalman with rail reflection, distance-dependent R, RTS smoother), `stitch.py` (2-DoF Mahalanobis d² < 9.21, the chi-square 99% cutoff) | Tracks continue across live angle switches and break where the edit skips time |
| 4. Phases + replays | `phases.py` (hysteresis state machine, goal counting, replay matching against earlier puck paths) | Phase labels agree with hand review on the first few matches |
| 5. End to end + BC | `dataset.py` (per-timestep state, actions, visibility flags, mirroring, episodes), segment table as Parquet | ~3 h processed, a behavioral-cloning baseline trained, usable-hours metric reported |

## Decisions

- **Table:** 8 ft × 4 ft (2.438 m × 1.219 m). Table coordinates: origin at table center, x along the long axis in meters. Goal width, center-circle radius and mallet height h are still to be measured, from hand-clicked frames or a spec sheet.
- **Labeling:** Roboflow. Export COCO JSON for RF-DETR. Projects: `airhockey-objects` (puck, mallet; puck_streak and hand were dropped 2026-10-27: blurred pucks are labeled `puck`, and nothing downstream needs hands) and `airhockey-table` (table landmarks).
- **Frame rate:** first footage is 1080p at 24-60 fps (2024 Worlds 60, 2022-2023 Worlds 30, 2022 Texas 24). Fast pucks will mostly be streaks, hence step 2c.

## Footage

| Video | ID | fps | Length |
|---|---|---|---|
| 2024 Worlds finals | g2xtRuDjb5o | 60 | 31:36 |
| 2023 Worlds finals | BR5zW8vHGJU | 30 | 46:45 |
| 2022 Worlds final match | ep0KRI2pgRw | 30 | 13:43 |
| 2022 World Championship | ilGhVvwdVyE | 30 | 32:44 |
| 2022 Texas State Championship | A5gYiPuAT6g | 24 | 41:02 |

---

## Original plan

### 1. Collect footage
- About 1-3 hours of varied footage for training the vision models. Variety (angles, lighting, players) matters more than length.
- For the learning dataset, scale by usable play time: 1-2 h prototype; 5-20 h decent policy; 20-50+ h offline RL or simulator fitting.
- Expect edited matches to lose a large share of runtime to breaks, replays, and cutaways.
- Prefer high frame rates (120-240 fps) when you control the recording.

### 2. Label and train two RF-DETR models
- Object detector: classes "puck", "mallet", optionally "hand". Assign left/right player afterwards by table half. About 500-1,500 labeled frames, including blurred, occluded, and far-end pucks.
- Table keypoint model: corners, goal edges, center line ends, center circle points. About 100-300 frames across many camera setups.
- Workflow: label about 500 frames, train, auto-label more, correct mistakes, repeat.

### 3. Prepare the video
- Use real frame timestamps for timing, not frame numbers.
- Split the video into shots at every camera cut.
- Keep a shot only if the table is clearly visible: enough landmarks found and a good calibration fit. This drops close-ups, crowd shots, and graphics automatically.

### 4. Convert to table coordinates
- Fit a homography H per frame from the table landmarks (broadcast cameras pan and zoom): `[x, y, 1] ~ H^-1 [u, v, 1]`, then divide by the third value.
- Puck: project its box center directly.
- Mallets stand above the table: recover the full camera pose, then intersect the ray through the mallet's top with the plane at the mallet's known height h.
- Check: overlay the table outline on the frame and confirm the warped top-down view looks right.

### 5. Track
- One Kalman filter per object in table coordinates (position + velocity).
- Reflect predictions off the rails to model bounces.
- Trust far-away detections less.
- Smooth each clip offline to fill gaps from occlusion.
- If two synchronized views exist, feed both into the same filter.

### 6. Stitch across camera switches
- At each cut, predict where the puck and mallets should be and compare with the new shot.
- Small mismatch (Mahalanobis d² < 9.21): live angle switch, continue the track.
- Otherwise the edit skipped time: start a new episode.

### 7. Label game phases

| Phase | Detected by | Keep? |
|---|---|---|
| Live | Puck tracked and moving | Yes |
| Serve | Puck slow on one side | Yes, flagged |
| Dead ball | After a goal, puck missing or in a hand | No |
| Break / timeout | No activity for several seconds | No |
| Replay | Puck path matches earlier play, often slow | No |
| Invalid | No usable table view | No |

- Use a short delay before switching phases so brief pauses don't split rallies.
- Score: count goals (puck enters a goal mouth and disappears). No scoreboard reading needed.

### 8. Produce outputs
- Top-down render: table outline, center line, goals, puck and mallet circles, puck trail; shown next to the warped camera frame for debugging.
- Dataset per timestep: positions and velocities of puck, own mallet, opponent mallet, plus visibility flags.
- Action: the mallet's next position or velocity, from the smoothed tracks.
- Mirror the table so both players provide demonstrations.
- Episodes run from a serve to a goal and break at any invalid frame.

### 9. Quality control
- Segment table: video, shot, start/end time, phase, calibration error, stitch decision, episode ID.
- Review the first few matches by hand, then only flagged cases.
- Metrics: landmark calibration error, tracking gaps, usable hours per video.

### 10. Build order
1. Geometry module and top-down render, tested on hand-clicked landmarks.
2. Train the keypoint model, then the detector.
3. Shot splitting, tracking, and stitching.
4. Phase labeling and replay removal.
5. Process about 3 hours end to end, train a quick BC baseline, then scale up based on what it gets wrong.
