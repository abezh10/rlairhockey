import numpy as np
import pandas as pd

from rlairhockey.track import best_path, load_config, parked, smooth

FPS = 30.0


def dets(rows):
    return pd.DataFrame(rows, columns=["frame", "x", "y", "score"]).astype(dict(frame=int))


def test_path_follows_moving_puck_past_parked_spare_and_noise():
    cfg = load_config()
    n = 90
    rng = np.random.default_rng(0)
    rows = []
    for t in range(n):
        x = -1.0 + 2.0 * t / n                      # 2 m in 3 s, then a gap
        if not 40 <= t < 44:
            rows.append((t, x, 0.1 * np.sin(t / 5), 0.6))
        rows.append((t, 1.1, -0.5, 0.95))            # spare puck parked in a corner, scored higher
        if t % 7 == 0:
            rows.append((t, rng.uniform(-1, 1), rng.uniform(-0.5, 0.5), 0.2))   # stray false positives
    d = dets(rows)
    spare = parked(d, n, cfg)
    assert spare.sum() == n and (d[spare].x == 1.1).all()
    d = d[~spare].reset_index(drop=True)
    path = d.iloc[best_path(d, n, FPS, 30, cfg)]
    assert len(path) == n - 4                        # every true detection, through the 4-frame gap
    assert np.allclose(path.x.to_numpy(), -1.0 + 2.0 * path.frame.to_numpy() / n)


def test_lone_resting_puck_is_not_parked():
    cfg = load_config()
    d = dets([(t, 0.0, 0.0, 0.9) for t in range(60)])
    assert not parked(d, 60, cfg).any()


def test_path_reacquires_after_long_absence():
    cfg = load_config()
    d = dets([(t, -0.8, 0.0, 0.8) for t in range(20)] + [(t, 0.9, 0.3, 0.8) for t in range(50, 80)])
    path = d.iloc[best_path(d, 80, FPS, 30, cfg)]
    assert len(path) == 50


def test_smooth_recovers_velocity_and_fills_short_gaps_only():
    t = np.r_[0:30, 33:60, 80:90]                   # 2-frame gap (filled), 19-frame gap (left empty)
    xy = np.c_[0.5 * t / FPS, np.full(len(t), 0.2)]
    s = smooth(t, xy, 90, FPS, meas_sigma=0.005, accel_sigma=5, max_gap=0.2)
    assert np.isfinite(s[31, 0]) and np.isnan(s[70, 0])
    assert abs(s[45, 2] - 0.5) < 0.02 and abs(s[45, 3]) < 0.02


def test_puck_bounce_and_strike_are_sharp_impacts():
    from rlairhockey.geometry import Table
    from rlairhockey.track import smooth_puck
    cfg, table = load_config(), Table.load()
    rail = table.width / 2 - cfg["impacts"]["puck_r"]
    rng = np.random.default_rng(2)
    n = 90
    t = np.arange(n) / FPS
    # into the far rail at (0, rail) at t = 1 s, out with normal restitution 0.8, then struck at t = 2 s
    vin, vout = np.array([1.5, 3.0]), np.array([1.5, -2.4])
    p = np.where(t[:, None] < 1, [0, rail] + vin * (t[:, None] - 1), [0, rail] + vout * (t[:, None] - 1))
    hit = np.array([0, rail]) + vout * 1.0
    p = np.where(t[:, None] < 2, p, hit + np.array([-4.0, 0.5]) * (t[:, None] - 2))
    mallet = np.full((n, 4), np.nan)
    mallet[:, :2] = hit + [0.09, 0]                 # waiting just beside the strike point
    s, ev, _, _ = smooth_puck(np.arange(n), p + rng.normal(0, 0.005, p.shape), n, FPS, {"mallet_r": mallet}, table, cfg)
    assert list(ev.kind) == ["rail", "mallet_r"]
    assert abs(ev.t[0] - 1) < 0.5 / FPS and abs(ev.y[0] - rail) < 0.02 and abs(ev.restitution[0] - 0.8) < 0.1
    assert abs(ev.t[1] - 2) < 0.5 / FPS
    assert np.allclose(s[28, 2:], vin, atol=0.3) and np.allclose(s[31, 2:], vout, atol=0.3)   # no blending


def test_straight_noisy_puck_has_no_impacts():
    from rlairhockey.geometry import Table
    from rlairhockey.track import smooth_puck
    cfg, table = load_config(), Table.load()
    t = np.arange(150) / FPS
    p = np.c_[-1 + 0.4 * t, 0.3 - 0.1 * t] + np.random.default_rng(3).normal(0, 0.015, (150, 2))
    s, ev, _, _ = smooth_puck(np.arange(150), p, 150, FPS, {}, table, cfg)
    assert len(ev) == 0 and np.isfinite(s[:, 0]).all()
    speed = np.hypot(s[:, 2], s[:, 3])                # free flight: steady, despite 1.5 cm detection noise
    assert np.abs(speed - np.hypot(0.4, 0.1)).max() < 0.08


def test_gap_over_a_bounce_is_bridged_through_the_rail():
    from rlairhockey.geometry import Table
    from rlairhockey.track import smooth_puck
    cfg, table = load_config(), Table.load()
    rail = table.width / 2 - cfg["impacts"]["puck_r"]
    n = 90
    t = np.arange(n) / FPS
    vin, vout = np.array([1.0, 2.0]), np.array([1.0, -2.0])
    p = np.where(t[:, None] < 1.5, [0, rail] + vin * (t[:, None] - 1.5), [0, rail] + vout * (t[:, None] - 1.5))
    seen = (t < 1.25) | (t > 1.75)                   # the bounce itself is hidden for half a second
    s, ev, filled, _ = smooth_puck(np.flatnonzero(seen), p[seen], n, FPS, {}, table, cfg)
    assert filled[~seen].all() and not filled[seen].any()
    assert list(ev.kind) == ["rail"] and not ev.seen[0]
    assert abs(ev.t[0] - 1.5) < 1 / FPS and abs(ev.x[0]) < 0.03
    assert np.allclose(s[~seen, :2], p[~seen], atol=0.03)


def test_gap_in_straight_motion_is_bridged_straight():
    from rlairhockey.geometry import Table
    from rlairhockey.track import smooth_puck
    cfg, table = load_config(), Table.load()
    t = np.arange(60) / FPS
    p = np.c_[-0.6 + 0.8 * t, 0.1 + 0.2 * t]
    seen = (t < 0.6) | (t > 1.2)
    s, ev, filled, _ = smooth_puck(np.flatnonzero(seen), p[seen], 60, FPS, {}, table, cfg)
    assert len(ev) == 0 and filled[~seen].all()
    assert np.allclose(s[:, :2], p, atol=0.01)


def test_pulldown_pictures_are_retimed_evenly():
    from rlairhockey.track import retime
    cfg = load_config()
    shows = np.tile([2, 2, 2, 4], 30)                 # a 24 fps source at 60 fps: 4 pictures per 10 frames
    changes = np.concatenate([[5000] + [0] * (k - 1) for k in shows])
    t, _ = retime(changes, cfg)
    assert np.isnan(t[changes == 0]).all()
    new = t[np.isfinite(t)]
    assert np.allclose(np.diff(new), 2.5, atol=0.1)


def test_doubled_then_dropped_frame_is_moved_back():
    from rlairhockey.track import retime
    cfg = load_config()
    picture = np.r_[0:20, 19, 20, 22:40]              # frame 20 repeats picture 19; picture 21 is dropped
    xy = np.c_[0.1 * picture, np.zeros(len(picture))] - 1.5   # the puck moves 10 cm per picture
    changes = np.r_[-1, np.where(np.diff(picture) == 0, 0, 5000)]
    t, unsure = retime(changes, cfg, xy)
    assert np.isnan(t[20]) and t[21] == 20 and t[22] == 22 and not unsure.any()
    assert np.array_equal(np.delete(t, [20, 21]), np.delete(np.arange(len(t), dtype=float), [20, 21]))


def test_constrain_keeps_puck_on_the_table_and_off_the_mallets():
    from rlairhockey.geometry import Table
    from rlairhockey.track import constrain
    cfg, table = load_config(), Table.load()
    ic = cfg["impacts"]
    W = table.width / 2
    tracks = {"puck": np.array([[0.3, W + 0.02, 0, 0], [0.5, 0.0, 0, 0]]),
              "mallet_l": np.full((2, 4), np.nan),
              "mallet_r": np.array([[0.9, 0.0, 0, 0], [0.56, 0.0, 0, 0]])}   # frame 1: 6 cm apart, overlapping
    before = {o: t.copy() for o, t in tracks.items()}
    ev = pd.DataFrame(dict(kind=["rail"], x=[0.3], y=[W + 0.01]))
    moved = constrain(tracks, ev, table, cfg)
    assert tracks["puck"][0, 1] == W - ic["puck_r"] and ev.y[0] == W - ic["puck_r"]
    gap = np.hypot(*(tracks["puck"][1, :2] - tracks["mallet_r"][1, :2]))
    assert abs(gap - (ic["puck_r"] + ic["mallet_r"])) < 1e-6
    assert abs(tracks["mallet_r"][1, 0] - before["mallet_r"][1, 0]) > abs(tracks["puck"][1, 0] - before["puck"][1, 0])
    assert moved == {"off_wall": 1, "apart": 1}


def test_rail_bounce_among_three_detections_is_found():
    from rlairhockey.geometry import Table
    from rlairhockey.track import smooth_puck
    cfg, table = load_config(), Table.load()
    rail = table.width / 2 - cfg["impacts"]["puck_r"]
    # a slow puck, then a hard shot into the near rail and off it toward the right goal, seen only 3 times
    t = np.r_[0:20, 21, 23, 24]
    vin, e = np.array([8.0, -6.0]), 0.8
    p0 = np.array([-0.3, -0.25])                                # struck here at frame 21
    tc = 21 / FPS + (p0[1] + rail) / 6.0                        # when it reaches y = -rail
    pts = []
    for f in t:
        s = f / FPS
        if f <= 20:
            pts.append(p0 + [0.3 * (s - 20 / FPS), 0])
        elif s <= tc:
            pts.append(p0 + vin * (s - 21 / FPS))
        else:
            hit = p0 + vin * (tc - 21 / FPS)
            pts.append(hit + np.array([vin[0], -e * vin[1]]) * (s - tc))
    s, ev, _, _ = smooth_puck(t, np.array(pts), 30, FPS, {}, table, cfg)
    rails = ev[ev.kind == "rail"]
    assert len(rails) == 1 and abs(rails.y.iloc[0] + rail) < 1e-6 and abs(rails.t.iloc[0] - tc) < 0.5 / FPS
    assert s[24, 3] > 0 and np.allclose(s[24, :2], pts[-1], atol=0.03)   # heading away from the rail, through the last detection


def test_game_finds_goal_pickup_and_serve():
    from rlairhockey.game import game_events
    from rlairhockey.geometry import Table
    cfg, table = load_config(), Table.load()
    n = 300
    puck = np.full((n, 4), np.nan)
    puck[0:60] = [0.5, 0.1, 0.0, 0.0]                           # set up on the right, struck at frame 60
    f = np.arange(60, 80)
    puck[60:80] = np.c_[0.5 - 12.0 * (f - 60) / FPS, np.full(20, 0.1), np.full(20, -12.0), np.zeros(20)]
    # gone at x = -1.1, heading into the left goal
    puck[150:200] = [-0.6, 0.3, 0.0, 0.0]                       # put down on the left, then picked up
    tracks = pd.concat([pd.DataFrame(dict(frame=np.arange(n), obj=o, x=a[:, 0], y=a[:, 1], vx=a[:, 2], vy=a[:, 3]))
                        for o, a in (("puck", puck), ("mallet_l", np.full((n, 4), np.nan)),
                                     ("mallet_r", np.full((n, 4), np.nan)))], ignore_index=True)
    impacts = pd.DataFrame(dict(t=[60 / FPS], frame=[60], x=[0.5], y=[0.1], vx_in=[0.0], vy_in=[0.0], vx_out=[-12.0],
                                vy_out=[0.0], kind=["mallet_r"], offset=[np.nan], restitution=[np.nan], seen=[True]))
    ev, phase = game_events(tracks, impacts, FPS, table, cfg)
    kinds = list(ev[ev.kind != "cross"].kind)
    assert kinds == ["serve", "shot", "goal", "placed", "pickup"], kinds
    assert ev[ev.kind == "goal"].side.iloc[0] == "left" and ev[ev.kind == "shot"].detail.iloc[0] == "goal"
    assert phase[30] == "setup" and phase[70] == "live" and phase[100] == "dead" and phase[170] == "setup"
