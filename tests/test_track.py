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
    s, ev = smooth_puck(np.arange(n), p + rng.normal(0, 0.005, p.shape), n, FPS, {"mallet_r": mallet}, table, cfg)
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
    s, ev = smooth_puck(np.arange(150), p, 150, FPS, {}, table, cfg)
    assert len(ev) == 0 and np.isfinite(s[:, 0]).all()
