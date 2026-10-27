import cv2
import numpy as np

from rlairhockey.geometry import Camera, Table, apply_h, fit_camera, fit_homography, refine_quad, QUAD


def synthetic_camera(f=1400.0, size=(1920, 1080)):
    # 3 m back from the near side, 2 m up, looking at the table center
    C = np.array([0.3, -3.0, 2.0])
    z = -C / np.linalg.norm(C)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])
    return Camera(K, cv2.Rodrigues(R)[0], (-R @ C).reshape(3, 1), size)


def test_fit_recovers_focal_and_pose():
    table = Table.load()
    cam = synthetic_camera()
    lm = table.landmarks()
    world = np.array(list(lm.values()))
    uv = cam.project(np.c_[world, np.zeros(len(world))])
    fit = fit_camera(world, uv, cam.size)
    assert abs(fit.K[0, 0] - 1400) < 1
    assert np.allclose(fit.center, cam.center, atol=1e-3)
    assert np.allclose(fit.H, fit_homography(world, uv), atol=1e-6 * np.abs(fit.H).max())


def test_ray_plane_roundtrip():
    cam = synthetic_camera()
    xyz = np.array([[0.4, -0.2, 0.05], [-1.0, 0.5, 0.05]])
    back = cam.to_table(cam.project(xyz), z=0.05)
    assert np.allclose(back, xyz, atol=1e-9)
    assert np.allclose(cam.to_table(apply_h(cam.H, xyz[:, :2])), np.c_[xyz[:, :2], [0, 0]], atol=1e-9)


def test_refine_quad_snaps_to_edges():
    table = Table.load()
    cam = synthetic_camera()
    lm = table.landmarks()
    true = cam.project(np.c_[np.array([lm[k] for k in QUAD]), np.zeros(4)])
    # render at 8x and area-downsample: output pixel center i sits at input coordinate 8 i + 3.5
    big = np.full((1080 * 8, 1920 * 8), 60, np.uint8)
    cv2.fillPoly(big, [np.round((true * 8 + 3.5) * 16).astype(np.int32)], 200, shift=4)
    img = cv2.resize(big, (1920, 1080), interpolation=cv2.INTER_AREA)
    noisy = true + np.random.default_rng(0).uniform(-3, 3, true.shape)
    refined, info = refine_quad(img, noisy)
    assert all(i is not None and i["support"] > 0.8 for i in info)
    assert np.abs(refined - true).max() < 0.5 < np.abs(noisy - true).max()


def test_lines_fit_without_one_end():
    from rlairhockey.geometry import fit_homography_lines, world_lines
    table = Table.load()
    cam = synthetic_camera()
    rng = np.random.default_rng(2)
    pts = {}
    for k, (a, b) in world_lines(table).items():
        if k == "left":
            continue
        s = rng.uniform(0.1, 0.9, 12)[:, None]
        xy = np.array(a) + s * (np.array(b) - np.array(a))
        pts[k] = cam.project(np.c_[xy, np.zeros(len(xy))]) + rng.normal(0, 0.3, (len(xy), 2))
    H, rms = fit_homography_lines(table, pts)
    L, W = table.length / 2, table.width / 2
    grid = np.array([[x, y] for x in np.linspace(-L, L, 9) for y in np.linspace(-W, W, 5)])
    assert np.abs(apply_h(H, grid) - apply_h(cam.H, grid)).max() < 2.0
    assert all(0.1 < v < 0.6 for v in rms.values())
