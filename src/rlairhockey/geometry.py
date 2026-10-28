"""Table model and camera geometry: homographies, camera pose, image <-> table mapping.

Table coordinates: origin at the table center, x along the long axis, y across, z up, meters.
In a calibration frame the ends and sides are named as that camera sees them: the "left"
end is at -x and the "far" side at +y, which keeps every camera's frame right-handed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
FT = 0.3048
QUAD = ("corner_fl", "corner_fr", "corner_nr", "corner_nl")


@dataclass
class Table:
    length: float                       # playing surface, m
    width: float
    goal_width: float | None = None
    mallet_height: float | None = None

    @classmethod
    def load(cls, path: str | Path = ROOT / "configs/table.yaml") -> "Table":
        c = yaml.safe_load(open(path))
        return cls(c["length_ft"] * FT, c["width_ft"] * FT, c.get("goal_width"), c.get("mallet_height"))

    def landmarks(self) -> dict[str, tuple[float, float]]:
        """Surface points with known positions. Goal posts only count once goal_width is known."""
        L, W = self.length / 2, self.width / 2
        pts = {"corner_fl": (-L, W), "corner_fr": (L, W), "corner_nr": (L, -W), "corner_nl": (-L, -W),
               "center_far": (0.0, W), "center_near": (0.0, -W)}
        if self.goal_width:
            g = self.goal_width / 2
            pts |= {"goal_l_far": (-L, g), "goal_l_near": (-L, -g), "goal_r_far": (L, g), "goal_r_near": (L, -g)}
        return pts


def fit_homography(world_xy: np.ndarray, image_uv: np.ndarray) -> np.ndarray:
    """Table (x, y) -> image (u, v) from >= 4 points: least squares over all of them, LM-refined."""
    H, _ = cv2.findHomography(np.asarray(world_xy, np.float64), np.asarray(image_uv, np.float64), 0)
    return H / H[2, 2]


def apply_h(H: np.ndarray, pts) -> np.ndarray:
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    if not len(p):                      # cv2 returns None for an empty array
        return np.empty((0, 2))
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)


def focal_from_homography(H: np.ndarray, size: tuple[int, int]) -> float | None:
    """Focal length in pixels implied by a plane homography, assuming square pixels, no skew and
    the principal point at the image center (Zhang's two constraints, solved jointly).
    None when the view is too close to fronto-parallel for f to be observable."""
    w, h = size
    A = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1.0]]) @ H
    a, b = A[:, 0], A[:, 1]
    # K^-1 = diag(1/f, 1/f, 1), q = 1/f^2:
    #   r1 . r2 = 0        ->  (a0 b0 + a1 b1) q + a2 b2 = 0
    #   |r1|^2 = |r2|^2    ->  (a0^2 + a1^2 - b0^2 - b1^2) q + a2^2 - b2^2 = 0
    m = np.array([a[0] * b[0] + a[1] * b[1], a[0] ** 2 + a[1] ** 2 - b[0] ** 2 - b[1] ** 2])
    c = -np.array([a[2] * b[2], a[2] ** 2 - b[2] ** 2])
    q = float(m @ c / (m @ m))
    if q <= 0:
        return None
    f = 1 / np.sqrt(q)
    return f if 0.3 * w < f < 20 * w else None


@dataclass
class Camera:
    K: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    size: tuple[int, int]

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(self.rvec)[0]

    @property
    def center(self) -> np.ndarray:
        """Camera position in table coordinates."""
        return (-self.R.T @ self.tvec).ravel()

    @property
    def H(self) -> np.ndarray:
        """Table plane (z = 0) -> image homography."""
        R = self.R
        H = self.K @ np.c_[R[:, 0], R[:, 1], self.tvec.ravel()]
        return H / H[2, 2]

    def project(self, xyz) -> np.ndarray:
        uv, _ = cv2.projectPoints(np.asarray(xyz, np.float64).reshape(-1, 3), self.rvec, self.tvec, self.K, None)
        return uv.reshape(-1, 2)

    def to_table(self, uv, z: float = 0.0) -> np.ndarray:
        """Intersect each pixel's viewing ray with the horizontal plane at height z (e.g. mallet tops)."""
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        d = self.R.T @ np.linalg.inv(self.K) @ np.c_[uv, np.ones(len(uv))].T
        C = self.center
        return (C[:, None] + (z - C[2]) / d[2] * d).T

    def to_dict(self) -> dict:
        return dict(K=self.K.tolist(), rvec=self.rvec.ravel().tolist(), tvec=self.tvec.ravel().tolist(),
                    size=list(self.size))

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        return cls(np.array(d["K"]), np.array(d["rvec"]).reshape(3, 1), np.array(d["tvec"]).reshape(3, 1),
                   tuple(d["size"]))


def fit_camera(world_xy: np.ndarray, image_uv: np.ndarray, size: tuple[int, int],
               f: float | None = None) -> Camera | None:
    """Full pose from points on the table plane. f defaults to the value the homography implies."""
    f = f or focal_from_homography(fit_homography(world_xy, image_uv), size)
    if f is None:
        return None
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])
    obj = np.c_[np.asarray(world_xy, np.float64), np.zeros(len(world_xy))]
    img = np.asarray(image_uv, np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, None, rvec, tvec)
    return Camera(K, rvec, tvec, size)


def _line_through(pts: np.ndarray) -> np.ndarray:
    """Total least squares line (a, b, c) with a^2 + b^2 = 1, a u + b v + c = 0."""
    m = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - m)
    n = vt[1]
    return np.array([n[0], n[1], -n @ m])


def surface_score(img: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """How much each pixel looks like the playing surface, 0..255. The surface color is the median
    Lab color inside the polygon `inside` (any region mostly on the surface); players, logos and
    the center line inside it barely move the median. Lightness counts for a third, since the
    lighting varies across the table."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(inside).astype(np.int32)], 1)
    ref = np.median(lab[mask > 0], axis=0)
    d = lab - ref
    d[..., 0] *= 0.35                                                  # lighting changes mostly move L
    d = np.linalg.norm(d, axis=2)
    tau = 3.0 * max(np.percentile(d[mask > 0], 75), 3.0)
    return np.clip(1 - d / tau, 0, 1).astype(np.float32) * 255


def edge_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    return cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)


def snap_edge(grads: tuple[np.ndarray, np.ndarray], p: np.ndarray, q: np.ndarray, search: float = 6.0,
              n: int = 80, min_support: float = 0.4, inside: np.ndarray | None = None) -> dict | None:
    """Find the straight image edge near the clicked segment p-q.

    At n samples along the segment, takes the strongest step within +-search px across it, to
    sub-pixel precision, then fits a line, dropping outliers such as arms crossing the rail.
    With `inside` (a point on the table side of the edge) only steps where the image value
    falls going outward count, which on a surface_score image is the playing surface's
    boundary; without it, the side's dominant polarity is used. Returns the line (a, b, c
    with a u + b v + c = 0), its inlier edge points, the share of samples that support it and
    their rms distance, or None when under min_support of the in-frame samples agree.
    """
    gx, gy = grads
    h, w = gx.shape
    t = (q - p) / np.linalg.norm(q - p)
    nrm = np.array([-t[1], t[0]])
    if inside is not None and (np.asarray(inside) - p) @ nrm > 0:
        nrm = -nrm                                                   # point outward
    base = p + np.linspace(0.04, 0.96, n)[:, None] * (q - p)
    base = base[(base[:, 0] > search + 2) & (base[:, 0] < w - search - 3) & (base[:, 1] > search + 2) & (base[:, 1] < h - search - 3)]
    if len(base) < 4:
        return None
    n = len(base)
    offs = np.arange(-search, search + 1e-6, 0.25)
    samp = base[:, None, :] + offs[None, :, None] * nrm             # n x offsets x 2
    mx, my = samp[..., 0].astype(np.float32), samp[..., 1].astype(np.float32)
    prof = cv2.remap(gx, mx, my, cv2.INTER_LINEAR) * nrm[0] + cv2.remap(gy, mx, my, cv2.INTER_LINEAR) * nrm[1]
    if inside is not None:
        prof = -prof                                                 # value falls going outward
    else:
        prof = prof * np.sign(np.median(prof[np.arange(n), np.abs(prof).argmax(1)]))   # dominant polarity only
    k = prof.argmax(1).clip(1, len(offs) - 2)
    y0, y1, y2 = (prof[np.arange(n), k + d] for d in (-1, 0, 1))
    den = y0 - 2 * y1 + y2
    sub = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1), 0.0).clip(-1, 1)
    pts = base + (offs[k] + sub * 0.25)[:, None] * nrm
    strong = (y1 > 0) & (y1 > 0.25 * np.percentile(y1, 90))
    keep = strong.copy()
    for _ in range(3):
        if keep.sum() < 2:
            return None
        line = _line_through(pts[keep])
        r = np.abs(pts @ line[:2] + line[2])
        keep = strong & (r < max(0.75, 2.5 * np.median(r[keep])))
    support = float(keep.mean())
    if support < min_support or keep.sum() < 2:
        return None
    line = _line_through(pts[keep])
    rms = float(np.sqrt(np.mean((pts[keep] @ line[:2] + line[2]) ** 2)))
    return dict(line=line, pts=pts[keep], support=round(support, 2), rms=round(rms, 2))


def intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    x = np.cross(a, b)
    return None if abs(x[2]) < 1e-9 else x[:2] / x[2]


def refine_quad(gray: np.ndarray, quad: np.ndarray, search: float = 6.0, n: int = 80,
                min_support: float = 0.4) -> tuple[np.ndarray, list[dict | None]]:
    """Snap a clicked surface quadrilateral (fl, fr, nr, nl) to the image edges: snap each side
    (snap_edge) and intersect neighbors. A corner keeps its click when either of its sides
    lacks support or the snapped corner lands more than 1.5 * search px away."""
    grads = edge_gradients(gray)
    sides = [snap_edge(grads, quad[i], quad[(i + 1) % 4], search, n, min_support) for i in range(4)]
    out = quad.astype(np.float64).copy()
    for i in range(4):  # corner i joins side i-1 (arriving) and side i (leaving)
        a, b = sides[i - 1], sides[i]
        c = None if a is None or b is None else intersect(a["line"], b["line"])
        if c is not None and np.linalg.norm(c - quad[i]) <= 1.5 * search:
            out[i] = c
    return out, [None if s is None else dict(support=s["support"], rms=s["rms"]) for s in sides]


def world_lines(table: Table) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    """Two points on each straight table line: rail edges of the playing surface and the center line."""
    L, W = table.length / 2, table.width / 2
    return {"far": ((-L, W), (L, W)), "near": ((-L, -W), (L, -W)), "left": ((-L, W), (-L, -W)),
            "right": ((L, W), (L, -W)), "center": ((0.0, W), (0.0, -W))}


def fit_homography_lines(table: Table, pts: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, float]] | None:
    """Table -> image homography from image points known to lie on table lines (world_lines keys).

    Starts from the crossings of the per-line image fits, then minimizes the distance of every
    point to its projected line, each line weighted equally whatever its point count. Needs
    two points on the far and near rails and on two of left, right, center. Returns H and the
    rms point-to-line distance per line in px."""
    wl = world_lines(table)
    img_lines = {k: _line_through(v) for k, v in pts.items() if len(v) >= 2}
    world, image = [], []
    for side in ("far", "near"):
        for k in ("left", "right", "center"):
            if side in img_lines and k in img_lines:
                c = intersect(img_lines[side], img_lines[k])
                if c is not None:
                    world.append((wl[k][0][0], wl[side][0][1])); image.append(c)
    if sum(w[1] > 0 for w in world) < 2 or sum(w[1] < 0 for w in world) < 2:
        return None
    H0 = fit_homography(np.array(world), np.array(image))
    from scipy.optimize import least_squares
    keys = list(img_lines)
    ends = {k: np.array([[*wl[k][0], 1.0], [*wl[k][1], 1.0]]) for k in keys}

    def resid(h: np.ndarray) -> np.ndarray:
        H = np.append(h, 1.0).reshape(3, 3)
        out = []
        for k in keys:
            a, b = H @ ends[k][0], H @ ends[k][1]
            ln = np.cross(a, b)
            ln = ln / np.hypot(ln[0], ln[1])
            out.append((pts[k] @ ln[:2] + ln[2]) / np.sqrt(len(pts[k])))
        return np.concatenate(out)

    r = least_squares(resid, H0.ravel()[:8], method="lm", x_scale="jac")
    H = np.append(r.x, 1.0).reshape(3, 3)
    res = resid(r.x)
    rms, i = {}, 0
    for k in keys:
        n = len(pts[k])
        rms[k] = float(np.sqrt(np.mean((res[i:i + n] * np.sqrt(n)) ** 2)))
        i += n
    return H, rms
