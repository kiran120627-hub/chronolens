"""Camera-motion compensation for hand-held footage.

Each analysed frame is registered to a reference frame (the one zones are drawn on) with ORB features + RANSAC,
estimating a similarity transform (translation, scale, rotation). Moving people and vehicles are rejected as
outliers, so a fixed camera measures as no motion even with a lot of activity in the scene.

Transforms are stored per sample in series.csv as cam_a, cam_b, cam_tx, cam_c, cam_d, cam_ty and work in
normalised coordinates (0..1): [x_ref, y_ref] = [[a, b], [c, d]] @ [x, y] + [tx, ty].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLS = ["cam_a", "cam_b", "cam_tx", "cam_c", "cam_d", "cam_ty"]
IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


class Registrar:
    def __init__(self, ref_bgr: np.ndarray | None, width: int = 640):
        import cv2

        self.cv2 = cv2
        self.orb = cv2.ORB_create(nfeatures=1200, fastThreshold=12)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.width = width
        self.last = IDENTITY
        self.ref = self._features(ref_bgr) if ref_bgr is not None else None

    def _features(self, bgr: np.ndarray):
        cv2 = self.cv2
        h, w = bgr.shape[:2]
        scale = self.width / w
        g = cv2.cvtColor(cv2.resize(bgr, (self.width, int(h * scale))), cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(g, None)
        size = np.array([g.shape[1], g.shape[0]], np.float32)
        pts = np.array([k.pt for k in kp], np.float32) / size if kp else np.zeros((0, 2), np.float32)
        return pts, des

    def register(self, bgr: np.ndarray) -> tuple:
        """Similarity transform mapping this frame's normalised coordinates onto the reference frame."""
        if self.ref is None or self.ref[1] is None:
            return self.last
        pts, des = self._features(bgr)
        if des is None or len(pts) < 20:
            return self.last
        matches = self.bf.match(des, self.ref[1])
        if len(matches) < 20:
            return self.last
        src = np.float32([pts[m.queryIdx] for m in matches])
        dst = np.float32([self.ref[0][m.trainIdx] for m in matches])
        M, inl = self.cv2.estimateAffinePartial2D(src, dst, method=self.cv2.RANSAC, ransacReprojThreshold=0.006,
                                             maxIters=2000, confidence=0.99)
        if M is None or inl is None or int(inl.sum()) < 15:
            return self.last
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        if not 0.6 < scale < 1.6 or abs(M[0, 2]) > 0.5 or abs(M[1, 2]) > 0.5:
            return self.last  # implausible jump: keep the previous estimate
        self.last = (float(M[0, 0]), float(M[0, 1]), float(M[0, 2]), float(M[1, 0]), float(M[1, 1]), float(M[1, 2]))
        return self.last


class CameraPath:
    """Look up the per-sample transform and map points between frame and reference coordinates."""

    def __init__(self, series: pd.DataFrame):
        self.t = series.t.to_numpy() if len(series) else np.array([0.0])
        if len(series) and all(c in series for c in COLS):
            self.m = series[COLS].to_numpy(dtype=float)
            # light temporal smoothing of the jitter of individual estimates
            k = 3
            pad = np.pad(self.m, ((k, k), (0, 0)), mode="edge")
            self.m = np.stack([np.median(pad[i:i + 2 * k + 1], axis=0) for i in range(len(self.m))])
        else:
            self.m = np.tile(np.array(IDENTITY), (len(self.t), 1))

    @property
    def moving(self) -> bool:
        return bool(len(self.m)) and float(np.abs(self.m - np.array(IDENTITY)).max()) > 0.01

    def _idx(self, t) -> np.ndarray:
        return np.clip(np.searchsorted(self.t, np.atleast_1d(t)), 0, len(self.t) - 1)

    def to_ref(self, t, x, y):
        """Frame -> reference (normalised). t, x, y: scalars or arrays of equal length."""
        m = self.m[self._idx(t)]
        x, y = np.asarray(x, float), np.asarray(y, float)
        return m[:, 0] * x + m[:, 1] * y + m[:, 2], m[:, 3] * x + m[:, 4] * y + m[:, 5]

    def from_ref(self, t, x, y):
        """Reference -> frame (normalised)."""
        m = self.m[self._idx(t)]
        x, y = np.asarray(x, float) - m[:, 2], np.asarray(y, float) - m[:, 5]
        det = m[:, 0] * m[:, 4] - m[:, 1] * m[:, 3]
        return (m[:, 4] * x - m[:, 1] * y) / det, (-m[:, 3] * x + m[:, 0] * y) / det

    def speed(self) -> np.ndarray:
        """Per-sample displacement of the frame centre in reference coordinates (camera moving)."""
        cx, cy = self.to_ref(self.t, np.full(len(self.t), 0.5), np.full(len(self.t), 0.5))
        return np.hypot(np.diff(cx, prepend=cx[0]), np.diff(cy, prepend=cy[0]))
