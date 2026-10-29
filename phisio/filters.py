"""Keypoint filtering: confidence gate + short hold + One-Euro smoothing per coordinate."""
import math

import numpy as np

from .one_euro_filter import OneEuroFilter


class KeypointFilter:
    """Filters a (17, D) keypoint array over time.

    - keypoints below `min_conf` are rejected; the last good value is held for `hold_s` seconds,
      after which the keypoint becomes NaN (missing);
    - accepted values go through a One-Euro filter (Casiez et al., CHI 2012), which removes jitter
      when the joint is still and keeps lag low when it moves fast.
    """

    def __init__(self, n=17, dims=2, min_conf=0.5, hold_s=0.3, mincutoff=1.0, beta=0.01, enabled=True):
        self.n, self.dims = n, dims
        self.min_conf, self.hold_s = min_conf, hold_s
        self.mincutoff, self.beta = mincutoff, beta
        self.enabled = enabled
        self._reset()

    def _reset(self):
        self.f = [[OneEuroFilter(30, self.mincutoff, self.beta, 1.0) for _ in range(self.dims)]
                  for _ in range(self.n)]
        self.last = np.full((self.n, self.dims), np.nan)
        self.last_t = np.full(self.n, -math.inf)

    def set_params(self, mincutoff, beta):
        if (mincutoff, beta) == (self.mincutoff, self.beta):
            return
        self.mincutoff, self.beta = mincutoff, beta
        for per_kp in self.f:
            for f in per_kp:
                f.setMinCutoff(mincutoff)
                f.setBeta(beta)

    def __call__(self, pts, conf, t):
        out = np.full((self.n, self.dims), np.nan)
        for i in range(self.n):
            ok = conf[i] >= self.min_conf and np.all(np.isfinite(pts[i]))
            if ok:
                if self.enabled:
                    self.last[i] = [self.f[i][d](float(pts[i][d]), t) for d in range(self.dims)]
                else:
                    self.last[i] = pts[i]
                self.last_t[i] = t
            if t - self.last_t[i] <= self.hold_s:
                out[i] = self.last[i]
        return out


class ScalarFilter:
    """One-Euro for a single signal (e.g. an angle), ignoring NaNs."""

    def __init__(self, mincutoff=1.0, beta=0.01):
        self.f = OneEuroFilter(30, mincutoff, beta, 1.0)

    def __call__(self, x, t):
        return x if not math.isfinite(x) else self.f(x, t)
