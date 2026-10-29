"""Pattern recognition + decision on top of filtered keypoints.

Each analyzer turns features (angles, velocities) into a discrete state, using thresholds,
hysteresis and time windows so that a single noisy frame never triggers a decision.
"""
import math
from collections import deque

import numpy as np

from . import geometry as G

OK, ATENCAO, RUIM = 0, 1, 2
LEVEL_NAMES = ["OK", "ATENCAO", "RUIM"]


def _finite(*vals):
    return all(np.all(np.isfinite(v)) for v in vals)


def _level(value, warn, bad):
    if not math.isfinite(value):
        return None
    return RUIM if value >= bad else ATENCAO if value >= warn else OK


class PostureAnalyzer:
    # (warn, bad) thresholds in degrees
    THRESHOLDS = {
        "pescoco": (20.0, 30.0),        # ear-shoulder line vs vertical (lateral view): forward head
        "tronco": (10.0, 20.0),         # hip-shoulder line vs vertical: trunk lean
        "ombros": (3.0, 6.0),           # shoulder line vs horizontal (frontal view): asymmetry
        "tronco_lateral": (5.0, 10.0),  # mid-hip -> mid-shoulder vs vertical (frontal view)
    }

    def __init__(self, debounce_s=1.0, alert_s=5.0):
        self.debounce_s, self.alert_s = debounce_s, alert_s
        self.state, self.state_since = OK, 0.0
        self.cand, self.cand_since = OK, 0.0
        self.time_in = [0.0, 0.0, 0.0]
        self.last_t = None
        self.view = "-"
        self.metrics = {}

    def features(self, p, conf):
        ls, rs, lh, rh = p[G.L_SHOULDER], p[G.R_SHOULDER], p[G.L_HIP], p[G.R_HIP]
        m = {}
        frontal = False
        if _finite(ls, rs, lh, rh):
            sh_w = np.linalg.norm(ls - rs)
            torso = np.linalg.norm(G.midpoint(ls, rs) - G.midpoint(lh, rh))
            frontal = torso > 1e-6 and sh_w / torso > 0.45
        elif _finite(ls, rs, p[G.L_EAR], p[G.R_EAR]):
            # sitting close to the camera, hips out of frame: both ears visible = facing the camera
            frontal = True
        if frontal:
            self.view = "frontal"
            m["ombros"] = G.tilt_from_horizontal(ls, rs)
            if _finite(lh, rh):
                m["tronco_lateral"] = G.tilt_from_vertical(G.midpoint(lh, rh), G.midpoint(ls, rs))
        else:
            self.view = "lateral"
            # use the side the camera sees best
            left = conf[[G.L_EAR, G.L_SHOULDER, G.L_HIP]].mean()
            right = conf[[G.R_EAR, G.R_SHOULDER, G.R_HIP]].mean()
            ear, sh, hip = ((G.L_EAR, G.L_SHOULDER, G.L_HIP) if left >= right
                            else (G.R_EAR, G.R_SHOULDER, G.R_HIP))
            if _finite(p[sh], p[ear]):
                m["pescoco"] = G.tilt_from_vertical(p[sh], p[ear])
            if _finite(p[sh], p[hip]):
                m["tronco"] = G.tilt_from_vertical(p[hip], p[sh])
        return m

    def update(self, p, conf, t):
        return self.decide(self.features(p, conf), t)

    def decide(self, metrics, t):
        """Thresholds + debounce on already-computed metrics (also used by the 2-camera fusion)."""
        if self.last_t is not None:
            self.time_in[self.state] += t - self.last_t
        self.last_t = t
        self.metrics = metrics
        levels = [_level(v, *self.THRESHOLDS[k]) for k, v in self.metrics.items()]
        levels = [lv for lv in levels if lv is not None]
        raw = max(levels) if levels else self.state
        # debounce: a new state must persist for debounce_s before it is accepted
        if raw != self.cand:
            self.cand, self.cand_since = raw, t
        if self.cand != self.state and t - self.cand_since >= self.debounce_s:
            self.state, self.state_since = self.cand, t
        return self.state

    def alert(self, t):
        return self.state == RUIM and t - self.state_since >= self.alert_s


class Goniometer:
    """Live angle of one joint plus its range of motion (ADM = max - min).

    min/max are taken over a 5-frame median, so a single-frame keypoint glitch (e.g. the wrist
    leaving the image) does not set a fake 179 deg maximum."""

    def __init__(self, joint="cotovelo_D", median_n=5):
        self.joint = joint
        self.recent = deque(maxlen=median_n)
        self.reset()

    def reset(self):
        self.min, self.max, self.value = math.inf, -math.inf, math.nan
        self.recent.clear()

    def update(self, angle):
        self.value = angle
        if math.isfinite(angle):
            self.recent.append(angle)
            if len(self.recent) == self.recent.maxlen:
                m = float(np.median(self.recent))
                self.min, self.max = min(self.min, m), max(self.max, m)
        return angle

    @property
    def rom(self):
        return self.max - self.min if math.isfinite(self.max) else math.nan


class RepCounter:
    # exercise -> (joint without side, low threshold, high threshold) in clinical degrees
    PRESETS = {
        "agachamento": ("joelho", 25.0, 80.0),
        "rosca_biceps": ("cotovelo", 40.0, 110.0),
        "elevacao_lateral": ("ombro", 30.0, 80.0),
    }

    def __init__(self, exercise="agachamento", side="D"):
        self.side = side
        self.set_exercise(exercise)

    def set_exercise(self, exercise):
        self.exercise = exercise
        base, self.low, self.high = self.PRESETS[exercise]
        self.joint = f"{base}_{self.side}"
        self.reset()

    def reset(self):
        self.count, self.phase, self.peak, self.peaks = 0, "inicio", -math.inf, []

    def update(self, angle):
        """State machine with hysteresis: rest (< low) -> active (> high) -> rest counts one rep."""
        if not math.isfinite(angle):
            return self.count
        if self.phase in ("inicio", "repouso") and angle > self.high:
            self.phase, self.peak = "ativo", angle
        elif self.phase == "ativo":
            self.peak = max(self.peak, angle)
            if angle < self.low:
                self.count += 1
                self.peaks.append(self.peak)
                self.phase = "repouso"
        elif self.phase == "inicio" and angle < self.low:
            self.phase = "repouso"
        return self.count


class FallDetector:
    """Distinguishes a FALL (fast drop + horizontal + stillness) from just lying down."""

    def __init__(self, drop_speed=0.8, horiz_deg=60.0, confirm_s=2.0, window_s=1.5, speed_win_s=0.4):
        self.drop_speed, self.horiz_deg = drop_speed, horiz_deg
        self.confirm_s, self.window_s, self.speed_win_s = confirm_s, window_s, speed_win_s
        self.hist = deque()           # (t, hip_y)
        self.heights = deque(maxlen=90)
        self.state, self.since = "normal", 0.0
        self.last_drop_t = -math.inf
        self.events = []
        self.speed = 0.0
        self.horizontal = False

    def update(self, p, box, t):
        ls, rs, lh, rh = p[G.L_SHOULDER], p[G.R_SHOULDER], p[G.L_HIP], p[G.R_HIP]
        if not _finite(ls, rs, lh, rh):
            return self.state
        hip, sh = G.midpoint(lh, rh), G.midpoint(ls, rs)
        trunk = G.tilt_from_vertical(hip, sh)
        w, h = box[2] - box[0], box[3] - box[1]
        self.horizontal = trunk > self.horiz_deg or w > 1.3 * h
        if not self.horizontal:
            self.heights.append(h)
        ref_h = float(np.median(self.heights)) if self.heights else h
        # downward hip speed in body-heights per second over a short window
        self.hist.append((t, hip[1]))
        while self.hist and t - self.hist[0][0] > self.speed_win_s:
            self.hist.popleft()
        t0, y0 = self.hist[0]
        self.speed = (hip[1] - y0) / max(ref_h, 1.0) / max(t - t0, 1e-3) if t > t0 else 0.0
        if self.speed > self.drop_speed:
            self.last_drop_t = t

        new = self.state
        if not self.horizontal:
            new = "normal"
        elif self.state in ("normal", "deitado"):
            new = "queda?" if t - self.last_drop_t <= self.window_s else "deitado"
        elif self.state == "queda?" and t - self.since >= self.confirm_s:
            new = "QUEDA"
            self.events.append(t)
        if new != self.state:
            self.state, self.since = new, t
        return self.state
