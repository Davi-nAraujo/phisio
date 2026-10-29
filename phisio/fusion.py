"""Two-camera fusion: pick, per joint, the view that sees it best, and merge posture/fall decisions.

A 2D angle is only correct when the limb moves parallel to the image plane; out of plane it is
foreshortened. With two cameras at different angles, at least one usually sees the limb close to
its true length, so we score each view by keypoint confidence x projected limb length (normalized
by the torso, so camera distance cancels out) and keep the angle from the best-scoring view.
"""
import math

import numpy as np

from . import geometry as G

FALL_RANK = {"normal": 0, "deitado": 1, "queda?": 2, "QUEDA": 3}


def torso_length(p):
    sh = G.midpoint(p[G.L_SHOULDER], p[G.R_SHOULDER])
    hip = G.midpoint(p[G.L_HIP], p[G.R_HIP])
    if not (np.all(np.isfinite(sh)) and np.all(np.isfinite(hip))):
        # side view: one shoulder/hip may be missing, use whichever side is visible
        for s, h in ((G.L_SHOULDER, G.L_HIP), (G.R_SHOULDER, G.R_HIP)):
            if np.all(np.isfinite(p[s])) and np.all(np.isfinite(p[h])):
                return float(np.linalg.norm(p[s] - p[h]))
        return math.nan
    return float(np.linalg.norm(sh - hip))


def view_score(p, conf, joint):
    """How well this view sees `joint`: min keypoint confidence x limb length / torso length."""
    a, b, c, _ = G.JOINTS[joint]
    if not all(np.all(np.isfinite(p[i])) for i in (a, b, c)):
        return 0.0
    torso = torso_length(p)
    if not math.isfinite(torso) or torso < 1e-6:
        return 0.0
    limb = np.linalg.norm(p[a] - p[b]) + np.linalg.norm(p[c] - p[b])
    return float(min(conf[a], conf[b], conf[c]) * limb / torso)


def best_view(angles, scores):
    """angles, scores: one value per camera. Returns (fused angle, chosen camera index or -1)."""
    ok = [i for i, (a, s) in enumerate(zip(angles, scores)) if math.isfinite(a) and s > 0]
    if not ok:
        return math.nan, -1
    i = max(ok, key=lambda k: scores[k])
    return angles[i], i


def merge_posture(metrics_per_cam):
    """Frontal metrics come from the frontal camera and sagittal ones from the side camera.
    If both cameras produce the same metric, keep the worse (larger) value: a safety-first choice."""
    out = {}
    for m in metrics_per_cam:
        for k, v in m.items():
            if math.isfinite(v):
                out[k] = max(out.get(k, -math.inf), v)
    return out


def merge_fall(states):
    """Most severe state wins: a fall hidden from one camera (occlusion) is still seen by the other."""
    return max(states, key=lambda s: FALL_RANK.get(s, 0))
