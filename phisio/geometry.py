"""Keypoint indices (COCO-17) and the joint-angle math used by every analyzer."""
import math

import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16

# index permutation that swaps every left/right keypoint pair. Used when the image is mirrored for
# display: the model then sees a mirror image of the person and labels the right arm as the left one.
FLIP_LR = [NOSE, R_EYE, L_EYE, R_EAR, L_EAR, R_SHOULDER, L_SHOULDER, R_ELBOW, L_ELBOW, R_WRIST, L_WRIST,
           R_HIP, L_HIP, R_KNEE, L_KNEE, R_ANKLE, L_ANKLE]

SKELETON = [
    (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP), (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
    (L_EAR, L_SHOULDER), (R_EAR, R_SHOULDER),
]

# Clinical joints: (a, vertex, c, kind). "flexion" reports 180 - interior angle, so a straight
# limb reads 0 deg like a goniometer; "interior" reports the angle itself (e.g. shoulder abduction).
JOINTS = {
    "cotovelo_E": (L_SHOULDER, L_ELBOW, L_WRIST, "flexion"),
    "cotovelo_D": (R_SHOULDER, R_ELBOW, R_WRIST, "flexion"),
    "joelho_E": (L_HIP, L_KNEE, L_ANKLE, "flexion"),
    "joelho_D": (R_HIP, R_KNEE, R_ANKLE, "flexion"),
    "quadril_E": (L_SHOULDER, L_HIP, L_KNEE, "flexion"),
    "quadril_D": (R_SHOULDER, R_HIP, R_KNEE, "flexion"),
    "ombro_E": (L_HIP, L_SHOULDER, L_ELBOW, "interior"),
    "ombro_D": (R_HIP, R_SHOULDER, R_ELBOW, "interior"),
}


def angle_at(a, b, c):
    """Interior angle ABC in degrees (0-180); works for 2D or 3D points. NaN if degenerate."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    ba, bc = a - b, c - b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return math.nan
    cos = np.dot(ba, bc) / (na * nc)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def joint_angle(points, joint):
    a, b, c, kind = JOINTS[joint]
    interior = angle_at(points[a], points[b], points[c])
    return 180.0 - interior if kind == "flexion" else interior


def tilt_from_vertical(lower, upper):
    """Angle (deg) between the segment lower->upper and the image's 'up' direction (y grows down)."""
    dx, dy = upper[0] - lower[0], upper[1] - lower[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return math.nan
    return float(np.degrees(np.arctan2(abs(dx), -dy)))


def tilt_from_horizontal(p, q):
    """Absolute angle (deg, 0-90) of segment p-q against the horizontal."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return math.nan
    ang = abs(np.degrees(np.arctan2(dy, dx)))
    return float(min(ang, 180.0 - ang))


def midpoint(p, q):
    return (np.asarray(p, float) + np.asarray(q, float)) / 2.0


def focal_from_fov(width_px, hfov_deg):
    return (width_px / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def backproject(uv, depth_m, f_px, cx, cy):
    """Pinhole back-projection of pixel (u, v) at metric depth Z to camera coordinates (X, Y, Z)."""
    u, v = uv
    return np.array([(u - cx) * depth_m / f_px, (v - cy) * depth_m / f_px, depth_m])
