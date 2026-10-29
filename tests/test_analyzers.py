"""Synthetic-keypoint tests for the geometry and the decision logic.  run: uv run python -m pytest -q"""
import math

import numpy as np

from phisio import geometry as G
from phisio.analyzers import ATENCAO, OK, RUIM, FallDetector, Goniometer, PostureAnalyzer, RepCounter


def body(trunk_deg=0.0, neck_deg=0.0, lateral=True, hip=(300.0, 400.0), torso=150.0):
    """17x2 skeleton: trunk leaning `trunk_deg` from vertical, ear `neck_deg` ahead of the shoulder."""
    p = np.full((17, 2), np.nan)
    hx, hy = hip
    a = math.radians(trunk_deg)
    sx, sy = hx + torso * math.sin(a), hy - torso * math.cos(a)
    n = math.radians(neck_deg)
    ex, ey = sx + 40 * math.sin(n), sy - 40 * math.cos(n)
    half = 5.0 if lateral else 60.0
    for (s, h, e, k, an), sign in (((G.L_SHOULDER, G.L_HIP, G.L_EAR, G.L_KNEE, G.L_ANKLE), -1),
                                   ((G.R_SHOULDER, G.R_HIP, G.R_EAR, G.R_KNEE, G.R_ANKLE), 1)):
        p[s] = (sx + sign * half, sy)
        p[h] = (hx + sign * half, hy)
        p[e] = (ex + sign * half, ey)
        p[k] = (hx + sign * half, hy + 120)
        p[an] = (hx + sign * half, hy + 240)
    return p


CONF = np.ones(17)


def test_angles():
    assert math.isclose(G.angle_at((1, 0), (0, 0), (0, 1)), 90.0, abs_tol=1e-6)
    assert math.isclose(G.angle_at((1, 0, 0), (0, 0, 0), (0, 0, 1)), 90.0, abs_tol=1e-6)
    straight = body()
    assert G.joint_angle(straight, "joelho_D") < 1.0          # straight leg = 0 deg flexion
    assert math.isclose(G.tilt_from_vertical((0, 0), (10, -10)), 45.0, abs_tol=1e-6)
    assert math.isnan(G.angle_at((0, 0), (0, 0), (1, 1)))


def run_posture(p, seconds=3.0, fps=30):
    pa = PostureAnalyzer()
    for i in range(int(seconds * fps)):
        pa.update(p, CONF, i / fps)
    return pa


def test_posture_levels_and_debounce():
    assert run_posture(body(trunk_deg=3, neck_deg=5)).state == OK
    assert run_posture(body(trunk_deg=3, neck_deg=25)).state == ATENCAO
    assert run_posture(body(trunk_deg=25, neck_deg=5)).state == RUIM
    # a single bad frame must not change the state
    pa = PostureAnalyzer()
    for i in range(60):
        pa.update(body(trunk_deg=25) if i == 30 else body(), CONF, i / 30)
    assert pa.state == OK


def test_posture_frontal_view():
    p = body(lateral=False)
    p[G.L_SHOULDER][1] += 15   # shoulders tilted ~7 deg
    pa = run_posture(p)
    assert pa.view == "frontal" and pa.state == RUIM


def test_rep_counter():
    rc = RepCounter("agachamento", side="D")
    for angle in [5, 30, 60, 90, 60, 20, 10, 50, 85, 40, 15, 70, 10]:   # 2 full reps + 1 partial
        rc.update(angle)
    assert rc.count == 2
    assert rc.peaks == [90, 85]


def test_fall_vs_lying_down():
    fps = 30
    box_up, box_down = np.array([250, 100, 350, 650]), np.array([100, 500, 600, 640])

    def lying(p_hip_y):
        p = body(trunk_deg=90, hip=(300.0, p_hip_y))
        return p

    # fast fall: hip drops 400 px (~0.7 body heights) in 0.3 s, then stays horizontal and still
    fd = FallDetector()
    t = 0.0
    for i in range(30):
        fd.update(body(), box_up, t); t += 1 / fps
    for k in range(9):
        fd.update(body(hip=(300.0, 400 + 30 * k)), box_up, t); t += 1 / fps
    for _ in range(90):
        fd.update(lying(640), box_down, t); t += 1 / fps
    assert fd.state == "QUEDA" and len(fd.events) == 1

    # slow lying down over 4 s -> "deitado", never a fall
    fd = FallDetector()
    t = 0.0
    for i in range(30):
        fd.update(body(), box_up, t); t += 1 / fps
    for k in range(120):
        fd.update(body(hip=(300.0, 400 + 2 * k)), box_up, t); t += 1 / fps
    for _ in range(90):
        fd.update(lying(640), box_down, t); t += 1 / fps
    assert fd.state == "deitado" and not fd.events


def test_best_view_prefers_unforeshortened_limb():
    from phisio import fusion
    side = body(lateral=True)
    # camera A sees the arm in the image plane (elbow at 90 deg); camera B sees it almost end-on
    a, b = side.copy(), side.copy()
    a[G.R_ELBOW], a[G.R_WRIST] = a[G.R_SHOULDER] + (0, 80), a[G.R_SHOULDER] + (80, 80)
    b[G.R_ELBOW], b[G.R_WRIST] = b[G.R_SHOULDER] + (0, 15), b[G.R_SHOULDER] + (8, 15)
    scores = [fusion.view_score(p, CONF, "cotovelo_D") for p in (a, b)]
    angles = [G.joint_angle(p, "cotovelo_D") for p in (a, b)]
    fused, cam = fusion.best_view(angles, scores)
    assert cam == 0 and math.isclose(fused, 90.0, abs_tol=1e-6)
    assert fusion.best_view([math.nan, 40.0], [0.0, 0.5]) == (40.0, 1)


def test_merge_posture_and_fall():
    from phisio import fusion
    merged = fusion.merge_posture([{"ombros": 4.0, "tronco_lateral": 2.0}, {"pescoco": 25.0, "tronco": 5.0}])
    assert set(merged) == {"ombros", "tronco_lateral", "pescoco", "tronco"}
    pa = PostureAnalyzer()
    for i in range(30):
        pa.decide(merged, i * 0.1)
    assert pa.state == ATENCAO
    assert fusion.merge_fall(["normal", "QUEDA"]) == "QUEDA"


def test_goniometer_rom_ignores_single_frame_glitch():
    g = Goniometer()
    for a in [10, 20, 30, 40, 50, 179, 60, 70, 80, 90]:
        g.update(a)
    assert g.max < 100
    assert 10 <= g.min <= 30


def test_frontal_view_without_hips():
    """Sitting close to the webcam: hips out of frame, both ears visible -> frontal, no fake neck angle."""
    p = body(lateral=False)
    p[[G.L_HIP, G.R_HIP, G.L_KNEE, G.R_KNEE, G.L_ANKLE, G.R_ANKLE]] = np.nan
    pa = run_posture(p)
    assert pa.view == "frontal" and pa.state == OK and "pescoco" not in pa.metrics


def test_flip_lr_swaps_sides():
    p = body(lateral=False)
    p[G.R_WRIST] = p[G.R_SHOULDER] + (0, 100)
    p[G.R_ELBOW] = p[G.R_SHOULDER] + (0, 50)
    p[G.L_ELBOW] = p[G.L_SHOULDER] + (0, 50)
    p[G.L_WRIST] = p[G.L_ELBOW] + (40, 0)            # left elbow bent 90 deg
    q = p[G.FLIP_LR]
    assert sorted(G.FLIP_LR) == list(range(17))
    assert abs(G.joint_angle(q, "cotovelo_D") - G.joint_angle(p, "cotovelo_E")) < 1e-9
