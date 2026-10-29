"""Phisio — webcam posture & mobility monitor.

Pipeline per frame: camera -> YOLO26-pose (detection) -> confidence gate + One-Euro (filtering)
-> joint angles (features, 2D or 3D with monocular depth) -> analyzers (pattern recognition and
decision) -> overlay, dashboard and CSV log.
"""
import argparse
import csv
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import geometry as G
from . import ui
from .analyzers import FallDetector, Goniometer, PostureAnalyzer, RepCounter
from .engines import DepthEngine, MotionEngine, PoseEngine, SegEngine, pick_device
from .filters import KeypointFilter

MODES = ["Postura", "Angulos", "Exercicio", "Queda"]
JOINT_NAMES = list(G.JOINTS)
DISPLAY_H = 720

POSES = ("Sente reto", "Cabeça um pouco à frente", "Cabeça bem à frente (celular)")

# Guided sessions for when the person is far from the keyboard: (on-screen instruction, true posture
# label for the log or None, seconds, mode index or None to keep the current one).
GUIDES = {
    "postura": [("Posicione-se: olhe para o notebook", None, 8, 0)]
    + [step for order in ((0, 1, 2), (2, 0, 1), (1, 2, 0)) for lv in order for step in (
        ("Prepare: " + POSES[lv].lower(), None, 4, None),
        (POSES[lv], lv, 16, None))]
    + [("Fim, obrigado!", None, 3, None)],
    "queda": [
        ("Prepare: corpo inteiro e o sofá na imagem", None, 10, 2),
        ("Exercício: faça 5 agachamentos", None, 25, 2),
        ("Queda: deite devagar no sofá (não é queda)", None, 12, 3),
        ("Queda: levante", None, 8, 3),
        ("Queda: caia rápido no sofá e fique parado", None, 15, 3),
        ("Fim", None, 3, None),
    ],
    "so_queda": [
        ("Prepare: você e o sofá inteiros na imagem", None, 15, 3),
        ("Deite devagar no sofá (não é queda)", None, 12, 3),
        ("Levante", None, 8, 3),
        ("Caia rápido no sofá e fique parado", None, 15, 3),
        ("Fim", None, 3, None),
    ],
    "demo": [
        ("Ângulos: dobre e estique o cotovelo", None, 20, 1),
        ("Postura: cabeça bem à frente", None, 15, 0),
        ("Postura: agora sente reto", None, 8, 0),
        ("Exercício: faça 5 agachamentos", None, 25, 2),
        ("Queda: deite devagar (não é queda)", None, 12, 3),
        ("Queda: levante", None, 8, 3),
        ("Queda: caia rápido no sofá e fique parado", None, 14, 3),
        ("Fim", None, 3, None),
    ],
}


def guide_step(guide, elapsed):
    """Current step of a guided session and its remaining seconds; None when it is over."""
    for text, label, dur, mode in guide:
        if elapsed < dur:
            return text, label, dur - elapsed, mode
        elapsed -= dur
    return None


big_caption = ui.caption

LOG_FIELDS = (["t", "fps", "mode", "person", "view", "posture", "posture_alert", "gt"]
              + list(PostureAnalyzer.THRESHOLDS)
              + [f"{j}" for j in JOINT_NAMES] + [f"{j}_raw" for j in JOINT_NAMES] + [f"{j}_3d" for j in JOINT_NAMES]
              + ["gonio_joint", "exercise", "reps", "fall_state", "fall_speed", "motion"])


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--pose", default="yolo26n-pose.pt")
    ap.add_argument("--device", default="auto", help="'auto', '0' for GPU, 'cpu' for CPU")
    ap.add_argument("--hfov", type=float, default=65.0, help="camera horizontal field of view (deg) for 3D")
    ap.add_argument("--mode", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--joint", default="cotovelo_D", choices=JOINT_NAMES)
    ap.add_argument("--exercise", default="agachamento", choices=list(RepCounter.PRESETS))
    ap.add_argument("--depth", action="store_true", help="start with 3D (depth) on")
    ap.add_argument("--privacy", action="store_true", help="start with privacy blur on")
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--mirror", action="store_true", help="flip the image horizontally (selfie view)")
    ap.add_argument("--logdir", default="sessions")
    ap.add_argument("--record", action="store_true", help="also save the annotated video")
    ap.add_argument("--headless", action="store_true", help="no window (batch processing of a video)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    ap.add_argument("--guide", choices=list(GUIDES), help="guided session with big on-screen instructions "
                    "(postura: labels the true posture automatically; demo: walks through the 4 modes)")
    return ap.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"[phisio] device: {device}")
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        # laptop webcams deliver only ~3 FPS at 720p in raw YUYV; MJPG gives ~25-30 FPS
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"could not open source {args.source}")
    is_file = not isinstance(src, int)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    pose = PoseEngine(args.pose, device=device)
    depth_engine = seg_engine = None
    motion_engine = MotionEngine()

    kf = KeypointFilter(enabled=not args.no_filter)
    kf3 = KeypointFilter(dims=3, enabled=not args.no_filter, mincutoff=0.5)
    posture, gonio = PostureAnalyzer(), Goniometer(args.joint)
    reps, fall = RepCounter(args.exercise), FallDetector()

    mode = args.mode - 1
    use_depth, use_privacy, show_motion = args.depth, args.privacy, False
    gt = None
    hist_angle, hist_rep, hist_speed = deque(maxlen=400), deque(maxlen=400), deque(maxlen=400)
    hist_angle3d = deque(maxlen=400)

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logdir / f"sessao_{stamp}.csv"
    log_f = open(log_path, "w", newline="")
    log = csv.DictWriter(log_f, fieldnames=LOG_FIELDS)
    log.writeheader()
    writer = None

    t_start, frame_idx, fps = time.time(), 0, 0.0
    last_tick = time.time()
    print(f"[phisio] logging to {log_path}")

    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and frame_idx >= args.max_frames):
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)
        t = frame_idx / src_fps if is_file else time.time() - t_start
        frame_idx += 1
        h, w = frame.shape[:2]
        step = guide_step(GUIDES[args.guide], time.time() - t_start) if args.guide else None
        if args.guide:
            if step is None:
                break
            gt = step[1]
            if step[3] is not None:
                mode = step[3]

        # --- perception
        person = pose(frame)
        if person is not None and args.mirror:
            person.kpts, person.conf = person.kpts[G.FLIP_LR], person.conf[G.FLIP_LR]
        display = frame.copy()
        if use_privacy:
            seg_engine = seg_engine or SegEngine(device=device)
            display = SegEngine.privacy(frame, seg_engine.person_mask(frame))
        fg, motion = motion_engine(frame)
        if show_motion:
            display = cv2.addWeighted(display, 0.6, cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR), 0.4, 0)

        row = {k: "" for k in LOG_FIELDS}
        angles, angles_raw, angles3d = {}, {}, {}
        if person is not None:
            pts = kf(person.kpts, person.conf, t)
            raw = np.where(person.conf[:, None] >= kf.min_conf, person.kpts, np.nan)
            for j in JOINT_NAMES:
                angles[j] = G.joint_angle(pts, j)
                angles_raw[j] = G.joint_angle(raw, j)

            if use_depth:
                depth_engine = depth_engine or DepthEngine(device=device)
                dmap = depth_engine(frame)
                f_px = G.focal_from_fov(w, args.hfov)
                p3 = np.full((17, 3), np.nan)
                for i in range(17):
                    if np.all(np.isfinite(pts[i])):
                        z = DepthEngine.sample(dmap, pts[i])
                        if math.isfinite(z) and z > 0:
                            p3[i] = G.backproject(pts[i], z, f_px, w / 2, h / 2)
                p3 = kf3(p3, np.where(np.isfinite(p3[:, 0]), person.conf, 0.0), t)
                for j in JOINT_NAMES:
                    angles3d[j] = G.joint_angle(p3, j)

            posture.update(pts, person.conf, t)
            gonio.update(angles[gonio.joint])
            reps.update(angles[reps.joint])
            fall.update(pts, person.box, t)
            hist_angle.append((t, angles[gonio.joint]))
            hist_angle3d.append((t, angles3d.get(gonio.joint, math.nan)))
            hist_rep.append((t, angles[reps.joint]))
            hist_speed.append((t, fall.speed))

            ui.draw_skeleton(display, raw, (110, 110, 110), 1, 2)
            ui.draw_skeleton(display, pts, None, 3, 5)
            focus = gonio.joint if mode in (0, 1, 3) else reps.joint
            ui.highlight_joint(display, pts, focus, angles[focus])

        # --- decisions -> banners
        if fall.state == "QUEDA":
            ui.banner(display, "Alerta: queda detectada", ui.C["bad"])
        elif mode == 0 and posture.alert(t):
            ui.banner(display, "Corrija a postura", ui.C["bad"])

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_tick, 1e-6)) if fps else 1.0 / max(now - last_tick, 1e-6)
        last_tick = now

        # --- log
        row.update(t=f"{t:.3f}", fps=f"{fps:.1f}", mode=MODES[mode], person=int(person is not None),
                   view=posture.view, posture=posture.state, posture_alert=int(posture.alert(t)),
                   gt="" if gt is None else gt, gonio_joint=gonio.joint, exercise=reps.exercise,
                   reps=reps.count, fall_state=fall.state, fall_speed=f"{fall.speed:.3f}", motion=f"{motion:.4f}")
        for k, v in posture.metrics.items():
            row[k] = f"{v:.2f}"
        for j in JOINT_NAMES:
            for suffix, src_d in (("", angles), ("_raw", angles_raw), ("_3d", angles3d)):
                v = src_d.get(j, math.nan)
                row[j + suffix] = f"{v:.2f}" if math.isfinite(v) else ""
        log.writerow(row)

        # --- compose window
        scale = DISPLAY_H / h
        view = cv2.resize(display, (int(w * scale), DISPLAY_H))
        panel = ui.dashboard(DISPLAY_H, dict(
            mode_name=MODES[mode], mode_tab=ui.MODE_TABS[mode], fps=fps, thresholds=PostureAnalyzer.THRESHOLDS,
            hist_angle=hist_angle, hist_angle3d=hist_angle3d, hist_rep=hist_rep, hist_speed=hist_speed, rep_thresholds=(reps.low, reps.high),
            drop_speed=fall.drop_speed, person=person is not None, filter=kf.enabled, depth=use_depth,
            privacy=use_privacy, motion=show_motion, posture_state=posture.state, view=posture.view,
            posture_metrics=posture.metrics, posture_time=posture.time_in,
            gonio_joint=gonio.joint, gonio_value=gonio.value, gonio_min=gonio.min, gonio_max=gonio.max,
            gonio_rom=gonio.rom, gonio_3d=angles3d.get(gonio.joint, math.nan),
            exercise=reps.exercise, rep_joint=reps.joint, reps=reps.count, rep_phase=reps.phase,
            fall_state=fall.state, fall_speed=fall.speed, falls=len(fall.events), gt=gt))
        if step is not None:
            big_caption(view, step[0], step[2])
        out = np.hstack([view, panel])

        if args.record and (is_file or frame_idx > 30):  # live: wait for a stable FPS estimate
            if writer is None:
                writer = cv2.VideoWriter(str(logdir / f"sessao_{stamp}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         src_fps if is_file else max(5.0, round(fps)), (out.shape[1], out.shape[0]))
            writer.write(out)
        if args.headless:
            continue

        cv2.imshow("Phisio", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key in map(ord, "1234"):
            mode = key - ord("1")
        elif key == ord("j"):
            gonio.joint = JOINT_NAMES[(JOINT_NAMES.index(gonio.joint) + 1) % len(JOINT_NAMES)]
            gonio.reset()
        elif key == ord("e"):
            names = list(RepCounter.PRESETS)
            reps.set_exercise(names[(names.index(reps.exercise) + 1) % len(names)])
        elif key == ord("l"):
            reps.side = "E" if reps.side == "D" else "D"
            reps.set_exercise(reps.exercise)
            side = gonio.joint[-1]
            gonio.joint = gonio.joint[:-1] + ("E" if side == "D" else "D")
            gonio.reset()
        elif key == ord("r"):
            gonio.reset()
            reps.reset()
        elif key == ord("f"):
            kf.enabled = kf3.enabled = not kf.enabled
        elif key == ord("d"):
            use_depth = not use_depth
        elif key == ord("p"):
            use_privacy = not use_privacy
        elif key == ord("m"):
            show_motion = not show_motion
        elif key in map(ord, "789"):
            gt = key - ord("7")
        elif key == ord("0"):
            gt = None
    hist_angle, hist_rep, hist_speed = deque(maxlen=400), deque(maxlen=400), deque(maxlen=400)

    cap.release()
    if writer is not None:
        writer.release()
    log_f.close()
    cv2.destroyAllWindows()
    print(f"[phisio] session saved: {log_path}")
    print(f"[phisio] report: uv run python -m phisio.report {log_path}")


if __name__ == "__main__":
    main()
