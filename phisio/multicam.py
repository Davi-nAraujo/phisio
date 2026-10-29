"""Phisio multi-câmera — two webcams (e.g. one frontal, one lateral) watching the same patient.

What two views add over one:
- goniometry by best view: per joint, the angle comes from the camera that sees the limb with the
  least foreshortening (see fusion.py); the gap between the two cameras is shown as a quality cue;
- complete posture: the frontal camera measures shoulder asymmetry / lateral lean and the side
  camera measures forward head / trunk lean, at the same time;
- redundant fall detection: if one camera is occluded, the other still sees the fall.

usage: uv run python -m phisio.multicam --sources 0 2
       uv run python -m phisio.multicam --sources video_a.mp4 video_b.mp4 --headless
"""
import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import fusion
from .app import GUIDES, big_caption, guide_step  # noqa: F401
from . import geometry as G
from . import ui
from .analyzers import FallDetector, PostureAnalyzer
from .engines import PoseEngine, pick_device
from .filters import KeypointFilter

JOINT_NAMES = list(G.JOINTS)
VIEW_H = 360
CAM_RGB = [ui.P["accent"], ui.P["warn"]]          # camera A sky, camera B amber
CAM_COLORS = [ui.bgr(c) for c in CAM_RGB]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs=2, default=["0", "2"], help="two camera indices or video files")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--pose", default="yolo26n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size (480 is faster on CPU)")
    ap.add_argument("--device", default="auto", help="'auto', '0' for GPU, 'cpu' for CPU")
    ap.add_argument("--joint", default="cotovelo_D", choices=JOINT_NAMES)
    ap.add_argument("--mirror", action="store_true", help="flip both images horizontally")
    ap.add_argument("--logdir", default="sessions")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--guide", choices=["postura"], help="guided posture session with big on-screen "
                    "instructions; logs the true label so eval.posture_eval can score A, B and fused")
    return ap.parse_args()


def open_source(s, w, h):
    src = int(s) if s.isdigit() else s
    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened():
        raise SystemExit(f"could not open source {s} (list cameras with: v4l2-ctl --list-devices)")
    return cap, not isinstance(src, int)


def fit(img, h):
    return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))


def panel(height, s):
    p = ui.Panel(height)
    P = ui.P
    p.text((20, 16), "Phisio", 30, P["text"], True)
    p.text((128, 30), "duas câmeras", 13, P["sub"])
    p.chip(p.w - 104, 22, f"{s['fps']:4.1f} FPS", P["ok"] if s["fps"] >= 12 else P["warn"], filled=False, size=13)
    x = 16
    for k, name in enumerate("AB"):
        x = p.chip(x, 66, f"câmera {name} ({s['views'][k]})", CAM_RGB[k] if s["seen"][k] else P["border"],
                   filled=s["seen"][k], size=13) + 8

    # angle by best view
    x, ty = p.card(110, 210, "ângulo pela melhor vista")
    p.text((x, ty), ui.joint_label(s["joint"]), 16, P["sub"], True)
    p.text((x - 4, ty + 18), ui.fmt(s["fused"]), 60, P["text"], True)
    if s["chosen"] >= 0:
        p.chip(x + 190, ty + 6, f"usando {'AB'[s['chosen']]}", CAM_RGB[s["chosen"]], size=13)
    p.text((x + 190, ty + 44), f"ADM {ui.fmt(s['rom'])}", 16, P["accent"], True)
    for k, name in enumerate("AB"):
        yy = ty + 104 + 26 * k
        p.text((x, yy), f"câmera {name}", 14, CAM_RGB[k], True)
        p.text((x + 120, yy), ui.fmt(s["angles"][k]), 14, P["text"], True)
        p.text((x + 200, yy), f"score {s['scores'][k]:.2f}", 13, P["sub"])
    gap = abs(s["angles"][0] - s["angles"][1])
    p.text((x, ty + 158), f"diferença entre vistas: {ui.fmt(gap)}", 13, P["sub"])

    # complete posture
    st = s["posture_state"]
    x, ty = p.card(334, 236, "postura completa (frente + lado)")
    p.chip(x, ty, ui.LEVEL_NAMES[st], ui.LEVEL_RGB[st], size=22, pad=(14, 5))
    xx = x + 170
    for k, name in enumerate("AB"):
        p.text((xx, ty + 2 + 22 * k), f"só {name}:", 13, P["sub"])
        p.chip(xx + 52, ty + 22 * k, ui.LEVEL_NAMES[s["single"][k]], ui.LEVEL_RGB[s["single"][k]], size=11, pad=(7, 1))
    yy = ty + 58
    for k, v in s["posture_metrics"].items():
        warn, bad = PostureAnalyzer.THRESHOLDS[k]
        p.text((x, yy), ui.METRIC_LABEL.get(k, k), 14, P["text"], True)
        p.text((p.w - 36, yy), ui.fmt(v, nd=1), 14, P["text"], True, anchor="ra")
        p.bar(x, yy + 22, p.w - x - 36, v, warn, bad, h=6)
        yy += 42

    # redundant fall
    fs = s["fall_state"]
    x, ty = p.card(584, 76, "queda (vale a pior das duas)")
    col = P["bad"] if fs == "QUEDA" else P["warn"] if fs != "normal" else P["ok"]
    p.chip(x, ty, ui.FALL_LABEL.get(fs, fs), col, size=15)
    p.text((x + 140, ty + 4), f"A: {s['falls'][0]}   B: {s['falls'][1]}", 13, P["sub"])
    p.text((18, height - 30), "j junta · l lado · r zera ADM · f filtro · q sair", 12, P["sub"])
    return p.to_bgr()


def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"[multicam] device: {device}")
    caps = [open_source(s, args.width, args.height) for s in args.sources]
    is_file = caps[0][1]
    src_fps = caps[0][0].get(cv2.CAP_PROP_FPS) or 30.0

    poses = [PoseEngine(args.pose, device=device, imgsz=args.imgsz) for _ in caps]
    filters = [KeypointFilter() for _ in caps]
    postures = [PostureAnalyzer() for _ in caps]   # per-camera features (and view detection)
    fused_posture = PostureAnalyzer()              # decision on the merged metrics
    falls = [FallDetector() for _ in caps]
    joint = args.joint
    rom_min, rom_max = math.inf, -math.inf

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logdir / f"multicam_{stamp}.csv"
    fields = (["t", "fps", "person", "person_A", "person_B", "view_A", "view_B", "gt", "posture", "posture_A",
               "posture_B", "fall_state", "joint", "chosen"] + [f"{j}_{c}" for j in JOINT_NAMES for c in ("A", "B", "fused")])
    log_f = open(log_path, "w", newline="")
    log = csv.DictWriter(log_f, fieldnames=fields)
    log.writeheader()
    writer = None
    print(f"[multicam] logging to {log_path}")

    t_start, frame_idx, fps, last_tick = time.time(), 0, 0.0, time.time()
    while True:
        # grab both first, then decode: keeps the two frames as close in time as possible
        if not all(cap.grab() for cap, _ in caps) or (args.max_frames and frame_idx >= args.max_frames):
            break
        frames = [cap.retrieve()[1] for cap, _ in caps]
        if any(f is None for f in frames):
            break
        if args.mirror:
            frames = [cv2.flip(f, 1) for f in frames]
        t = frame_idx / src_fps if is_file else time.time() - t_start
        frame_idx += 1
        step = guide_step(GUIDES[args.guide], time.time() - t_start) if args.guide else None
        if args.guide and step is None:
            break
        gt = step[1] if step else None

        pts_all, conf_all, angles_all, metrics_all, views = [], [], [], [], []
        for k, frame in enumerate(frames):
            person = poses[k](frame)
            if person is None:
                pts_all.append(None)
                conf_all.append(None)
                angles_all.append({j: math.nan for j in JOINT_NAMES})
                metrics_all.append({})
                views.append("-")
                continue
            pts = filters[k](person.kpts, person.conf, t)
            metrics_all.append(postures[k].features(pts, person.conf))
            views.append(postures[k].view)
            falls[k].update(pts, person.box, t)
            pts_all.append(pts)
            conf_all.append(person.conf)
            angles_all.append({j: G.joint_angle(pts, j) for j in JOINT_NAMES})

        fused, chosen = {}, {}
        for j in JOINT_NAMES:
            scores = [fusion.view_score(p, c, j) if p is not None else 0.0 for p, c in zip(pts_all, conf_all)]
            fused[j], chosen[j] = fusion.best_view([a[j] for a in angles_all], scores)
            if j == joint:
                focus_scores = scores
        if math.isfinite(fused[joint]):
            rom_min, rom_max = min(rom_min, fused[joint]), max(rom_max, fused[joint])
        p_state = fused_posture.decide(fusion.merge_posture(metrics_all), t)
        single = [postures[k].decide(metrics_all[k], t) for k in range(len(caps))]  # each camera alone
        fall_state = fusion.merge_fall([f.state for f in falls])

        views_img = []
        for k, frame in enumerate(frames):
            img = frame.copy()
            if pts_all[k] is not None:
                ui.draw_skeleton(img, pts_all[k], CAM_COLORS[k], 2, 4)
                color = (0, 220, 0) if chosen[joint] == k else (160, 160, 160)
                ui.highlight_joint(img, pts_all[k], joint, angles_all[k][joint], color)
            img = fit(img, VIEW_H)
            ui.pill(img, f"câmera {'AB'[k]} · {views[k]}", (10, VIEW_H - 40), 16, fg=ui.P["bg"], bg=CAM_RGB[k])
            views_img.append(img)
        w = max(v.shape[1] for v in views_img)
        stack = np.vstack([cv2.copyMakeBorder(v, 0, 0, 0, w - v.shape[1], cv2.BORDER_CONSTANT) for v in views_img])
        if step is not None:
            big_caption(stack, step[0], step[2])
        if fall_state == "QUEDA":
            ui.banner(stack, "Alerta: queda detectada", ui.C["bad"])
        elif fused_posture.alert(t):
            ui.banner(stack, "Corrija a postura", ui.C["bad"])

        now = time.time()
        inst = 1.0 / max(now - last_tick, 1e-6)
        fps, last_tick = (0.9 * fps + 0.1 * inst) if fps else inst, now

        out = np.hstack([stack, panel(stack.shape[0], dict(
            fps=fps, seen=[p is not None for p in pts_all], joint=joint, fused=fused[joint],
            chosen=chosen[joint], angles=[a[joint] for a in angles_all], scores=focus_scores,
            rom=rom_max - rom_min if math.isfinite(rom_max) else math.nan,
            posture_state=p_state, views=views, posture_metrics=fused_posture.metrics,
            fall_state=fall_state, falls=[f.state for f in falls], single=single))])

        row = dict(t=f"{t:.3f}", fps=f"{fps:.1f}", person=int(any(p is not None for p in pts_all)),
                   gt="" if gt is None else gt, posture_A=single[0], posture_B=single[1],
                   person_A=int(pts_all[0] is not None),
                   person_B=int(pts_all[1] is not None), view_A=views[0], view_B=views[1],
                   posture=p_state, fall_state=fall_state, joint=joint, chosen="AB"[chosen[joint]]
                   if chosen[joint] >= 0 else "")
        for j in JOINT_NAMES:
            for c, v in (("A", angles_all[0][j]), ("B", angles_all[1][j]), ("fused", fused[j])):
                row[f"{j}_{c}"] = f"{v:.2f}" if math.isfinite(v) else ""
        log.writerow(row)

        if args.record and (is_file or frame_idx > 30):  # live: wait for a stable FPS estimate
            if writer is None:
                writer = cv2.VideoWriter(str(logdir / f"multicam_{stamp}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         src_fps if is_file else max(5.0, round(fps)), (out.shape[1], out.shape[0]))
            writer.write(out)
        if args.headless:
            continue
        cv2.imshow("Phisio - 2 cameras", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("j"):
            joint = JOINT_NAMES[(JOINT_NAMES.index(joint) + 1) % len(JOINT_NAMES)]
            rom_min, rom_max = math.inf, -math.inf
        elif key == ord("l"):
            joint = joint[:-1] + ("E" if joint[-1] == "D" else "D")
            rom_min, rom_max = math.inf, -math.inf
        elif key == ord("r"):
            rom_min, rom_max = math.inf, -math.inf
        elif key == ord("f"):
            for f in filters:
                f.enabled = not f.enabled

    for cap, _ in caps:
        cap.release()
    if writer is not None:
        writer.release()
    log_f.close()
    cv2.destroyAllWindows()
    print(f"[multicam] session saved: {log_path}")


if __name__ == "__main__":
    main()
