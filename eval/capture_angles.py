"""Capture reference photos for eval/angle_eval.py.

Hold the joint at a known angle (book corner = 90, book edge = 0, paper folded diagonally = 45) and
press the key: 0 -> 0 deg, 4 -> 45, 9 -> 90, 1 -> 135. Saves fotos/<joint>_<deg>_<n>.jpg and appends
fotos/labels.csv. The live angle on screen is just to check the person is detected.

usage: uv run python -m eval.capture_angles --source 0 --joint cotovelo_D
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from phisio import geometry as G
from phisio import ui
from phisio.engines import PoseEngine

KEYS = {ord("0"): 0, ord("4"): 45, ord("9"): 90, ord("1"): 135}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--joint", default="cotovelo_D", choices=list(G.JOINTS))
    ap.add_argument("--out", default="fotos")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    labels = out / "labels.csv"
    new = not labels.exists()
    f = open(labels, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["file", "joint", "true_deg"])

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    pose = PoseEngine()
    n = len(list(out.glob("*.jpg")))
    msg = "0=0  4=45  9=90  1=135  q=sair"
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        show = frame.copy()
        p = pose(frame)
        if p is not None:
            pts = np.where(p.conf[:, None] >= 0.3, p.kpts, np.nan)
            ui.draw_skeleton(show, pts, (255, 255, 255), 2, 4)
            ang = G.joint_angle(pts, args.joint)
            ui.highlight_joint(show, pts, args.joint, ang)
        ui.put(show, f"{args.joint} | {msg}", (15, 35), 0.8, (0, 0, 0), 4)
        ui.put(show, f"{args.joint} | {msg}", (15, 35), 0.8, (255, 255, 255), 2)
        cv2.imshow("captura de referencia", show)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in KEYS:
            n += 1
            name = f"{args.joint}_{KEYS[key]}_{n}.jpg"
            cv2.imwrite(str(out / name), frame)
            w.writerow([name, args.joint, KEYS[key]])
            f.flush()
            msg = f"salvo {name}"
            print(msg)
    cap.release()
    f.close()
    cv2.destroyAllWindows()
    print(f"labels: {labels}\nnext: uv run python -m eval.angle_eval {labels} --depth")


if __name__ == "__main__":
    main()
