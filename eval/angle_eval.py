"""Angle accuracy vs a reference goniometer/protractor.

Take photos of a joint held at known angles (measure with a protractor or a phone goniometer app)
and list them in a CSV:  file,joint,true_deg   e.g.  fotos/cotovelo_90.jpg,cotovelo_D,90

usage: uv run python eval/angle_eval.py fotos/labels.csv [--depth] [--hfov 65]
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from phisio import geometry as G
from phisio.engines import DepthEngine, PoseEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--depth", action="store_true")
    ap.add_argument("--hfov", type=float, default=65.0)
    ap.add_argument("--pose", default="yolo26n-pose.pt")
    args = ap.parse_args()

    base = Path(args.labels).parent
    df = pd.read_csv(args.labels)
    pose = PoseEngine(args.pose)
    depth = DepthEngine() if args.depth else None
    rows = []
    for r in df.itertuples():
        img = cv2.imread(str(base / r.file) if not Path(r.file).is_absolute() else r.file)
        if img is None:
            print(f"skip {r.file}: cannot read")
            continue
        p = pose(img)
        if p is None:
            rows.append(dict(file=r.file, joint=r.joint, true=r.true_deg, est2d=math.nan, est3d=math.nan))
            continue
        pts = np.where(p.conf[:, None] >= 0.3, p.kpts, np.nan)
        est2d = G.joint_angle(pts, r.joint)
        est3d = math.nan
        if depth is not None:
            h, w = img.shape[:2]
            d = depth(img)
            f = G.focal_from_fov(w, args.hfov)
            p3 = np.array([G.backproject(pt, DepthEngine.sample(d, pt), f, w / 2, h / 2)
                           if np.all(np.isfinite(pt)) else [np.nan] * 3 for pt in pts])
            est3d = G.joint_angle(p3, r.joint)
        rows.append(dict(file=r.file, joint=r.joint, true=r.true_deg, est2d=est2d, est3d=est3d))

    res = pd.DataFrame(rows)
    res["err2d"] = (res["est2d"] - res["true"]).abs()
    res["err3d"] = (res["est3d"] - res["true"]).abs()
    print(res.round(1).to_string(index=False))
    print(f"\nMAE 2D: {res['err2d'].mean():.1f} deg  (n={res['err2d'].notna().sum()})")
    if args.depth:
        print(f"MAE 3D: {res['err3d'].mean():.1f} deg  (n={res['err3d'].notna().sum()})")
    out = Path(args.labels).with_name("angle_eval_result.csv")
    res.to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
