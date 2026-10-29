"""Throughput benchmark: per-model latency and FPS on GPU vs CPU.

usage: uv run python eval/bench.py --source video.mp4 [--frames 150]
"""
import argparse
import time

import cv2
import numpy as np

from phisio.engines import DepthEngine, PoseEngine, SegEngine, pick_device


def frames_from(source, n):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    if not out:
        raise SystemExit("no frames read")
    return out


def bench(fn, frames, warmup=5):
    for f in frames[:warmup]:
        fn(f)
    lat = []
    for f in frames:
        t0 = time.perf_counter()
        fn(f)
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    return lat.mean(), np.percentile(lat, 95), 1000.0 / lat.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--frames", type=int, default=150)
    args = ap.parse_args()
    frames = frames_from(args.source, args.frames)
    print(f"{len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}\n")
    print("| modelo | dispositivo | latência média (ms) | p95 (ms) | FPS |")
    print("|---|---|---|---|---|")
    import platform

    import torch
    gpu = f"GPU {torch.cuda.get_device_name(0).replace('NVIDIA GeForce ', '')}" if torch.cuda.is_available() else ""
    cpu = "CPU " + next((ln.split(":")[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")),
                        platform.processor()) if platform.system() == "Linux" else f"CPU {platform.processor()}"
    for dev in ((0, "cpu") if pick_device() == 0 else ("cpu",)):
        n = frames if dev == 0 else frames[: max(20, len(frames) // 5)]
        engines = {
            "yolo26n-pose": PoseEngine(device=dev),
            "yolo26s-pose": PoseEngine("yolo26s-pose.pt", device=dev),
            "yolo26n-depth": DepthEngine(device=dev),
            "yolo26n-seg": SegEngine(device=dev).person_mask,
        }
        for name, fn in engines.items():
            mean, p95, fps = bench(fn, n)
            print(f"| {name} | {gpu if dev == 0 else cpu} | {mean:.1f} | {p95:.1f} | {fps:.1f} |")


if __name__ == "__main__":
    main()
