"""Perception engines: YOLO26 pose (detection), depth (2D->3D), segmentation (privacy) and MOG2 motion."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def pick_device(arg="auto"):
    """'auto' -> first CUDA GPU if available, else CPU (so the same command works on any laptop)."""
    if arg == "auto":
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    return "cpu" if arg == "cpu" else int(arg)


def _load(name):
    local = MODELS_DIR / name
    return YOLO(str(local) if local.exists() else name)


@dataclass
class Person:
    kpts: np.ndarray   # (17, 2) pixel coordinates
    conf: np.ndarray   # (17,) keypoint confidence
    box: np.ndarray    # (4,) xyxy
    score: float


class PoseEngine:
    def __init__(self, model="yolo26n-pose.pt", device="auto", imgsz=640):
        self.model = _load(model)
        self.device = pick_device(device) if device == "auto" else device
        self.imgsz = imgsz

        self.prev_box = None

    @staticmethod
    def _iou(a, b):
        x1, y1 = np.maximum(a[:2], b[:, :2]).T
        x2, y2 = np.minimum(a[2:], b[:, 2:]).T
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
        return inter / (area(a) + area(b) - inter + 1e-9)

    def __call__(self, frame):
        """Return the tracked patient: the box overlapping the previous one, else the largest."""
        r = self.model(frame, device=self.device, imgsz=self.imgsz, verbose=False)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return None
        boxes = r.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(np.argmax(areas))
        if self.prev_box is not None:
            iou = self._iou(self.prev_box, boxes)
            if iou.max() > 0.3:
                i = int(np.argmax(iou))
        self.prev_box = boxes[i]
        data = r.keypoints.data[i].cpu().numpy()
        return Person(kpts=data[:, :2].copy(), conf=data[:, 2].copy(), box=boxes[i],
                      score=float(r.boxes.conf[i]))


class DepthEngine:
    """Monocular metric depth (Mono2Depth) with YOLO26-depth; output is H x W in meters."""

    def __init__(self, model="yolo26n-depth.pt", device="auto"):
        self.model = _load(model)
        self.device = pick_device(device) if device == "auto" else device

    def __call__(self, frame):
        r = self.model(frame, device=self.device, verbose=False)[0]
        return r.depth.data.cpu().numpy()

    @staticmethod
    def sample(depth, uv, r=4):
        """Median depth in a (2r+1)^2 patch around a pixel — robust to edge pixels."""
        h, w = depth.shape
        u, v = int(round(uv[0])), int(round(uv[1]))
        patch = depth[max(0, v - r):min(h, v + r + 1), max(0, u - r):min(w, u + r + 1)]
        return float(np.median(patch)) if patch.size else float("nan")


class SegEngine:
    """Person segmentation (COCO class 0) used to blur everything except the patient."""

    def __init__(self, model="yolo26n-seg.pt", device="auto"):
        self.model = _load(model)
        self.device = pick_device(device) if device == "auto" else device

    def person_mask(self, frame):
        r = self.model(frame, device=self.device, classes=[0], verbose=False)[0]
        h, w = frame.shape[:2]
        if r.masks is None or len(r.masks) == 0:
            return np.zeros((h, w), np.uint8)
        m = r.masks.data.cpu().numpy().max(axis=0)
        return (cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)

    @staticmethod
    def privacy(frame, mask, k=51):
        blurred = cv2.GaussianBlur(frame, (k, k), 0)
        return np.where(mask[..., None] > 0, frame, blurred)


class MotionEngine:
    """Classical baseline: MOG2 background subtraction -> fraction of moving pixels."""

    def __init__(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def __call__(self, frame):
        fg = self.bg.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel)
        return fg, float(np.count_nonzero(fg)) / fg.size
