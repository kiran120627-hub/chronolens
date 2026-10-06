"""Perception: detect + track objects, collect appearance features and motion signals.

One pass over the video at `settings.target_fps`:
  * YOLO11 detection + BoT-SORT tracking  -> per-sample boxes with tracker IDs
  * appearance features per tracklet         -> HSV colour histograms + CNN embeddings (for re-identification)
  * motion energy per activity zone          -> machine running / stopped
  * global motion + brightness               -> sudden changes (alarms, lights, commotion)
Everything is written to an analysis folder so later stages never touch the GPU again.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from .config import TRACK_CLASSES, Settings

Progress = Callable[[float, str], None]


@dataclass
class VideoMeta:
    path: str
    fps: float
    frames: int
    width: int
    height: int
    duration: float
    stride: int
    sample_fps: float


def probe(path: str | Path, target_fps: float = 10.0) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    stride = max(1, int(round(fps / target_fps)))
    return VideoMeta(str(path), fps, n, w, h, n / fps if fps else 0.0, stride, fps / stride)


# ----------------------------------------------------------------------------- appearance
def hsv_hist(crop: np.ndarray) -> np.ndarray:
    """Colour signature: separate H-S histograms for upper and lower half (clothes / vehicle body)."""
    if crop.size == 0:
        return np.zeros(2 * 16 * 8, np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv.shape[0]
    feats = []
    for part in (hsv[: h // 2], hsv[h // 2:]):
        hist = cv2.calcHist([part], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
        feats.append(hist / (hist.sum() + 1e-6))
    return np.concatenate(feats).astype(np.float32)


class Embedder:
    """ImageNet ResNet-18 global features (512-d) on GPU; silently disabled if unavailable."""

    def __init__(self, device: str):
        self.ok = False
        try:
            import torch
            import torchvision

            self.torch = torch
            w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            m = torchvision.models.resnet18(weights=w)
            m.fc = torch.nn.Identity()
            self.model = m.eval().to(device)
            self.device = device
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            self.ok = True
        except Exception as e:  # noqa: BLE001
            print(f"[chronolens] CNN re-id embeddings disabled: {e}")

    def __call__(self, crops: list[np.ndarray]) -> np.ndarray | None:
        if not self.ok or not crops:
            return None
        torch = self.torch
        x = np.stack([cv2.resize(c, (128, 256))[:, :, ::-1] for c in crops])  # same size for every crop
        with torch.inference_mode():
            t = torch.from_numpy(np.ascontiguousarray(x)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            f = self.model((t - self.mean) / self.std)
            return torch.nn.functional.normalize(f, dim=1).cpu().numpy().astype(np.float32)


# ----------------------------------------------------------------------------- main pass
def write_tracker_cfg(out_dir: Path, settings: Settings) -> Path:
    cfg = f"""tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: {settings.track_buffer}
match_thresh: 0.8
fuse_score: True
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.8
with_reid: False
model: auto
"""
    p = out_dir / "tracker.yaml"
    p.write_text(cfg, encoding="utf-8")
    return p


def run_perception(video: str | Path, out_dir: Path, settings: Settings, progress: Progress | None = None) -> VideoMeta:
    from ultralytics import YOLO
    import torch

    progress = progress or (lambda f, m: None)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "thumbs").mkdir(exist_ok=True)
    (out_dir / "keyframes").mkdir(exist_ok=True)
    meta = probe(video, settings.target_fps)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    progress(0.0, f"Loading {settings.model} on {device}")
    model = YOLO(settings.model)
    embedder = Embedder(device)
    tracker = write_tracker_cfg(out_dir, settings)

    act_zones = [z for z in settings.zones if z.kind == "activity"]
    small_w = 320
    small_h = max(1, int(meta.height * small_w / max(meta.width, 1)))
    masks = {}
    for z in act_zones:
        m = np.zeros((small_h, small_w), np.uint8)
        cv2.fillPoly(m, [np.array([[x * small_w, y * small_h] for x, y in z.points], np.int32)], 1)
        masks[z.name] = m.astype(bool)

    rows, series = [], []
    feats: dict[int, dict] = {}
    best_thumb: dict[int, float] = {}
    prev_small = None
    next_keyframe = 0.0
    cap = cv2.VideoCapture(str(video))
    idx, t0 = 0, time.time()
    total = max(meta.frames, 1)
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % meta.stride:
            idx += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        t = idx / meta.fps

        # ---- motion signals (cheap, every sample)
        small = cv2.cvtColor(cv2.resize(frame, (small_w, small_h)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        rec = {"t": round(t, 3), "frame": idx, "brightness": float(small.mean())}
        b, g, r = [float(c.mean()) for c in cv2.split(cv2.resize(frame, (small_w, small_h)))]
        rec["redness"] = r - (b + g) / 2
        if prev_small is not None:
            diff = np.abs(small - prev_small)
            rec["motion"] = float(diff.mean())
            for name, m in masks.items():
                rec[f"act::{name}"] = float(diff[m].mean()) if m.any() else 0.0
        else:
            rec["motion"] = 0.0
            for name in masks:
                rec[f"act::{name}"] = 0.0
        prev_small = small
        series.append(rec)

        if settings.captions_every > 0 and t >= next_keyframe:
            kf = cv2.resize(frame, (640, int(640 * meta.height / max(meta.width, 1))))
            cv2.imwrite(str(out_dir / "keyframes" / f"{t:09.2f}.jpg"), kf, [cv2.IMWRITE_JPEG_QUALITY, 80])
            next_keyframe = t + settings.captions_every

        # ---- detection + tracking
        res = model.track(frame, persist=True, tracker=str(tracker), conf=settings.conf, imgsz=settings.imgsz,
                          classes=list(TRACK_CLASSES), verbose=False, device=device)[0]
        boxes = res.boxes
        if boxes is not None and boxes.id is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().numpy()
            cls = boxes.cls.int().cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            crops, crop_ids = [], []
            for (x1, y1, x2, y2), tid, c, cf in zip(xyxy, ids, cls, conf):
                tid = int(tid)
                rows.append({"t": round(t, 3), "frame": idx, "track": tid, "cls": TRACK_CLASSES.get(int(c), str(c)),
                             "conf": round(float(cf), 3), "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)})
                xi1, yi1 = max(0, int(x1)), max(0, int(y1))
                xi2, yi2 = min(meta.width, int(x2)), min(meta.height, int(y2))
                crop = frame[yi1:yi2, xi1:xi2]
                if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 6:
                    continue
                f = feats.setdefault(tid, {"hsv": [], "cnn": [], "last_t": -1e9})
                if t - f["last_t"] >= 0.5 and len(f["hsv"]) < 40:
                    f["hsv"].append(hsv_hist(crop))
                    f["last_t"] = t
                    crops.append(crop)
                    crop_ids.append(tid)
                score = float(cf) * (xi2 - xi1) * (yi2 - yi1)
                if score > best_thumb.get(tid, 0):
                    best_thumb[tid] = score
                    th = crop if crop.shape[0] <= 256 else cv2.resize(crop, (int(crop.shape[1] * 256 / crop.shape[0]), 256))
                    cv2.imwrite(str(out_dir / "thumbs" / f"track_{tid}.jpg"), th)
            emb = embedder(crops)
            if emb is not None:
                for tid, e in zip(crop_ids, emb):
                    feats[tid]["cnn"].append(e)
        if (idx // meta.stride) % 10 == 0:
            el = time.time() - t0
            frac = idx / total
            eta = el / max(frac, 1e-6) * (1 - frac)
            progress(min(frac, 0.99), f"Tracking {t:5.1f}s / {meta.duration:.1f}s · {len(rows):,} detections · ETA {eta:.0f}s")
        idx += 1
    cap.release()

    pd.DataFrame(rows, columns=["t", "frame", "track", "cls", "conf", "x1", "y1", "x2", "y2"]).to_csv(
        out_dir / "detections.csv", index=False)
    pd.DataFrame(series).to_csv(out_dir / "series.csv", index=False)
    np.savez_compressed(out_dir / "appearance.npz", **{
        f"{tid}__hsv": np.mean(f["hsv"], axis=0) for tid, f in feats.items() if f["hsv"]}, **{
        f"{tid}__cnn": np.mean(f["cnn"], axis=0) for tid, f in feats.items() if f["cnn"]})
    (out_dir / "meta.json").write_text(json.dumps({**meta.__dict__, "device": device, "model": settings.model,
                                                   "processing_seconds": round(time.time() - t0, 1)}, indent=2))
    progress(1.0, f"Perception done in {time.time() - t0:.0f}s")
    return meta
