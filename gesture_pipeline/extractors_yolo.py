"""Person detection + tracking + body keypoints with Ultralytics YOLO26-pose.

One model call per sampled frame gives, for every visible person:
  * a tracking id (ByteTrack), so the same person keeps the same id over time
  * a bounding box
  * 17 COCO keypoints with a confidence each:
      0 nose, 1 L-eye, 2 R-eye, 3 L-ear, 4 R-ear, 5 L-shoulder, 6 R-shoulder,
      7 L-elbow, 8 R-elbow, 9 L-wrist, 10 R-wrist, 11 L-hip, 12 R-hip,
      13 L-knee, 14 R-knee, 15 L-ankle, 16 R-ankle
That is enough for hand raises (wrist vs shoulder/head), head turns and nods
(nose vs eyes/ears), gesturing (wrist motion) and role cues (face visible or not,
standing or seated). Works on small, distant people in wide shots, where
face-mesh models (MediaPipe) do not.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .config import Config
from .extractors import register
from .frames import Frame
from .store import LandmarkWriter

log = logging.getLogger(__name__)

ASSETS = Path(__file__).parent / "assets"
KEYPOINT_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
                  "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
                  "l_knee", "r_knee", "l_ankle", "r_ankle"]
KP = {n: i for i, n in enumerate(KEYPOINT_NAMES)}


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def ensure_weights(name: str, model_dir: Path) -> Path:
    """Keep weights in models/ instead of wherever the process happens to run."""
    p = Path(name)
    if p.exists():
        return p
    target = Path(model_dir) / p.name
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    log.info("Downloading %s (first run only) ...", p.name)
    YOLO(p.name)  # downloads into the current directory
    if Path(p.name).exists():
        shutil.move(p.name, target)
    return target if target.exists() else Path(p.name)


def tracker_path(tracker: str) -> str:
    if tracker == "conference":
        return str(ASSETS / "bytetrack_conference.yaml")
    return tracker


class YoloPoseTracker:
    name = "pose"

    def __init__(self, cfg: Config):
        from ultralytics import YOLO

        ex = cfg.extract
        self.model = YOLO(str(ensure_weights(ex.pose_model, Path(ex.model_dir))))
        self.imgsz = ex.imgsz
        self.conf = ex.conf
        self.device = resolve_device(ex.device)
        self.tracker = tracker_path(ex.tracker)
        self.ignore = ex.ignore_zones
        log.info("Pose model %s on %s, imgsz=%d", ex.pose_model, self.device, self.imgsz)

    def _ignored(self, cx: float, cy: float) -> bool:
        return any(z[0] <= cx <= z[2] and z[1] <= cy <= z[3] for z in self.ignore)

    def process(self, frame: Frame, writer: LandmarkWriter) -> None:
        h, w = frame.image.shape[:2]
        res = self.model.track(frame.image, persist=True, tracker=self.tracker, imgsz=self.imgsz,
                               conf=self.conf, classes=[0], device=self.device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        ids = res.boxes.id.int().cpu().numpy() if res.boxes.id is not None else [-1] * len(boxes)
        kxy = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else None
        kcf = res.keypoints.conf.cpu().numpy() if res.keypoints is not None and res.keypoints.conf is not None else None
        for i, (b, s, tid) in enumerate(zip(boxes, scores, ids)):
            bx = [float(b[0] / w), float(b[1] / h), float(b[2] / w), float(b[3] / h)]
            if self._ignored((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2):
                continue
            row = {"frame": frame.index, "t": frame.t, "person_id": int(tid), "score": float(s), "bbox": bx}
            if kxy is not None:
                row["x"] = (kxy[i][:, 0] / w).round(5).tolist()
                row["y"] = (kxy[i][:, 1] / h).round(5).tolist()
                row["visibility"] = (kcf[i].round(3).tolist() if kcf is not None else [1.0] * len(kxy[i]))
            else:
                row["x"], row["y"] = [], []
            writer.add("pose", row)

    def close(self) -> None:
        # Reset tracker state so the next video starts with fresh ids.
        if getattr(self.model, "predictor", None) is not None:
            self.model.predictor = None


@register("pose")
def _make(cfg: Config) -> YoloPoseTracker:
    return YoloPoseTracker(cfg)
