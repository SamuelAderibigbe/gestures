"""Stage 2: read frames at a fixed analysis rate, with accurate timestamps.

Gestures last 0.3-3 s, so 5-10 analysed frames per second is enough and is
3-6x cheaper than analysing every frame of a 30 fps video.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class Frame:
    index: int          # 0, 1, 2 ... over sampled frames
    source_index: int   # frame number in the original video
    t: float            # seconds from start of video
    image: np.ndarray   # BGR, possibly downscaled
    scale: float        # image width / original width


class FrameSampler:
    """Iterate frames at `fps` between `start` and `end` seconds.

    Timestamps come from the decoder (handles variable frame rate). If the
    decoder does not report sane timestamps we fall back to index / fps.
    """

    def __init__(self, path: Path | str, fps: float, max_width: int | None = None,
                 start: float = 0.0, end: float | None = None):
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self.path = Path(path)
        self.fps = float(fps)
        self.max_width = max_width
        self.start = max(0.0, float(start))
        self.end = end

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open {self.path}")
        return cap

    def expected_count(self) -> int | None:
        cap = self._open()
        try:
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        finally:
            cap.release()
        if src_fps <= 0 or n <= 0:
            return None
        duration = n / src_fps
        end = min(self.end, duration) if self.end is not None else duration
        return max(0, math.ceil((end - self.start) * min(self.fps, src_fps)))

    def __iter__(self) -> Iterator[Frame]:
        cap = self._open()
        try:
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            if src_fps <= 0 or src_fps > 1000:
                src_fps = 25.0
            if self.start > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, self.start * 1000.0)

            interval = 1.0 / self.fps
            next_t = self.start
            src_idx = int(round(self.start * src_fps)) - 1
            last_t = -1.0
            out_idx = 0
            eps = 0.25 / src_fps  # tolerance so 8 fps out of 24 fps is exact

            while True:
                if not cap.grab():
                    break
                src_idx += 1
                pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                t_idx = src_idx / src_fps
                # Trust decoder timestamps only if they move forward sensibly.
                t = pts if (pts > last_t and abs(pts - t_idx) < 5.0) else t_idx
                if t < last_t:
                    t = last_t + 1.0 / src_fps
                last_t = t

                if self.end is not None and t > self.end:
                    break
                if t + eps < next_t:
                    continue
                ok, img = cap.retrieve()
                if not ok or img is None:
                    continue
                scale = 1.0
                if self.max_width and img.shape[1] > self.max_width:
                    scale = self.max_width / img.shape[1]
                    img = cv2.resize(img, (self.max_width, int(round(img.shape[0] * scale))),
                                     interpolation=cv2.INTER_AREA)
                yield Frame(out_idx, src_idx, round(t, 4), img, scale)
                out_idx += 1
                # Skip ahead; never build a backlog if the source has gaps.
                next_t += interval
                while next_t <= t + eps:
                    next_t += interval
        finally:
            cap.release()
