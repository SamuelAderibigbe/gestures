"""Scene analysis: find the projection screen so that people shown *on the slides*
(photos of staff, customers, crowds) are not counted as people in the room."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def detect_screen(video: Path, samples: int = 9, min_area: float = 0.08,
                  max_area: float = 0.6, threshold: int = 230) -> list[float] | None:
    """Return the projection screen as [x1, y1, x2, y2] (0-1), or None if there isn't one.

    The screen is the largest *very* bright area that stays in place through the video.
    Bright pixels are found in several frames spread over the video; the screen is where
    they agree, so changing slide content does not matter. The threshold is high on
    purpose: TWSE's hall has white walls that, at lower thresholds, merge with the screen.
    Calibrated on 6 TWSE videos (5 in the TWSE hall, 1 elsewhere): correct in all.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    masks = []
    for i in range(samples):
        if n > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (i + 0.5) / samples))
        ok, img = cap.read()
        if not ok:
            continue
        small = cv2.resize(img, (320, int(320 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
        masks.append(small.max(axis=2) > threshold)            # bright in any colour channel
    cap.release()
    if not masks:
        return None
    agree = (np.mean(masks, axis=0) > 0.6).astype(np.uint8)
    n_lab, _, stats, _ = cv2.connectedComponentsWithStats(agree, connectivity=4)
    h, w = agree.shape
    for idx in 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA]):
        x, y, bw, bh, area = stats[idx]
        if area < min_area * h * w:
            break
        if bw * bh > max_area * h * w:
            continue                                            # whole-room glare, not a screen
        box = [round(float(v), 4) for v in (x / w, y / h, (x + bw) / w, (y + bh) / h)]
        log.info("Projection screen at %s", box)
        return box
    return None


def fraction_inside(box: list[float], region: list[float]) -> float:
    x1, y1, x2, y2 = box
    ix = max(0.0, min(x2, region[2]) - max(x1, region[0]))
    iy = max(0.0, min(y2, region[3]) - max(y1, region[1]))
    area = max(1e-9, (x2 - x1) * (y2 - y1))
    return ix * iy / area
