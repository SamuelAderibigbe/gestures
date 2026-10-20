"""Hand tracking + fine-grained gesture recognition using MediaPipe Tasks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import Config
from .store import LandmarkWriter

log = logging.getLogger(__name__)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17), # pinky + palm
]


def classify_finger_gesture(landmarks: list[Any]) -> str:
    """Classify gesture from 21 normalized landmarks (wrist=0)."""
    if len(landmarks) < 21:
        return "neutral"
    w = landmarks[0]

    def dist(a, b) -> float:
        return float(np.hypot(a.x - b.x, a.y - b.y))

    # Extension test: fingertip further from wrist than PIP joint
    thumb_ext = dist(landmarks[4], w) > dist(landmarks[2], w)
    index_ext = dist(landmarks[8], w) > dist(landmarks[6], w)
    mid_ext = dist(landmarks[12], w) > dist(landmarks[10], w)
    ring_ext = dist(landmarks[16], w) > dist(landmarks[14], w)
    pinky_ext = dist(landmarks[20], w) > dist(landmarks[18], w)

    # Fist: all fingers curled
    if not index_ext and not mid_ext and not ring_ext and not pinky_ext:
        return "closed_fist"

    # Pinch detection: thumb tip touching index tip
    if dist(landmarks[4], landmarks[8]) < 0.08:
        return "pinch"

    # Pointing: index extended, other 3 fingers curled
    if index_ext and not mid_ext and not ring_ext and not pinky_ext:
        return "pointing"

    # Counting two: index + middle extended
    if index_ext and mid_ext and not ring_ext and not pinky_ext:
        return "counting_two"

    # Open palm: all 4 fingers extended
    if index_ext and mid_ext and ring_ext and pinky_ext:
        return "open_palm"

    return "neutral"


class HandTracker:
    name = "hand"

    def __init__(self, cfg: Config):
        self.enabled = False
        self.recognizer = None
        model_name = getattr(cfg.extract, "hand_model", "gesture_recognizer.task")
        model_path = Path(cfg.extract.model_dir) / model_name
        if not model_path.exists():
            log.info("Hand model not found at %s. Hand tracking disabled.", model_path)
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=getattr(cfg.extract, "hand_conf", 0.2),
            )
            self.recognizer = vision.GestureRecognizer.create_from_options(options)
            self.mp = mp
            self.enabled = True
            log.info("Initialized MediaPipe Hand Gesture Recognizer from %s", model_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not initialize MediaPipe HandTracker: %s", exc)
            self.enabled = False

    def process_person_hands(self, image: np.ndarray, frame_index: int, t: float, person_id: int,
                             kxy: np.ndarray, kcf: np.ndarray, shoulder_w_norm: float,
                             writer: LandmarkWriter) -> list[dict[str, Any]]:
        if not self.enabled or self.recognizer is None:
            return []

        h, w = image.shape[:2]
        shoulder_w = shoulder_w_norm * w if shoulder_w_norm > 0 else 0.05 * w
        results = []

        for wrist_idx, elbow_idx, side in [(9, 7, "left"), (10, 8, "right")]:
            if wrist_idx >= len(kcf) or kcf[wrist_idx] < 0.25:
                continue

            wx, wy = float(kxy[wrist_idx, 0]), float(kxy[wrist_idx, 1])
            if elbow_idx < len(kcf) and kcf[elbow_idx] >= 0.25:
                ex, ey = float(kxy[elbow_idx, 0]), float(kxy[elbow_idx, 1])
                vx, vy = wx - ex, wy - ey
                length = float(np.hypot(vx, vy))
            else:
                vx, vy, length = 0.0, 0.0, 0.0

            if length > 5:
                hx = wx + 0.3 * vx
                hy = wy + 0.3 * vy
                R = max(int(0.85 * length), 28)
            else:
                hx, hy = wx, wy
                R = max(int(0.6 * shoulder_w), 28)

            x1 = max(0, int(hx - R))
            y1 = max(0, int(hy - R))
            x2 = min(w, int(hx + R))
            y2 = min(h, int(hy + R))

            crop = image[y1:y2, x1:x2]
            if crop.shape[0] < 16 or crop.shape[1] < 16:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=crop_rgb)
            try:
                out = self.recognizer.recognize(mp_img)
            except Exception:  # noqa: BLE001
                continue

            if not out.hand_landmarks:
                continue

            for lm_list, handedness_list in zip(out.hand_landmarks, out.handedness or [[]]):
                mp_cat = "None"
                if out.gestures:
                    for g in out.gestures:
                        for c in g:
                            if c.category_name != "None":
                                mp_cat = c.category_name
                                break

                # Global coordinates mapped back to full frame
                crop_w = x2 - x1
                crop_h = y2 - y1
                gx = [(x1 + pt.x * crop_w) / w for pt in lm_list]
                gy = [(y1 + pt.y * crop_h) / h for pt in lm_list]
                gz = [pt.z for pt in lm_list]

                # Blend MediaPipe gesture output with finger geometry
                geom_gesture = classify_finger_gesture(lm_list)
                if mp_cat == "Pointing_Up" or geom_gesture == "pointing":
                    final_gesture = "pointing"
                elif mp_cat == "Open_Palm" or geom_gesture == "open_palm":
                    final_gesture = "open_palm"
                elif mp_cat == "Closed_Fist" or geom_gesture == "closed_fist":
                    final_gesture = "closed_fist"
                elif geom_gesture in ("pinch", "counting_two"):
                    final_gesture = geom_gesture
                else:
                    final_gesture = "neutral"

                hand_score = float(handedness_list[0].score) if handedness_list else 0.8
                extra_data = {
                    "side": side,
                    "gesture": final_gesture,
                    "mp_category": mp_cat,
                    "geom_gesture": geom_gesture,
                }
                row = {
                    "frame": frame_index,
                    "t": t,
                    "person_id": person_id,
                    "region": side,
                    "score": round(hand_score, 3),
                    "bbox": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
                    "x": [round(v, 5) for v in gx],
                    "y": [round(v, 5) for v in gy],
                    "z": [round(v, 5) for v in gz],
                    "extra": json.dumps(extra_data),
                }
                writer.add("hand", row)
                results.append(row)

        return results

    def close(self) -> None:
        if self.recognizer is not None:
            try:
                self.recognizer.close()
            except Exception:  # noqa: BLE001
                pass
            self.recognizer = None
