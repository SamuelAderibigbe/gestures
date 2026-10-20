"""Helpers to turn stored landmark rows into numpy arrays of COCO-17 keypoints."""

from __future__ import annotations

import numpy as np
import pandas as pd

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16
N_KP = 17


def stack(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xy[n,17,2], conf[n,17], bbox[n,4]) for rows of the pose table."""
    n = len(df)
    xy = np.full((n, N_KP, 2), np.nan, dtype=np.float32)
    cf = np.zeros((n, N_KP), dtype=np.float32)
    for i, (x, y, v) in enumerate(zip(df["x"], df["y"], df["visibility"])):
        if x is not None and len(x) == N_KP:
            xy[i, :, 0], xy[i, :, 1] = x, y
            cf[i] = v if v is not None else 1.0
    bbox = np.stack(df["bbox"].to_numpy()).astype(np.float32) if n else np.zeros((0, 4), np.float32)
    return xy, cf, bbox


def masked(xy: np.ndarray, cf: np.ndarray, thr: float) -> np.ndarray:
    """Keypoints with confidence below thr set to NaN."""
    out = xy.copy()
    out[cf < thr] = np.nan
    return out


def body_scale(m: np.ndarray, bbox: np.ndarray) -> tuple[float, float]:
    """Per-person scale: S = median shoulder width, H = median nose-to-shoulder height.
    Falls back to proportions of the box when keypoints are missing."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        shw = np.abs(m[:, L_SHOULDER, 0] - m[:, R_SHOULDER, 0])
        S = float(np.nanmedian(shw)) if np.isfinite(shw).any() else np.nan
        sh_y = np.nanmean(m[:, [L_SHOULDER, R_SHOULDER], 1], 1)
        hh = sh_y - m[:, NOSE, 1]
        H = float(np.nanmedian(hh)) if np.isfinite(hh).any() else np.nan
        bh = float(np.nanmedian(bbox[:, 3] - bbox[:, 1])) if len(bbox) else 0.1
    if not np.isfinite(S) or S <= 0:
        S = 0.2 * bh
    if not np.isfinite(H) or H <= 0:
        H = 0.6 * S
    return S, H


def mic_mask(m: np.ndarray) -> np.ndarray:
    """Holding a microphone to the mouth: a forearm points up (wrist above elbow), the
    wrist is in front of the face horizontally and between chin and chest height.
    (A raised hand goes above the head; hands resting on a table are below the elbow.)"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sh_y = np.nanmean(m[:, [L_SHOULDER, R_SHOULDER], 1], 1)
        nose_y = m[:, NOSE, 1]
        head = sh_y - nose_y
        shw = np.abs(m[:, L_SHOULDER, 0] - m[:, R_SHOULDER, 0])
        mic = np.zeros(len(m), dtype=bool)
        with np.errstate(invalid="ignore", divide="ignore"):
            for e, w in ((L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST)):
                fl = np.linalg.norm(m[:, w] - m[:, e], axis=1)
                up = (m[:, e, 1] - m[:, w, 1]) / fl
                mic |= ((up > 0.5) & (m[:, w, 1] > nose_y - 0.2 * head) & (m[:, w, 1] < sh_y + 1.5 * head)
                        & (np.abs(m[:, w, 0] - m[:, NOSE, 0]) < shw))
    # Only when the face is visible: from behind, the pose model guesses nose and wrists.
    face = np.isfinite(m[:, [NOSE, L_EYE, R_EYE], 0]).all(1)
    return mic & face


def standing_mask(m: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Standing: thigh length >= 0.6 x torso length (seated thighs point at the camera and
    look short); if knees are hidden, box height >= 2.6 x width."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sh_y = np.nanmean(m[:, [L_SHOULDER, R_SHOULDER], 1], 1)
        hip_y = np.nanmean(m[:, [L_HIP, R_HIP], 1], 1)
        knee_y = np.nanmean(m[:, [L_KNEE, R_KNEE], 1], 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            thigh = (knee_y - hip_y) / (hip_y - sh_y)
            aspect = (bbox[:, 3] - bbox[:, 1]) / np.maximum(bbox[:, 2] - bbox[:, 0], 1e-6)
    return np.where(np.isfinite(thigh), thigh > 0.6, aspect > 2.6)
