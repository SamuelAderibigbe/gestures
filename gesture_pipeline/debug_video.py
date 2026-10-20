"""Render a checking video: every analysed frame with each person's box, id, role and skeleton.

This is how a human verifies the pipeline: open debug.mp4 and see whether
"P3 panel" really stays on the same person and whether roles are right.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .frames import FrameSampler

log = logging.getLogger(__name__)

ROLE_COLORS = {  # BGR
    "presenter": (0, 165, 255), "panel": (60, 200, 60), "audience": (255, 160, 60),
    "other": (180, 180, 180), "transient": (110, 110, 110), "on_screen": (200, 0, 200),
}
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


class _Writer:
    """H.264 via FFmpeg when available (plays everywhere), else OpenCV mp4v."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        self.path, self.proc, self.cv = path, None, None
        w, h = size
        if shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, img: np.ndarray) -> None:
        if self.proc:
            self.proc.stdin.write(img.tobytes())
        else:
            self.cv.write(img)

    def close(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError("ffmpeg failed while writing debug video")
        else:
            self.cv.release()


def render(video: Path, out: Path, pose: pd.DataFrame, mapping: dict[int, int], people: pd.DataFrame,
           fps: float, max_width: int | None, start: float = 0.0, end: float | None = None,
           events: pd.DataFrame | None = None, kp_thr: float = 0.4, talk: pd.DataFrame | None = None) -> Path:
    from .people import assign_people

    key = "role_key" if "role_key" in people.columns else "role"
    roles = dict(zip(people.person_id, people[key])) if not people.empty else {}
    labels = dict(zip(people.person_id, people.role)) if not people.empty else {}
    rows = assign_people(pose, mapping)
    by_frame = {f: g for f, g in rows.groupby("frame")} if not rows.empty else {}
    tmp = out.with_suffix(".partial.mp4")
    writer = None
    try:
        for fr in FrameSampler(video, fps, max_width, start, end):
            img = fr.image
            h, w = img.shape[:2]
            if writer is None:
                writer = _Writer(tmp, fps, (w, h))
            for _, r in by_frame.get(fr.index, pd.DataFrame()).iterrows():
                pid = int(r.pid)
                role = roles.get(pid, "transient")
                col = ROLE_COLORS.get(role, (200, 200, 200))
                x1, y1, x2, y2 = (np.array(r.bbox) * [w, h, w, h]).astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 1)
                label = f"P{pid} {labels.get(pid, role)}"
                cv2.putText(img, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
                if r.x is not None and len(r.x) == 17:
                    pts = np.c_[np.array(r.x) * w, np.array(r.y) * h].astype(int)
                    vis = np.array(r.visibility) >= kp_thr
                    for a, b in SKELETON:
                        if vis[a] and vis[b]:
                            cv2.line(img, tuple(pts[a]), tuple(pts[b]), col, 1, cv2.LINE_AA)
            if events is not None and not events.empty:
                active = events[(events.start_s <= fr.t) & (events.end_s >= fr.t)]
                for j, (_, e) in enumerate(active.iterrows()):
                    cv2.putText(img, f"P{e.person_id}: {e.gesture}", (10, h - 12 - 18 * j),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            if talk is not None and len(talk):
                now = talk[(talk.start_s <= fr.t) & (talk.end_s >= fr.t)]
                if len(now):
                    r0 = now.iloc[0]
                    who = f"P{int(r0.person_id)} ({r0.method})" if r0.person_id >= 0 else "someone off-screen / unknown"
                    cv2.putText(img, f"SPEAKING: {who}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(img, f"SPEAKING: {who}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 80), 2, cv2.LINE_AA)
            mm, ss = divmod(fr.t, 60)
            cv2.putText(img, f"{int(mm):02d}:{ss:05.2f}", (w - 110, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(img)
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(out)
    return out
