import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg not installed")


def make_numbered_video(path: Path, n_frames: int = 72, fps: float = 24.0, size=(320, 240)) -> Path:
    """Each frame's brightness encodes its frame number, so tests can check which frame was sampled."""
    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    assert vw.isOpened()
    for i in range(n_frames):
        img = np.full((h, w, 3), (i * 3) % 256, dtype=np.uint8)
        vw.write(img)
    vw.release()
    return path


def frame_number_of(img: np.ndarray) -> int:
    return int(round(float(img.mean()) / 3))


@pytest.fixture
def numbered_video(tmp_path) -> Path:
    return make_numbered_video(tmp_path / "numbered.avi")


@pytest.fixture
def av_video(tmp_path) -> Path:
    """3 s test pattern with a 440 Hz tone (needs FFmpeg)."""
    if not HAS_FFMPEG:
        pytest.skip("FFmpeg not installed")
    out = tmp_path / "會議 test clip.mp4"  # non-ASCII + space on purpose
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=640x360:rate=25:duration=3",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)],
                   check=True)
    return out
