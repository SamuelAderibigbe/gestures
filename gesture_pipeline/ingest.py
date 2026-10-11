"""Stage 1: inspect videos (ffprobe) and extract a clean mono audio track (ffmpeg)."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

log = logging.getLogger(__name__)


class ToolMissingError(RuntimeError):
    pass


def require_tool(name: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise ToolMissingError(
            f"'{name}' was not found on PATH. Install FFmpeg "
            "(macOS: `brew install ffmpeg`, Ubuntu: `sudo apt install ffmpeg`)."
        )
    return exe


@dataclass
class VideoMeta:
    path: str
    duration_s: float
    width: int
    height: int
    fps: float
    n_frames: int | None
    video_codec: str
    has_audio: bool
    audio_codec: str | None
    audio_sample_rate: int | None
    rotation: int
    size_bytes: int

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_rate(rate: str | None) -> float:
    if not rate or rate in ("0/0", "0"):
        return 0.0
    try:
        return float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path: Path) -> VideoMeta:
    """Read container/stream metadata. Raises ValueError for unreadable or video-less files."""
    exe = require_tool("ffprobe")
    proc = subprocess.run(
        [exe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise ValueError(f"ffprobe could not read {path.name}: {proc.stderr.strip()[:500]}")
    info = json.loads(proc.stdout or "{}")
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and s.get("disposition", {}).get("attached_pic", 0) == 0), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ValueError(f"{path.name} has no video stream")

    fmt = info.get("format", {})
    duration = float(fmt.get("duration") or video.get("duration") or 0.0)
    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
    n_frames = int(video["nb_frames"]) if str(video.get("nb_frames", "")).isdigit() else None

    rotation = 0
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    rotation = int(video.get("tags", {}).get("rotate", rotation))

    return VideoMeta(
        path=str(path),
        duration_s=round(duration, 3),
        width=int(video.get("width", 0)),
        height=int(video.get("height", 0)),
        fps=round(fps, 3),
        n_frames=n_frames,
        video_codec=video.get("codec_name", "unknown"),
        has_audio=audio is not None,
        audio_codec=audio.get("codec_name") if audio else None,
        audio_sample_rate=int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
        rotation=rotation,
        size_bytes=int(fmt.get("size") or path.stat().st_size),
    )


def extract_audio(video: Path, out_wav: Path, sample_rate: int = 16000) -> Path:
    """Write 16-bit PCM mono WAV at `sample_rate` (what VAD/diarization models expect)."""
    exe = require_tool("ffmpeg")
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_wav.with_suffix(".tmp.wav")
    proc = subprocess.run(
        [exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
         "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(tmp)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg audio extraction failed for {video.name}: {proc.stderr.strip()[:500]}")
    tmp.replace(out_wav)
    return out_wav
