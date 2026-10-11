"""Where every artefact for a video lives. One folder per video keeps runs resumable."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


def video_id_for(path: Path) -> str:
    """Stable, filesystem-safe id from the file name (keeps CJK characters)."""
    stem = path.stem.strip()
    safe = re.sub(r"[^\w\-.]+", "_", stem, flags=re.UNICODE).strip("._")
    return safe or "video"


@dataclass(frozen=True)
class VideoPaths:
    root: Path

    @classmethod
    def for_video(cls, output_dir: Path, video_id: str) -> "VideoPaths":
        return cls(Path(output_dir) / video_id)

    @property
    def status(self) -> Path:
        return self.root / "status.json"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"

    @property
    def audio(self) -> Path:
        return self.root / "audio.wav"

    @property
    def landmarks_dir(self) -> Path:
        return self.root / "landmarks"

    @property
    def frames_index(self) -> Path:
        return self.root / "frames.parquet"

    @property
    def people(self) -> Path:
        return self.root / "people.csv"

    @property
    def scene(self) -> Path:
        return self.root / "scene.json"

    @property
    def person_map(self) -> Path:
        return self.root / "person_map.json"

    @property
    def events(self) -> Path:
        return self.root / "events.csv"

    @property
    def phases(self) -> Path:
        return self.root / "phases.csv"

    @property
    def visibility(self) -> Path:
        return self.root / "visibility.csv"

    @property
    def speech(self) -> Path:
        return self.root / "speech.csv"

    @property
    def talk(self) -> Path:
        return self.root / "talk.csv"

    @property
    def talk_summary(self) -> Path:
        return self.root / "talk_summary.csv"

    @property
    def summary(self) -> Path:
        return self.root / "summary.csv"

    @property
    def debug_video(self) -> Path:
        return self.root / "debug.mp4"

    def ensure(self) -> "VideoPaths":
        self.root.mkdir(parents=True, exist_ok=True)
        return self
