"""Stage 3 storage: landmarks are saved once, then gesture rules re-run on them freely.

Layout (one folder per video):
    landmarks/<model>/part-00000.parquet ...   e.g. model = "pose", "face", "hand"
    frames.parquet                              sampled frame index -> timestamp

One row = one detected person/face/hand in one sampled frame. Landmark
coordinates are stored as lists, normalised to the ORIGINAL frame (0-1), so
results do not depend on the downscaling used during analysis.

Writes go to a temporary folder that is renamed into place only when the stage
finishes, so a crash never leaves a half-written landmark set that looks complete.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

LANDMARK_SCHEMA = pa.schema([
    ("frame", pa.int32()),
    ("t", pa.float64()),
    ("person_id", pa.int32()),        # tracker id; -1 if not tracked yet
    ("region", pa.string()),          # which tile/crop it came from ("full" for whole frame)
    ("score", pa.float32()),          # detector confidence
    ("bbox", pa.list_(pa.float32(), 4)),  # x1, y1, x2, y2 normalised
    ("x", pa.list_(pa.float32())),
    ("y", pa.list_(pa.float32())),
    ("z", pa.list_(pa.float32())),
    ("visibility", pa.list_(pa.float32())),
    ("extra", pa.string()),           # JSON for model-specific outputs (blendshapes, head pose, handedness)
])

FRAME_SCHEMA = pa.schema([
    ("frame", pa.int32()),
    ("source_index", pa.int64()),
    ("t", pa.float64()),
])


class _PartWriter:
    def __init__(self, folder: Path, schema: pa.Schema, rows_per_part: int):
        self.folder = folder
        self.schema = schema
        self.rows_per_part = rows_per_part
        self.buffer: list[dict[str, Any]] = []
        self.parts = 0
        self.rows = 0

    def add(self, row: dict[str, Any]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.rows_per_part:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        pq.write_table(table, self.folder / f"part-{self.parts:05d}.parquet", compression="zstd")
        self.rows += len(self.buffer)
        self.parts += 1
        self.buffer.clear()


class LandmarkWriter:
    """Context manager: `with LandmarkWriter(dir) as w: w.add("pose", {...})`."""

    def __init__(self, landmarks_dir: Path, frames_index: Path, rows_per_part: int = 20_000):
        self.final_dir = Path(landmarks_dir)
        self.tmp_dir = self.final_dir.with_name(self.final_dir.name + ".partial")
        self.frames_index = Path(frames_index)
        self.rows_per_part = rows_per_part
        self._writers: dict[str, _PartWriter] = {}
        self._frames: list[dict[str, Any]] = []

    def __enter__(self) -> "LandmarkWriter":
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        self.tmp_dir.mkdir(parents=True)
        return self

    def add_frame(self, frame: int, source_index: int, t: float) -> None:
        self._frames.append({"frame": frame, "source_index": source_index, "t": t})

    def add(self, model: str, row: dict[str, Any]) -> None:
        w = self._writers.get(model)
        if w is None:
            w = self._writers[model] = _PartWriter(self.tmp_dir / model, LANDMARK_SCHEMA, self.rows_per_part)
        full = {"person_id": -1, "region": "full", "score": None, "bbox": None,
                "z": None, "visibility": None, "extra": None, **row}
        w.add(full)

    def counts(self) -> dict[str, int]:
        return {m: w.rows + len(w.buffer) for m, w in self._writers.items()}

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
            return
        for w in self._writers.values():
            w.flush()
        frames_tmp = self.frames_index.with_suffix(".partial.parquet")
        pq.write_table(pa.Table.from_pylist(self._frames, schema=FRAME_SCHEMA), frames_tmp)
        shutil.rmtree(self.final_dir, ignore_errors=True)
        self.tmp_dir.rename(self.final_dir)
        frames_tmp.replace(self.frames_index)


def available_models(landmarks_dir: Path) -> list[str]:
    d = Path(landmarks_dir)
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


def read_landmarks(landmarks_dir: Path, model: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
    folder = Path(landmarks_dir) / model
    if not folder.exists():
        return pd.DataFrame(columns=list(columns or LANDMARK_SCHEMA.names))
    return pq.read_table(folder, columns=list(columns) if columns else None).to_pandas()


def read_frames(frames_index: Path) -> pd.DataFrame:
    return pq.read_table(frames_index).to_pandas()
