"""Per-video stage tracking so batch runs can resume after a crash.

A stage is skipped on re-run only if it completed AND the input video and the
relevant settings are unchanged (fingerprints match).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON so a crash mid-write never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def file_fingerprint(path: Path) -> str:
    """Cheap identity for a large video: size + hash of the first and last 1 MB.
    Deliberately ignores the modification time, so copying or moving a video does not
    trigger re-analysis; changing its content does."""
    st = path.stat()
    h = hashlib.sha1(f"{st.st_size}".encode())
    with open(path, "rb") as fh:
        h.update(fh.read(1 << 20))
        if st.st_size > (1 << 20):
            fh.seek(max(0, st.st_size - (1 << 20)))
            h.update(fh.read(1 << 20))
    return h.hexdigest()[:16]


def settings_fingerprint(settings: dict[str, Any]) -> str:
    blob = json.dumps(settings, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


class StageStatus:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = {"stages": {}}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
                self.data.setdefault("stages", {})
            except json.JSONDecodeError:
                self.data = {"stages": {}}  # corrupted -> recompute everything

    def is_done(self, stage: str, fingerprint: str) -> bool:
        s = self.data["stages"].get(stage)
        return bool(s and s.get("state") == "done" and s.get("fingerprint") == fingerprint)

    def mark(self, stage: str, state: str, fingerprint: str = "", **extra: Any) -> None:
        self.data["stages"][stage] = {
            "state": state,
            "fingerprint": fingerprint,
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **extra,
        }
        atomic_write_json(self.path, self.data)

    def invalidate_from(self, stages: list[str], first: str) -> None:
        """Mark `first` and every later stage stale (upstream output changed)."""
        if first not in stages:
            return
        for s in stages[stages.index(first):]:
            self.data["stages"].pop(s, None)
        atomic_write_json(self.path, self.data)
