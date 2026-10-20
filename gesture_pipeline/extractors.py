"""Landmark extractors plug in here. Each one turns a Frame into rows in the LandmarkWriter.

Concrete extractors (MediaPipe pose/face/hand, YOLO tracking) are registered in
EXTRACTORS once the footage layout is confirmed. Keeping them behind this small
interface means the rest of the pipeline never changes when a model is swapped.
"""

from __future__ import annotations

from typing import Callable, Protocol

from .config import Config
from .frames import Frame
from .store import LandmarkWriter


class Extractor(Protocol):
    name: str

    def process(self, frame: Frame, writer: LandmarkWriter) -> None: ...

    def close(self) -> None: ...


EXTRACTORS: dict[str, Callable[[Config], Extractor]] = {}


def register(name: str):
    def deco(factory: Callable[[Config], Extractor]):
        EXTRACTORS[name] = factory
        return factory
    return deco


def _load_builtin() -> None:
    # Imported lazily so heavy ML libraries load only when actually used.
    from . import extractors_yolo  # noqa: F401


def build_extractors(cfg: Config) -> list[Extractor]:
    if any(m not in EXTRACTORS for m in cfg.extract.models):
        _load_builtin()
    unknown = [m for m in cfg.extract.models if m not in EXTRACTORS]
    if unknown:
        raise ValueError(f"Unknown extract.models {unknown}. Available: {sorted(EXTRACTORS)}")
    return [EXTRACTORS[m](cfg) for m in cfg.extract.models]
