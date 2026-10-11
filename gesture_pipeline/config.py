"""Typed configuration with defaults; YAML files override only what they set."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    output_dir: str = "outputs"


@dataclass
class SamplingConfig:
    fps: float = 8.0
    max_width: int | None = 1280

    def validate(self) -> None:
        if self.fps <= 0:
            raise ValueError("sampling.fps must be > 0")
        if self.max_width is not None and self.max_width < 64:
            raise ValueError("sampling.max_width must be >= 64 or null")


@dataclass
class AudioConfig:
    extract: bool = True
    sample_rate: int = 16000


@dataclass
class ExtractConfig:
    models: list[str] = field(default_factory=lambda: ["pose"])  # registered extractors to run
    model_dir: str = "models"                 # downloaded model weights are kept here
    pose_model: str = "yolo26n-pose.pt"       # n = fastest; s = more accurate, ~2.5x slower
    imgsz: int = 1280                         # detector input size; 1280 needed for small, distant people
    conf: float = 0.3                         # minimum person confidence
    device: str = "auto"                      # auto | cpu | mps (Apple GPU) | cuda
    tracker: str = "conference"               # "conference" (bundled) or a path to a tracker YAML
    ignore_zones: list[list[float]] = field(default_factory=list)
    # rectangles [x1, y1, x2, y2] in 0-1 frame coordinates; people whose box centre falls
    # inside are dropped (e.g. a picture-in-picture inset or a TV screen showing people)

    def validate(self) -> None:
        if not 0 < self.conf < 1:
            raise ValueError("extract.conf must be between 0 and 1")
        if self.imgsz % 32:
            raise ValueError("extract.imgsz must be a multiple of 32")
        for z in self.ignore_zones:
            if len(z) != 4 or not (0 <= z[0] < z[2] <= 1 and 0 <= z[1] < z[3] <= 1):
                raise ValueError(f"extract.ignore_zones entry {z} must be [x1,y1,x2,y2] within 0-1")


@dataclass
class PeopleConfig:
    min_presence_s: float = 10.0     # people visible for less than this are marked "transient"
    min_detections: int = 3          # tracklets shorter than this are treated as noise
    merge_max_gap_s: float = 300.0   # re-join a lost person if they reappear within this time
    merge_max_dist: float = 0.6      # ... and within this distance (x their own body height)
    face_conf: float = 0.5           # keypoint confidence to count nose/eyes as visible
    edge_margin: float = 0.08        # standing people this close to the left/right edge -> "other" (staff)
    detect_screen: bool = True       # find the projection screen; people shown on slides -> "on_screen"
    zones: list[dict] = field(default_factory=list)
    # optional manual roles: [{role: panel, box: [x1, y1, x2, y2]}, ...] in 0-1 frame coords.
    # A person whose median position falls inside a zone gets that role (overrides auto rules).
    labels: dict[str, str] = field(default_factory=lambda: {
        "presenter": "presenter", "panel": "panel", "audience": "audience",
        "other": "other", "transient": "transient", "on_screen": "on_screen"})
    # rename roles for other kinds of video, e.g. {panel: participant} for a meeting recording

    def validate(self) -> None:
        for z in self.zones:
            if "role" not in z or "box" not in z or len(z["box"]) != 4:
                raise ValueError(f"people.zones entry {z} needs 'role' and 'box: [x1,y1,x2,y2]'")


ROLES = ["presenter", "panel", "audience", "other", "transient", "on_screen"]

# Which gestures are evaluated for which role (canonical role names).
DEFAULT_ROLE_GESTURES: dict[str, list[str]] = {
    "hand_raise":          ["presenter", "panel", "audience", "other"],
    "hand_above_shoulder": ["presenter", "panel", "audience", "other"],
    "holding_mic":         ["presenter", "panel", "audience", "other"],
    "nod":                 ["presenter", "panel"],
    "head_shake":          ["presenter", "panel"],
    "head_down":           ["presenter", "panel"],
    "look_away":           ["presenter", "panel"],
    "gesturing":           ["presenter", "panel"],
    "stand_up":            ["audience"],
}

# Rule thresholds. Sizes are relative to the person's own body:
#   S = shoulder width, H = nose-to-shoulder height. Times in seconds.
DEFAULT_GESTURE_PARAMS: dict[str, float] = {
    "kp_conf": 0.4,                 # keypoints below this confidence are ignored
    "smooth_s": 0.375,              # median smoothing window before rules apply
    "merge_gap_s": 0.5,             # same gesture again within this gap = one event
    "hand_raise_min_s": 1.0,
    "hand_raise_above_nose": 0.5,   # wrist above nose by > this x H
    "above_shoulder_min_s": 0.5,
    "above_shoulder_margin": 0.25,  # wrist above shoulder line by > this x S
    "above_shoulder_margin_back": 0.4,  # stricter when seen from behind (hidden hands get guessed)
    "above_shoulder_elbow_back": 0.6,   # ...and the elbow no lower than this x S below the shoulder
    "nod_min_amp": 0.15,            # head dips by >= this x H relative to shoulders
    "noise_k": 4.0,                 # ... and by >= this x the person's own keypoint jitter
    "nod_max_dip_s": 0.8,           # dip width at half depth; slower dips are glances down, not nods
    "nod_group_gap_s": 1.0,         # dips closer than this form one nod event
    "shake_min_amp": 0.10,          # side-to-side swing >= this x S
    "shake_max_step_s": 0.75,       # max time between left/right extremes
    "shake_min_reversals": 2,
    "head_down_ratio": 0.6,         # nose-to-shoulder height < this x usual
    "head_down_min_s": 3.0,
    "look_away_offset": 0.3,        # head turned > this x S from usual position
    "look_away_min_s": 2.0,
    "gesture_speed": 0.8,           # wrist speed > this x S per second
    "gesture_min_s": 0.5,
    "mic_min_s": 1.0,
    "mic_merge_gap_s": 2.0,         # a mic held continuously with brief dropouts = one event
    "stand_min_s": 1.0,
    "stand_prior_seated_s": 2.0,
    "stand_rise": 0.8,              # shoulders rise by >= this x S (visible even from behind)
    "trend_s": 3.0,                 # window for the person's slowly-changing baseline
}


@dataclass
class EventsConfig:
    exclude_roles: list[str] = field(default_factory=lambda: ["transient", "on_screen"])
    gestures: dict[str, list[str]] = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_ROLE_GESTURES.items()})
    params: dict[str, float] = field(default_factory=dict)   # overrides of DEFAULT_GESTURE_PARAMS
    min_phase_s: float = 20.0       # shorter presentation/Q&A switches are smoothed away

    def resolved_params(self) -> dict[str, float]:
        return {**DEFAULT_GESTURE_PARAMS, **self.params}

    def validate(self) -> None:
        bad = [k for k in self.params if k not in DEFAULT_GESTURE_PARAMS]
        if bad:
            raise KeyError(f"Unknown events.params {bad}. Valid: {sorted(DEFAULT_GESTURE_PARAMS)}")
        bad = [g for g in self.gestures if g not in DEFAULT_ROLE_GESTURES]
        if bad:
            raise KeyError(f"Unknown gestures {bad}. Valid: {sorted(DEFAULT_ROLE_GESTURES)}")
        for g, roles in self.gestures.items():
            wrong = [r for r in roles if r not in ROLES]
            if wrong:
                raise ValueError(f"events.gestures.{g}: unknown roles {wrong}. Valid: {ROLES}")


@dataclass
class BatchConfig:
    video_extensions: list[str] = field(
        default_factory=lambda: [".mp4", ".avi", ".mkv", ".mov", ".webm", ".ts"]
    )
    continue_on_error: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    people: PeopleConfig = field(default_factory=PeopleConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> "Config":
        self.sampling.validate()
        self.extract.validate()
        self.people.validate()
        self.events.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _apply(obj: Any, overrides: dict[str, Any], where: str) -> None:
    """Recursively copy overrides onto a dataclass, rejecting unknown keys (typos)."""
    known = {f.name: f for f in fields(obj)}
    for key, value in overrides.items():
        if key not in known:
            raise KeyError(f"Unknown config key '{where}{key}'. Valid keys: {sorted(known)}")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"Config '{where}{key}' must be a mapping")
            _apply(current, value, f"{where}{key}.")
        else:
            setattr(obj, key, value)


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Config file {path} must contain a mapping at the top level")
        _apply(cfg, data, "")
    return cfg.validate()
