"""`gestures demo`: see the whole pipeline work in about a minute, with no video of your own.

It builds a short synthetic "conference" (a presenter, a seated panel of three and two
audience members seen from behind) in which every gesture happens at a known time, then
runs the real role, gesture, debug-video and Excel stages on it and compares what the
program found with what was scripted.

The figures are drawn stick figures, so the person detector (YOLO) is skipped: their
keypoints are written directly, exactly as the detector would store them. Everything
after that, which is where roles and gesture rules live, is the real code.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from .config import Config
from .debug_video import SKELETON, _Writer
from .export import export_batch
from .paths import VideoPaths, video_id_for
from .pipeline import process_video
from .store import LandmarkWriter

log = logging.getLogger(__name__)

W, H_IMG, FPS, SECONDS = 1280, 720, 8, 40

# What the scripted people do, and therefore what the program should find.
EXPECTED = [
    ("presenter", "holding_mic", 1),
    ("presenter", "gesturing", 1),
    ("presenter", "facing_audience", 1),
    ("panel", "nod", 2),
    ("panel", "head_shake", 1),
    ("panel", "head_down", 1),
    ("panel", "facing_audience", 4),
    ("audience", "hand_raise", 1),
    ("audience", "stand_up", 1),
]


def window(t: float, a: float, b: float) -> bool:
    return a <= t < b


def base_pose(cx: float, hy: float, S: float, standing: bool) -> np.ndarray:
    """17 COCO keypoints in pixels for a person with head at (cx, hy), shoulder width S."""
    k = np.zeros((17, 2))
    k[0] = (cx, hy)
    k[1], k[2] = (cx - 0.12 * S, hy - 0.08 * S), (cx + 0.12 * S, hy - 0.08 * S)
    k[3], k[4] = (cx - 0.25 * S, hy - 0.04 * S), (cx + 0.25 * S, hy - 0.04 * S)
    k[5], k[6] = (cx - 0.5 * S, hy + 0.7 * S), (cx + 0.5 * S, hy + 0.7 * S)
    k[7], k[8] = (cx - 0.6 * S, hy + 1.5 * S), (cx + 0.6 * S, hy + 1.5 * S)
    k[9], k[10] = (cx - 0.35 * S, hy + 1.9 * S), (cx + 0.35 * S, hy + 1.9 * S)
    k[11], k[12] = (cx - 0.3 * S, hy + 2.3 * S), (cx + 0.3 * S, hy + 2.3 * S)
    if standing:
        k[9], k[10] = (cx - 0.65 * S, hy + 2.2 * S), (cx + 0.65 * S, hy + 2.2 * S)
        k[13], k[14] = (cx - 0.3 * S, hy + 3.5 * S), (cx + 0.3 * S, hy + 3.5 * S)
        k[15], k[16] = (cx - 0.3 * S, hy + 4.6 * S), (cx + 0.3 * S, hy + 4.6 * S)
    else:
        k[13], k[14] = (cx - 0.3 * S, hy + 2.45 * S), (cx + 0.3 * S, hy + 2.45 * S)
        k[15], k[16] = (cx - 0.3 * S, hy + 3.2 * S), (cx + 0.3 * S, hy + 3.2 * S)
    return k


@dataclass
class Actor:
    track_id: int
    label: str
    cx: float
    hy: float
    S: float
    standing: bool = False
    facing: bool = True
    visible: tuple[float, float] = (0.0, SECONDS)
    moves: list[Callable[[float, np.ndarray, "Actor"], None]] = field(default_factory=list)

    def pose(self, t: float) -> np.ndarray:
        k = base_pose(self.cx, self.hy, self.S, self.standing)
        for mv in self.moves:
            mv(t, k, self)
        return k


# ---- scripted movements (sizes relative to the actor) ---------------------------------

def nod_at(*starts: float):
    def mv(t, k, a):
        for s in starts:
            if window(t, s, s + 1.5):                     # two quick dips = one nod event
                k[:5, 1] += 0.35 * 0.7 * a.S * math.sin(math.pi * ((t - s) % 0.75) / 0.75) ** 2
    return mv


def shake_at(s: float):
    def mv(t, k, a):
        if window(t, s, s + 1.5):
            k[:5, 0] += 0.25 * a.S * math.sin(2 * math.pi * 1.5 * (t - s))
    return mv


def head_down(s: float, e: float):
    def mv(t, k, a):
        if window(t, s, e):
            k[:5, 1] += 0.6 * 0.7 * a.S
    return mv


def mic(s: float, e: float):
    def mv(t, k, a):
        if window(t, s, e):
            k[8] = (a.cx + 0.2 * a.S, a.hy + 1.3 * a.S)
            k[10] = (a.cx + 0.08 * a.S, a.hy + 0.45 * a.S)
    return mv


def gesture(s: float, e: float):
    def mv(t, k, a):
        if window(t, s, e):
            ang = 2 * math.pi * 1.5 * t
            k[7] = (a.cx - 0.7 * a.S, a.hy + 1.4 * a.S)
            k[9] = (a.cx - 0.6 * a.S + 0.4 * a.S * math.cos(ang), a.hy + 1.2 * a.S + 0.4 * a.S * math.sin(ang))
    return mv


def raise_hand(s: float, e: float):
    def mv(t, k, a):
        if window(t, s, e):
            k[8] = (a.cx + 0.6 * a.S, a.hy + 0.6 * a.S)
            k[10] = (a.cx + 0.65 * a.S, a.hy - 1.2 * a.S)
    return mv


def stand_up(s: float):
    def mv(t, k, a):
        if t >= s:
            k[:13, 1] -= 1.5 * a.S
    return mv


def cast() -> list[Actor]:
    return [
        Actor(1, "presenter", 1060, 250, 60, standing=True, visible=(0, 20),
              moves=[mic(2, 18), gesture(8, 12)]),
        Actor(2, "panel A", 420, 420, 64, moves=[nod_at(5, 27)]),
        Actor(3, "panel B", 620, 420, 64, moves=[shake_at(25)]),
        Actor(4, "panel C", 820, 420, 64, moves=[head_down(30, 36)]),
        Actor(5, "audience A", 170, 575, 90, facing=False, moves=[raise_hand(10, 13)]),
        Actor(6, "audience B", 1110, 585, 90, facing=False, moves=[stand_up(32)]),
    ]


def _confidence(actor: Actor, k: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    c = np.clip(rng.normal(0.9, 0.04, 17), 0.5, 1.0)
    if not actor.facing:
        c[:5] = 0.05                                       # back of the head: no face points
        c[13:] = 0.05                                      # legs hidden behind chairs
    return c


def _draw(img: np.ndarray, actor: Actor, k: np.ndarray, conf: np.ndarray) -> None:
    col = (70, 70, 70)
    for a, b in SKELETON:
        if a > 4 and b > 4 and conf[a] > 0.3 and conf[b] > 0.3:
            cv2.line(img, tuple(int(v) for v in k[a]), tuple(int(v) for v in k[b]), col, 6, cv2.LINE_AA)
    head = k[0] if actor.facing else (k[5] + k[6]) / 2 - (0, 0.75 * actor.S)
    cv2.circle(img, tuple(int(v) for v in head), int(0.38 * actor.S), (60, 60, 60) if not actor.facing else (190, 170, 150), -1, cv2.LINE_AA)
    if actor.facing:
        for e in (1, 2):
            cv2.circle(img, tuple(int(v) for v in k[e]), 3, (30, 30, 30), -1)


def build_demo_video(out_dir: Path, seed: int = 7) -> tuple[Path, Path]:
    """Write the synthetic video and its keypoints. Returns (video path, video output dir)."""
    rng = np.random.default_rng(seed)
    video = out_dir / "input" / "demo_conference.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    actors = cast()
    paths = VideoPaths.for_video(out_dir, video_id_for(video)).ensure()
    writer = _Writer(video, FPS, (W, H_IMG))
    try:
        with LandmarkWriter(paths.landmarks_dir, paths.frames_index) as lw:
            for f in range(SECONDS * FPS):
                t = f / FPS
                img = np.full((H_IMG, W, 3), 205, np.uint8)
                cv2.rectangle(img, (300, 40), (980, 330), (250, 250, 250), -1)      # projection screen
                cv2.putText(img, "Q3 results", (520, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (150, 110, 60), 3)
                cv2.rectangle(img, (300, 520), (980, 560), (120, 100, 90), -1)      # panel table
                lw.add_frame(f, f, t)
                for a in actors:
                    if not window(t, *a.visible):
                        continue
                    k = a.pose(t) + rng.normal(0, 0.02 * a.S, (17, 2))
                    conf = _confidence(a, k, rng)
                    _draw(img, a, k, conf)
                    x1, y1 = k[:, 0].min() - 0.3 * a.S, k[:, 1].min() - 0.5 * a.S
                    x2, y2 = k[:, 0].max() + 0.3 * a.S, min(H_IMG - 1, k[:, 1].max() + 0.2 * a.S)
                    lw.add("pose", {"frame": f, "t": t, "person_id": a.track_id, "score": 0.9,
                                    "bbox": [x1 / W, y1 / H_IMG, x2 / W, y2 / H_IMG],
                                    "x": (k[:, 0] / W).tolist(), "y": (k[:, 1] / H_IMG).tolist(),
                                    "visibility": conf.tolist()})
                cv2.putText(img, "SYNTHETIC DEMO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 180), 2)
                writer.write(img)
    finally:
        writer.close()
    return video, paths.root


def compare(out_dir: Path, video_id: str) -> pd.DataFrame:
    vdir = out_dir / video_id
    ev = pd.read_csv(vdir / "events.csv")
    people = pd.read_csv(vdir / "people.csv")
    keys = dict(zip(people.person_id, people.role_key))
    ev["role_key"] = ev.person_id.map(keys)
    rows = []
    for role, gesture, n in EXPECTED:
        found = int(((ev.role_key == role) & (ev.gesture == gesture)).sum())
        rows.append({"who": role, "gesture": gesture, "scripted": n, "found": found,
                     "ok": "yes" if found == n else "NO"})
    extra = ev[~ev.set_index(["role_key", "gesture"]).index.isin([(r, g) for r, g, _ in EXPECTED])]
    for (role, gesture), g in extra.groupby(["role_key", "gesture"]):
        rows.append({"who": role, "gesture": gesture, "scripted": 0, "found": len(g), "ok": "NO"})
    return pd.DataFrame(rows)


def run_demo(out_dir: Path = Path("outputs_demo")) -> tuple[bool, pd.DataFrame, Path]:
    out_dir = Path(out_dir)
    video, vroot = build_demo_video(out_dir)
    cfg = Config()
    cfg.paths.output_dir = str(out_dir)
    cfg.sampling.max_width = None
    res = process_video(video, cfg, stages=["probe", "speech", "people", "events", "talk", "debug"], force=True)
    if not res.ok:
        raise RuntimeError(res.error)
    xlsx = export_batch(out_dir, role_gestures=cfg.events.gestures)
    table = compare(out_dir, res.video_id)
    roles = pd.read_csv(vroot / "people.csv")[["person_id", "role", "role_reason"]]
    log.info("Roles found:\n%s", roles.to_string(index=False))
    return bool((table.ok == "yes").all()), table, xlsx
