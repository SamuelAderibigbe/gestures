"""Run the stages for one video or a whole batch, resumably."""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from .tables import read_table
from tqdm import tqdm

import json

from . import __version__, export, ingest
from .debug_video import render as render_debug
from .gestures import detect_events, map_hand_ids, phase_at
from .measures import facing_and_hands, timeline
from .people import assign_people, build_people
from .scene import detect_screen
from .speech import (SPEECH_COLUMNS, TALK_SUMMARY_COLUMNS, attribute_speech, clip_to_window, detect_speech,
                     summarize_talk)
from .config import Config
from .extractors import build_extractors
from .frames import FrameSampler
from .paths import VideoPaths, video_id_for
from .status import StageStatus, atomic_write_json, file_fingerprint, settings_fingerprint
from .store import LandmarkWriter, read_frames, read_landmarks

log = logging.getLogger(__name__)

STAGES = ["probe", "audio", "speech", "extract", "people", "events", "talk", "debug"]
DEFAULT_STAGES = ["probe", "audio", "speech", "extract", "people", "events", "talk"]  # debug video is opt-in


def discover_videos(inputs: Iterable[str | Path], extensions: list[str]) -> list[Path]:
    exts = {e.lower() for e in extensions}
    found: list[Path] = []
    for item in inputs:
        p = Path(item).expanduser()
        if p.is_dir():
            found += [f for f in sorted(p.rglob("*")) if f.is_file() and f.suffix.lower() in exts
                      and not f.name.startswith(".")]
        elif p.is_file():
            found.append(p)
        else:
            log.warning("Input not found, skipping: %s", p)
    unique, seen = [], set()
    for f in found:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            unique.append(f)
    return unique


def person_visibility(rows: pd.DataFrame, phases: pd.DataFrame, fps: float, video_id: str) -> pd.DataFrame:
    """Seconds each person is visible in each phase (denominator for per-minute rates)."""
    cols = ["video_id", "person_id", "phase", "visible_s"]
    if rows.empty:
        return pd.DataFrame(columns=cols)
    r = rows[["pid", "t"]].copy()
    r["phase"] = [phase_at(phases, t) for t in r.t]
    out = r.groupby(["pid", "phase"]).size().mul(1.0 / fps).round(1).reset_index(name="visible_s")
    out = out.rename(columns={"pid": "person_id"})
    out.insert(0, "video_id", video_id)
    return out[cols]


@dataclass
class VideoResult:
    video: str
    video_id: str
    ok: bool = True
    stages_run: list[str] = field(default_factory=list)
    stages_skipped: list[str] = field(default_factory=list)
    error: str = ""
    seconds: float = 0.0


def process_video(video: Path, cfg: Config, stages: list[str] | None = None, force: bool = False,
                  start: float = 0.0, end: float | None = None) -> VideoResult:
    stages = stages or DEFAULT_STAGES
    vid = video_id_for(video)
    paths = VideoPaths.for_video(Path(cfg.paths.output_dir), vid).ensure()
    status = StageStatus(paths.status)
    res = VideoResult(str(video), vid)
    t0 = time.time()

    fp_file = file_fingerprint(video)
    fps = {
        "probe": fp_file,
        "audio": settings_fingerprint({"f": fp_file, "sr": cfg.audio.sample_rate}),
        "extract": settings_fingerprint({"f": fp_file, "s": cfg.sampling.__dict__,
                                         "e": cfg.extract.__dict__, "win": [start, end],
                                         "hands_v": 2 if cfg.extract.detect_hands else 0}),
    }
    fps["people"] = settings_fingerprint({"x": fps["extract"], "p": cfg.people.__dict__, "v": __version__})
    fps["events"] = settings_fingerprint({"x": fps["people"], "e": cfg.events.__dict__,
                                         "p": cfg.events.resolved_params(), "v": __version__})
    fps["speech"] = settings_fingerprint({"a": fps["audio"], "s": cfg.speech.__dict__, "v": __version__,
                                          "win": [start, end]})
    fps["talk"] = settings_fingerprint({"e": fps["events"], "s": fps["speech"]})
    fps["debug"] = settings_fingerprint({"x": fps["talk"]})

    def run(stage: str, fn) -> None:
        if stage not in stages:
            return
        if not force and status.is_done(stage, fps[stage]):
            res.stages_skipped.append(stage)
            return
        log.info("[%s] %s ...", vid, stage)
        ts = time.time()
        status.mark(stage, "running", fps[stage])
        try:
            extra = fn() or {}
        except Exception as exc:
            status.mark(stage, "failed", fps[stage], error=f"{type(exc).__name__}: {exc}")
            raise
        status.mark(stage, "done", fps[stage], seconds=round(time.time() - ts, 1), **extra)
        res.stages_run.append(stage)

    def do_probe():
        meta = ingest.probe(video)
        atomic_write_json(paths.metadata, {"video_id": vid, **meta.to_dict()})
        return {}

    def do_audio():
        if not cfg.audio.extract:
            return {"skipped": "disabled"}
        meta = ingest.probe(video)
        if not meta.has_audio:
            log.warning("[%s] no audio track", vid)
            return {"skipped": "no audio track"}
        ingest.extract_audio(video, paths.audio, cfg.audio.sample_rate)
        return {}

    def do_extract():
        extractors = build_extractors(cfg)
        sampler = FrameSampler(video, cfg.sampling.fps, cfg.sampling.max_width, start, end)
        n = 0
        try:
            with LandmarkWriter(paths.landmarks_dir, paths.frames_index) as writer:
                for frame in tqdm(sampler, total=sampler.expected_count(), unit="f",
                                  desc=vid[:30], leave=False, mininterval=2):
                    writer.add_frame(frame.index, frame.source_index, frame.t)
                    for ex in extractors:
                        ex.process(frame, writer)
                    n += 1
                counts = writer.counts()
        finally:
            for ex in extractors:
                ex.close()
        if n == 0:
            raise RuntimeError("no frames could be decoded")
        return {"frames": n, "rows": counts}

    def load_people():
        pose = read_landmarks(paths.landmarks_dir, "pose")
        mapping = {int(k): v for k, v in json.loads(paths.person_map.read_text()).items()} \
            if paths.person_map.exists() else {}
        people = read_table(paths.people) if paths.people.exists() and paths.people.stat().st_size else pd.DataFrame()
        return pose, mapping, people

    def do_people():
        pose = read_landmarks(paths.landmarks_dir, "pose")
        meta = json.loads(paths.metadata.read_text()) if paths.metadata.exists() else {}
        aspect = meta["height"] / meta["width"] if meta.get("width") else 9 / 16
        screen = detect_screen(video) if cfg.people.detect_screen else None
        atomic_write_json(paths.scene, {"screen": screen, "aspect": aspect})
        people, mapping = build_people(pose, cfg.people, vid, cfg.sampling.fps, aspect, screen)
        atomic_write_json(paths.person_map, {str(k): v for k, v in mapping.items()})
        people.to_csv(paths.people, index=False, encoding="utf-8-sig")
        return {"people": int(len(people)),
                "roles": people.role.value_counts().to_dict() if len(people) else {}}

    def do_speech():
        if not cfg.speech.enabled:
            return {"skipped": "disabled"}
        if not paths.audio.exists():
            pd.DataFrame(columns=SPEECH_COLUMNS).to_csv(paths.speech, index=False)
            return {"skipped": "no audio"}
        sp = detect_speech(paths.audio, cfg.speech, vid)
        # Only keep speech inside the part of the video being analysed (--start/--end), so a
        # 2-minute test run is not compared with an hour of speech.
        sp = clip_to_window(sp, start, end)
        sp.to_csv(paths.speech, index=False, encoding="utf-8-sig")
        return {"speech_s": round(float(sp.duration_s.sum()), 1), "segments": int(len(sp))}

    def do_talk():
        speech = read_table(paths.speech) if paths.speech.exists() else pd.DataFrame(columns=SPEECH_COLUMNS)
        pose, mapping, people = load_people()
        rows = assign_people(pose, mapping)
        events = read_table(paths.events) if paths.events.exists() else pd.DataFrame()
        phases = read_table(paths.phases) if paths.phases.exists() else pd.DataFrame()
        visibility = read_table(paths.visibility) if paths.visibility.exists() else pd.DataFrame()
        talk = attribute_speech(speech, events, people, phases, rows, cfg.speech, vid)
        talk.to_csv(paths.talk, index=False, encoding="utf-8-sig")
        summary = summarize_talk(talk, people, visibility) if len(people) or len(talk) \
            else pd.DataFrame(columns=TALK_SUMMARY_COLUMNS)
        summary.to_csv(paths.talk_summary, index=False, encoding="utf-8-sig")
        att = float(talk[talk.person_id >= 0].duration_s.sum()) if len(talk) else 0.0
        return {"attributed_s": round(att, 1),
                "unattributed_s": round(float(talk[talk.person_id < 0].duration_s.sum()) if len(talk) else 0.0, 1)}

    def do_debug():
        pose, mapping, people = load_people()
        events = read_table(paths.events) if paths.events.exists() else None
        talk = read_table(paths.talk) if paths.talk.exists() and paths.talk.stat().st_size else None
        render_debug(video, paths.debug_video, pose, mapping, people, cfg.sampling.fps,
                     cfg.sampling.max_width, start, end, events, talk=talk)
        return {}

    def do_events():
        pose, mapping, people = load_people()
        rows = assign_people(pose, mapping)
        frames = read_frames(paths.frames_index) if paths.frames_index.exists() else pd.DataFrame(columns=["frame", "t"])
        hand = read_landmarks(paths.landmarks_dir, "hand") if (paths.landmarks_dir / "hand").exists() else None
        hand = map_hand_ids(hand, mapping)  # tracker ids -> person ids (as in people.csv)
        events, phases = detect_events(rows, people, frames, cfg.events, cfg.sampling.fps, vid, hand_landmarks=hand)
        phases.to_csv(paths.phases, index=False, encoding="utf-8-sig")
        visibility = person_visibility(rows, phases, cfg.sampling.fps, vid)
        visibility.to_csv(paths.visibility, index=False, encoding="utf-8-sig")
        export.write_video_outputs(events, people, visibility, cfg.events.gestures, paths.events, paths.summary)
        hands_tracked = bool(hand is not None and len(hand))
        facing_and_hands(events, people, visibility, hands_tracked).to_csv(paths.facing_hands, index=False,
                                                                         encoding="utf-8-sig")
        timeline(events, people, rows, vid).to_csv(paths.timeline, index=False, encoding="utf-8-sig")
        return {"events": int(len(events)),
                "by_gesture": events.gesture.value_counts().to_dict() if len(events) else {}}

    try:
        run("probe", do_probe)
        run("audio", do_audio)
        run("speech", do_speech)
        run("extract", do_extract)
        run("people", do_people)
        run("events", do_events)
        run("talk", do_talk)
        run("debug", do_debug)
    except Exception as exc:
        res.ok = False
        res.error = f"{type(exc).__name__}: {exc}"
        log.error("[%s] failed: %s", vid, res.error)
        log.debug(traceback.format_exc())
    res.seconds = round(time.time() - t0, 1)
    return res


def run_batch(videos: list[Path], cfg: Config, stages: list[str] | None = None, force: bool = False,
              start: float = 0.0, end: float | None = None) -> pd.DataFrame:
    results = []
    for i, v in enumerate(videos, 1):
        log.info("(%d/%d) %s", i, len(videos), v.name)
        r = process_video(v, cfg, stages, force, start, end)
        results.append(r)
        if not r.ok and not cfg.batch.continue_on_error:
            break
    df = pd.DataFrame([r.__dict__ for r in results])
    out = Path(cfg.paths.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "run_report.csv", index=False, encoding="utf-8-sig")
    return df
