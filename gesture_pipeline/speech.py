"""Speech detection and talk time.

1. Speech detection (Silero VAD): when is anyone talking? Works on Mandarin and English,
   runs on CPU in about 1% of real time. The model ships inside the `silero-vad` package,
   so nothing is downloaded at run time.

2. Attribution: who is talking? For each moment of speech, in order:
     mic        - exactly one person is holding a microphone to their mouth -> them
                  (if several, the one holding it most firmly in that moment)
     presenter  - during the presentation part, a presenter is on screen -> the presenter
                  (podium microphones are not hand-held, so no mic cue there)
     unattributed - otherwise (an off-camera speaker, someone far back, a voice over)
   Unattributed speech is reported, never guessed, so totals stay honest.
"""

from __future__ import annotations

import logging
import warnings
import wave
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SpeechConfig

log = logging.getLogger(__name__)

SPEECH_COLUMNS = ["video_id", "start_s", "end_s", "duration_s"]
TALK_COLUMNS = ["video_id", "person_id", "role", "start_s", "end_s", "duration_s", "method"]
# start_s/end_s = span of the turn (including short pauses); duration_s = seconds of speech in it

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from silero_vad import load_silero_vad
            _MODEL = load_silero_vad()
    return _MODEL


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as f:
        if f.getsampwidth() != 2:
            raise ValueError(f"{path.name}: expected 16-bit PCM audio")
        sr, ch = f.getframerate(), f.getnchannels()
        a = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(1)
    return a, sr


def detect_speech(wav: Path, cfg: SpeechConfig, video_id: str) -> pd.DataFrame:
    import torch
    from silero_vad import get_speech_timestamps

    audio, sr = read_wav(wav)
    if sr not in (8000, 16000):
        raise ValueError(f"speech detection needs 8 or 16 kHz audio, got {sr}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = get_speech_timestamps(torch.from_numpy(audio), _model(), sampling_rate=sr,
                                   threshold=cfg.threshold, min_speech_duration_ms=cfg.min_speech_ms,
                                   min_silence_duration_ms=cfg.min_silence_ms, return_seconds=True)
    rows = [{"video_id": video_id, "start_s": round(float(x["start"]), 2), "end_s": round(float(x["end"]), 2),
             "duration_s": round(float(x["end"] - x["start"]), 2)} for x in ts]
    out = pd.DataFrame(rows, columns=SPEECH_COLUMNS)
    log.info("speech: %.0f s in %d segments (%.0f%% of %.0f s audio)", out.duration_s.sum(), len(out),
             100 * out.duration_s.sum() / max(len(audio) / sr, 1e-6), len(audio) / sr)
    return out


def attribute_speech(speech: pd.DataFrame, events: pd.DataFrame, people: pd.DataFrame,
                     phases: pd.DataFrame, rows: pd.DataFrame, cfg: SpeechConfig,
                     video_id: str, step: float = 0.125) -> pd.DataFrame:
    """Split every speech segment into who was talking. `rows` = one row per person per
    frame (pid, t) so we know who is on screen when."""
    if speech.empty:
        return pd.DataFrame(columns=TALK_COLUMNS)
    key_col = "role_key" if "role_key" in people.columns else "role"
    role_key = dict(zip(people.person_id, people[key_col])) if len(people) else {}
    role_lbl = dict(zip(people.person_id, people.role)) if len(people) else {}
    mic = events[events.gesture == "holding_mic"] if len(events) else pd.DataFrame(columns=["person_id", "start_s", "end_s", "confidence"])
    presenters = {p for p, k in role_key.items() if k == "presenter"}
    pres_t = np.sort(rows[rows.pid.isin(presenters)].t.unique()) if len(rows) and presenters else np.array([])

    def presenter_at(t: float) -> int | None:
        if not cfg.presenter_fallback or not len(pres_t):
            return None
        if not len(phases) or not (phases[(phases.start_s <= t) & (phases.end_s >= t)].phase == "presentation").any():
            return None
        i = np.searchsorted(pres_t, t)
        near = [pres_t[j] for j in (i - 1, i) if 0 <= j < len(pres_t)]
        if not near or min(abs(x - t) for x in near) > 0.5:
            return None
        here = rows[(rows.t - t).abs() <= 0.5]
        cands = [p for p in here.pid.unique() if p in presenters]
        return int(cands[0]) if len(cands) == 1 else None

    # (time, speaker, method, seconds of speech this step really covers)
    labels: list[tuple[float, int, str, float]] = []
    for _, seg in speech.iterrows():
        for t in np.arange(seg.start_s, seg.end_s, step):
            active = mic[(mic.start_s <= t) & (mic.end_s >= t)]
            if len(active) >= 1:
                pid, how = int(active.sort_values("confidence").iloc[-1].person_id), "mic"
            elif (p := presenter_at(t)) is not None:
                pid, how = p, "presenter"
            else:
                pid, how = -1, "unattributed"
            labels.append((float(t), pid, how, float(min(step, seg.end_s - t))))

    # Group consecutive steps by the same speaker into turns. A turn spans short pauses
    # (start_s..end_s), but duration_s counts only the seconds of actual speech in it.
    turns: list[list] = []
    for t, pid, how, sec in labels:
        if turns and turns[-1][0] == pid and turns[-1][3] == how and t - turns[-1][2] <= cfg.merge_gap_s:
            turns[-1][2] = t + sec
            turns[-1][4] += sec
        else:
            turns.append([pid, t, t + sec, how, sec])
    out = [{"video_id": video_id, "person_id": pid,
            "role": role_lbl.get(pid, "(unattributed)") if pid >= 0 else "(unattributed)",
            "start_s": round(s, 2), "end_s": round(e, 2), "duration_s": round(sec, 2), "method": how}
           for pid, s, e, how, sec in turns if sec >= cfg.min_turn_s]
    talk = pd.DataFrame(out, columns=TALK_COLUMNS)
    if len(talk):
        attributed = talk[talk.person_id >= 0].duration_s.sum()
        log.info("talk: %.0f s attributed to people, %.0f s unattributed",
                 attributed, talk[talk.person_id < 0].duration_s.sum())
    return talk


TALK_SUMMARY_COLUMNS = ["video_id", "person_id", "role", "talk_s", "turns", "visible_min",
                        "talk_share_of_visible", "talk_per_visible_min_s", "method"]


def summarize_talk(talk: pd.DataFrame, people: pd.DataFrame, visibility: pd.DataFrame) -> pd.DataFrame:
    """Talk time per person (people who never spoke get 0) plus one '(unattributed)' row per video."""
    rows = []
    vis = visibility.groupby(["video_id", "person_id"]).visible_s.sum() if len(visibility) else None
    for _, p in people.iterrows():
        mine = talk[(talk.video_id == p.video_id) & (talk.person_id == p.person_id)] if len(talk) else talk
        visible_s = float(vis.get((p.video_id, p.person_id), 0.0)) if vis is not None else float(p.get("visible_s", 0))
        tsec = float(mine.duration_s.sum()) if len(mine) else 0.0
        rows.append({"video_id": p.video_id, "person_id": int(p.person_id), "role": p.role,
                     "talk_s": round(tsec, 1), "turns": int(len(mine)), "visible_min": round(visible_s / 60, 2),
                     "talk_share_of_visible": round(tsec / visible_s, 3) if visible_s else 0.0,
                     "talk_per_visible_min_s": round(tsec / (visible_s / 60), 1) if visible_s else 0.0,
                     "method": ",".join(sorted(mine.method.unique())) if len(mine) else ""})
    if len(talk):
        for vid, g in talk[talk.person_id < 0].groupby("video_id"):
            rows.append({"video_id": vid, "person_id": -1, "role": "(unattributed)",
                         "talk_s": round(float(g.duration_s.sum()), 1), "turns": int(len(g)),
                         "visible_min": None, "talk_share_of_visible": None, "talk_per_visible_min_s": None,
                         "method": "unattributed"})
    return pd.DataFrame(rows, columns=TALK_SUMMARY_COLUMNS)
