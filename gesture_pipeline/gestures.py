"""Gesture rules: turn each person's keypoint time series into gesture events.

Every rule is written to match docs/gestures.md exactly. All sizes are relative to
the person's own body so the same rule works near and far from the camera:
    S = median shoulder width, H = median nose-to-shoulder height (per person).

Input:  one row per person per sampled frame (from people.assign_people), the
        people table (roles) and the frame index (timestamps).
Output: events table (one row per gesture event) and phases table.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .config import EventsConfig
from .export import EVENT_COLUMNS
from .keypoints import (L_ELBOW, L_EYE, L_HIP, L_SHOULDER, L_WRIST, NOSE, R_ELBOW, R_EYE, R_HIP,
                        R_SHOULDER, R_WRIST, body_scale, masked, mic_mask, stack)

log = logging.getLogger(__name__)




# ----------------------------------------------------------------------------- helpers

def _odd(n: float) -> int:
    n = max(1, int(round(n)))
    return n if n % 2 else n + 1


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    """NaN-aware centred rolling median."""
    if k <= 1:
        return x
    return pd.Series(x).rolling(k, center=True, min_periods=1).median().to_numpy()


def trend(x: np.ndarray, k: int) -> np.ndarray:
    """Slow baseline (person's usual position) to subtract before looking for quick moves."""
    return pd.Series(x).rolling(k, center=True, min_periods=max(1, k // 4)).median().to_numpy()


def runs(mask: np.ndarray, min_len: int, merge_gap: int) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs where mask is True, gaps <= merge_gap joined,
    runs shorter than min_len dropped."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    d = np.diff(np.r_[0, mask.astype(int), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    merged = [[starts[0], ends[0]]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - merged[-1][1] - 1 <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(int(s), int(e)) for s, e in merged if e - s + 1 >= min_len]


def majority(mask: np.ndarray, k: int) -> np.ndarray:
    """Remove single-frame flicker: a frame is on if most of its neighbours are."""
    if k <= 1:
        return mask.astype(bool)
    return pd.Series(mask.astype(float)).rolling(k, center=True, min_periods=1).mean().to_numpy() > 0.5


@dataclass
class Track:
    """One person's signals on a regular grid of sampled frames (NaN where unseen)."""
    t: np.ndarray
    m: np.ndarray       # keypoints [n,17,2] with low-confidence points = NaN
    cf: np.ndarray      # keypoint confidence [n,17]
    bbox: np.ndarray    # [n,4]
    present: np.ndarray
    S: float
    H: float
    fps: float

    @property
    def sh_mid(self) -> np.ndarray:
        both = np.isfinite(self.m[:, L_SHOULDER, 0]) & np.isfinite(self.m[:, R_SHOULDER, 0])
        mid = (self.m[:, L_SHOULDER] + self.m[:, R_SHOULDER]) / 2
        mid[~both] = np.nan
        return mid

    @property
    def nose(self) -> np.ndarray:
        return self.m[:, NOSE]

    @property
    def face(self) -> np.ndarray:
        return np.isfinite(self.m[:, [NOSE, L_EYE, R_EYE], 0]).all(1)

    def frames(self, seconds: float) -> int:
        return max(1, int(round(seconds * self.fps)))


def build_track(g: pd.DataFrame, frame_t: pd.Series, fps: float, kp_conf: float) -> Track:
    g = g.sort_values("frame")
    f0, f1 = int(g.frame.iloc[0]), int(g.frame.iloc[-1])
    n = f1 - f0 + 1
    xy, cf, bb = stack(g)
    pos = g.frame.to_numpy() - f0
    M = np.full((n, 17, 2), np.nan, np.float32)
    C = np.zeros((n, 17), np.float32)
    B = np.full((n, 4), np.nan, np.float32)
    P = np.zeros(n, bool)
    M[pos], C[pos], B[pos], P[pos] = xy, cf, bb, True
    t = frame_t.reindex(range(f0, f1 + 1)).interpolate(limit_direction="both").to_numpy()
    m = masked(M, C, kp_conf)
    S, H = body_scale(m[P], B[P])
    return Track(t, m, C, B, P, S, H, fps)


def _conf(track: Track, s: int, e: int, kps: list[int], strength: float) -> float:
    """Event confidence = how well the keypoints were seen x how clearly the rule was met."""
    c = track.cf[s:e + 1][:, kps]
    seen = float(np.mean(c[c > 0])) if (c > 0).any() else 0.0
    return round(float(np.clip(seen * (0.5 + 0.5 * np.clip(strength, 0, 1)), 0, 1)), 3)


def _events_from_runs(track: Track, rr: list[tuple[int, int]], name: str, kps: list[int],
                      strength_fn=None) -> list[dict]:
    out = []
    for s, e in rr:
        strength = strength_fn(s, e) if strength_fn else 1.0
        out.append({"gesture": name, "s": s, "e": e, "confidence": _conf(track, s, e, kps, strength)})
    return out


# ----------------------------------------------------------------------------- rules

def detect_hand_raise(tr: Track, p: dict) -> tuple[list[dict], np.ndarray]:
    nose_y, sh_y = tr.nose[:, 1], tr.sh_mid[:, 1]
    # Top of the head: above the nose when the face is visible; from behind, ~1 S above shoulders.
    top = np.where(np.isfinite(nose_y), nose_y - p["hand_raise_above_nose"] * tr.H, sh_y - 1.0 * tr.S)
    raised = np.zeros(len(tr.t), bool)
    margin = np.full(len(tr.t), np.nan)
    with np.errstate(invalid="ignore"):
        for w, el in ((L_WRIST, L_ELBOW), (R_WRIST, R_ELBOW)):
            wy, ey = tr.m[:, w, 1], tr.m[:, el, 1]
            r = (wy < top) & (ey < sh_y + 0.1 * tr.S)        # elbow lifted too
            raised |= r
            margin = np.fmax(margin, np.where(r, (top - wy) / tr.H, np.nan))
    raised = majority(raised, tr.frames(p["smooth_s"]))
    rr = runs(raised, tr.frames(p["hand_raise_min_s"]), tr.frames(p["merge_gap_s"]))
    ev = _events_from_runs(tr, rr, "hand_raise", [L_WRIST, R_WRIST, L_ELBOW, R_ELBOW],
                           lambda s, e: np.nanmedian(margin[s:e + 1]) if np.isfinite(margin[s:e + 1]).any() else 0)
    return ev, raised


def detect_above_shoulder(tr: Track, p: dict, raised: np.ndarray, mic: np.ndarray) -> list[dict]:
    """Seen from behind, hidden hands get guessed near the shoulders, so the bar is higher
    there: wrist clearly above the shoulder AND elbow lifted. (Calibrated on TWSE audience:
    real hand-to-head / phone-photo cases had wrist >= 0.45 S above the shoulder, the false
    ones 0.27 S with the elbow hanging down.)"""
    sh_y = tr.sh_mid[:, 1]
    back = ~tr.face
    margin = np.where(back, p["above_shoulder_margin_back"], p["above_shoulder_margin"])
    up = np.zeros(len(tr.t), bool)
    with np.errstate(invalid="ignore"):
        for w, el in ((L_WRIST, L_ELBOW), (R_WRIST, R_ELBOW)):
            hand_up = tr.m[:, w, 1] < sh_y - margin * tr.S
            elbow_ok = ~back | (tr.m[:, el, 1] < sh_y + p["above_shoulder_elbow_back"] * tr.S)
            up |= hand_up & elbow_ok
    up = majority(up & ~raised & ~mic, tr.frames(p["smooth_s"]))
    rr = runs(up, tr.frames(p["above_shoulder_min_s"]), tr.frames(p["merge_gap_s"]))
    return _events_from_runs(tr, rr, "hand_above_shoulder", [L_WRIST, R_WRIST])


def detect_mic(tr: Track, p: dict) -> tuple[list[dict], np.ndarray]:
    mic = majority(mic_mask(tr.m), tr.frames(p["smooth_s"]))
    rr = runs(mic, tr.frames(p["mic_min_s"]), tr.frames(p["mic_merge_gap_s"]))
    return _events_from_runs(tr, rr, "holding_mic", [L_WRIST, R_WRIST, L_ELBOW, R_ELBOW, NOSE]), mic


def _head_height(tr: Track) -> np.ndarray:
    """Nose-to-shoulder height in units of the person's usual H (1 = usual, <1 = head lower)."""
    d = (tr.sh_mid[:, 1] - tr.nose[:, 1]) / tr.H
    d[~tr.face] = np.nan
    return d


def _fill(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).interpolate(limit=2, limit_direction="both").to_numpy()


def noise_level(x: np.ndarray) -> float:
    """Robust per-frame jitter of a signal: MAD of frame-to-frame differences / sqrt(2).
    Real movement is slow compared with frame rate, so differences are mostly jitter."""
    dx = np.diff(x[np.isfinite(x)])
    if len(dx) < 8:
        return 0.0
    return float(1.4826 * np.median(np.abs(dx - np.median(dx))) / np.sqrt(2))


def detect_nods(tr: Track, p: dict) -> list[dict]:
    d = _head_height(tr)
    if np.isfinite(d).sum() < tr.frames(1.0):
        return []
    k = _odd(tr.fps * 0.25)                                # light smoothing, keeps 0.3 s dips
    detr = smooth(_fill(d), k) - trend(_fill(d), _odd(p["trend_s"] * tr.fps))
    valid = np.isfinite(detr)
    sig = np.where(valid, -detr, 0.0)                      # dips become peaks
    # Threshold adapts to how jittery this person's keypoints are (small, distant people
    # jitter more): a dip must exceed both the fixed minimum and k x their own noise.
    thr = max(p["nod_min_amp"], p["noise_k"] * noise_level(_fill(d)))
    peaks, props = find_peaks(sig, prominence=thr,
                              width=(1, tr.frames(p["nod_max_dip_s"])), distance=2)
    peaks = [(int(pk), float(pr), int(lb), int(rb)) for pk, pr, lb, rb in
             zip(peaks, props["prominences"], props["left_bases"], props["right_bases"]) if valid[pk]]
    if not peaks:
        return []
    # group dips that follow each other quickly into one nod event
    groups, cur = [], [peaks[0]]
    gap = tr.frames(p["nod_group_gap_s"])
    for pk in peaks[1:]:
        if pk[0] - cur[-1][0] <= gap:
            cur.append(pk)
        else:
            groups.append(cur)
            cur = [pk]
    groups.append(cur)
    out = []
    half = tr.frames(p["nod_max_dip_s"]) // 2
    for grp in groups:
        s = max(grp[0][2], grp[0][0] - half)
        e = min(grp[-1][3], grp[-1][0] + half)
        amp = max(g[1] for g in grp)
        out.append({"gesture": "nod", "s": s, "e": e, "n_cycles": len(grp),
                    "confidence": _conf(tr, s, e, [NOSE, L_EYE, R_EYE, L_SHOULDER, R_SHOULDER],
                                        amp / (2 * thr))})
    return out


def detect_head_shakes(tr: Track, p: dict) -> list[dict]:
    hx = (tr.nose[:, 0] - tr.sh_mid[:, 0]) / tr.S
    hx[~tr.face] = np.nan
    if np.isfinite(hx).sum() < tr.frames(1.0):
        return []
    hxf = _fill(hx)
    detr = smooth(hxf, _odd(tr.fps * 0.25)) - trend(hxf, _odd(p["trend_s"] * tr.fps))
    valid = np.isfinite(detr)
    sig = np.where(valid, detr, 0.0)
    amp = max(p["shake_min_amp"], p["noise_k"] * noise_level(hxf))
    hi, phi = find_peaks(sig, prominence=amp)
    lo, plo = find_peaks(-sig, prominence=amp)
    ext = sorted([(int(i), 1, float(pr)) for i, pr in zip(hi, phi["prominences"]) if valid[i]] +
                 [(int(i), -1, float(pr)) for i, pr in zip(lo, plo["prominences"]) if valid[i]])
    step = tr.frames(p["shake_max_step_s"])
    out, seq = [], []

    def flush():
        if len(seq) - 1 >= p["shake_min_reversals"]:
            s, e = seq[0][0], seq[-1][0]
            strength = np.median([x[2] for x in seq]) / (2 * amp)
            out.append({"gesture": "head_shake", "s": s, "e": e,
                        "confidence": _conf(tr, s, e, [NOSE, L_SHOULDER, R_SHOULDER], strength)})

    for x in ext:
        if seq and x[1] != seq[-1][1] and x[0] - seq[-1][0] <= step:
            seq.append(x)
        else:
            flush()
            seq = [x]
    flush()
    return out


def _head_down_mask(tr: Track, p: dict) -> np.ndarray:
    d = smooth(_head_height(tr), _odd(p["smooth_s"] * tr.fps))
    with np.errstate(invalid="ignore"):
        down = d < p["head_down_ratio"]
    return majority(down, _odd(p["smooth_s"] * tr.fps))


def detect_head_down(tr: Track, p: dict, down: np.ndarray | None = None) -> list[dict]:
    if down is None:
        down = _head_down_mask(tr, p)
    rr = runs(down, tr.frames(p["head_down_min_s"]), tr.frames(p["merge_gap_s"]))
    d = smooth(_head_height(tr), _odd(p["smooth_s"] * tr.fps))
    return _events_from_runs(tr, rr, "head_down", [NOSE, L_SHOULDER, R_SHOULDER],
                             lambda s, e: (p["head_down_ratio"] - np.nanmedian(d[s:e + 1])) / 0.3)


def _look_away_mask(tr: Track, p: dict) -> np.ndarray:
    hx = (tr.nose[:, 0] - tr.sh_mid[:, 0]) / tr.S
    usual = np.nanmedian(hx[tr.face]) if tr.face.any() else np.nan
    off = np.abs(smooth(hx, _odd(p["smooth_s"] * tr.fps)) - usual)
    shoulders = np.isfinite(tr.sh_mid[:, 0])
    # "Face lost" only means looking away for someone whose face is normally visible.
    usually_facing = tr.face[tr.present].mean() >= 0.5 if tr.present.any() else False
    with np.errstate(invalid="ignore"):
        away = ((off > p["look_away_offset"]) & tr.face) | (shoulders & ~tr.face & tr.present & usually_facing)
    return majority(away, _odd(p["smooth_s"] * tr.fps))


def detect_look_away(tr: Track, p: dict, away: np.ndarray | None = None) -> list[dict]:
    if away is None:
        away = _look_away_mask(tr, p)
    rr = runs(away, tr.frames(p["look_away_min_s"]), tr.frames(p["merge_gap_s"]))
    return _events_from_runs(tr, rr, "look_away", [NOSE, L_EYE, R_EYE, L_SHOULDER, R_SHOULDER])


def detect_facing_audience(tr: Track, p: dict, away: np.ndarray | None = None,
                           down: np.ndarray | None = None) -> list[dict]:
    """Continuous engagement: person is present, face keypoints visible, not turned away
    (e.g. at slides) and not looking down (e.g. reading notes)."""
    usually_facing = tr.face[tr.present].mean() >= 0.5 if tr.present.any() else False
    if not usually_facing:
        return []
    if away is None:
        away = _look_away_mask(tr, p)
    if down is None:
        down = _head_down_mask(tr, p)
    facing = tr.present & tr.face & ~away & ~down
    facing = majority(facing, _odd(p["smooth_s"] * tr.fps))
    min_frames = tr.frames(p.get("facing_min_s", 1.0))
    rr = runs(facing, min_frames, tr.frames(p["merge_gap_s"]))
    return _events_from_runs(tr, rr, "facing_audience", [NOSE, L_EYE, R_EYE, L_SHOULDER, R_SHOULDER])


def detect_gesturing(tr: Track, p: dict, raised: np.ndarray) -> list[dict]:
    hip_y = np.nanmean(tr.m[:, [L_HIP, R_HIP], 1], 1)
    waist = np.where(np.isfinite(hip_y), hip_y, tr.sh_mid[:, 1] + 1.3 * tr.S)
    speed = np.zeros(len(tr.t))
    for w in (L_WRIST, R_WRIST):
        xy = np.c_[smooth(tr.m[:, w, 0], 3), smooth(tr.m[:, w, 1], 3)]
        v = np.r_[np.nan, np.linalg.norm(np.diff(xy, axis=0), axis=1)] / tr.S * tr.fps
        with np.errstate(invalid="ignore"):
            v[~(xy[:, 1] < waist)] = np.nan
        speed = np.fmax(speed, np.nan_to_num(v))
    speed = smooth(speed, _odd(p["smooth_s"] * tr.fps))
    moving = (speed > p["gesture_speed"]) & ~raised
    rr = runs(moving, tr.frames(p["gesture_min_s"]), tr.frames(p["merge_gap_s"]))
    return _events_from_runs(tr, rr, "gesturing", [L_WRIST, R_WRIST, L_ELBOW, R_ELBOW],
                             lambda s, e: np.median(speed[s:e + 1]) / (2 * p["gesture_speed"]))


def detect_stand_up(tr: Track, p: dict) -> list[dict]:
    """Standing up = shoulders rise by >= stand_rise x S above where they were while seated
    in the preceding stand_prior_seated_s, and stay up for >= stand_min_s. Uses only the
    shoulders, so it works from behind with legs hidden by chairs."""
    sh_y = smooth(tr.sh_mid[:, 1], _odd(p["smooth_s"] * tr.fps))
    prior = tr.frames(p["stand_prior_seated_s"])
    lag = tr.frames(0.5)
    base = pd.Series(sh_y).shift(lag).rolling(prior, min_periods=max(2, prior // 2)).median().to_numpy()
    with np.errstate(invalid="ignore"):
        rising = (base - sh_y) / tr.S >= p["stand_rise"]
    out = []
    for s, _ in runs(rising, tr.frames(p["stand_min_s"]), tr.frames(p["merge_gap_s"])):
        seated_level = base[s]
        # standing lasts until the shoulders come back down near the seated level
        down = np.flatnonzero((sh_y[s:] - seated_level) / tr.S > -0.4)
        e = s + int(down[0]) - 1 if len(down) else len(sh_y) - 1
        e = max(e, s)
        out.append({"gesture": "stand_up", "s": s, "e": e,
                    "confidence": _conf(tr, s, e, [L_SHOULDER, R_SHOULDER], 1.0)})
    return out


# ----------------------------------------------------------------------------- driver

def detect_person(tr: Track, gestures: set[str], p: dict) -> list[dict]:
    # Missing keypoints are NaN by design; numpy's "empty slice" warnings are expected noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _detect_person(tr, gestures, p)


def _detect_person(tr: Track, gestures: set[str], p: dict) -> list[dict]:
    ev: list[dict] = []
    raise_ev, raised = detect_hand_raise(tr, p)
    mic_ev, mic = detect_mic(tr, p)
    if "hand_raise" in gestures:
        ev += raise_ev
    if "holding_mic" in gestures:
        ev += mic_ev
    if "hand_above_shoulder" in gestures:
        ev += detect_above_shoulder(tr, p, raised, mic)
    if "nod" in gestures:
        ev += detect_nods(tr, p)
    if "head_shake" in gestures:
        ev += detect_head_shakes(tr, p)
    down = _head_down_mask(tr, p)
    if "head_down" in gestures:
        ev += detect_head_down(tr, p, down)
    away = _look_away_mask(tr, p)
    if "look_away" in gestures:
        ev += detect_look_away(tr, p, away)
    if "facing_audience" in gestures:
        ev += detect_facing_audience(tr, p, away, down)
    if "gesturing" in gestures:
        ev += detect_gesturing(tr, p, raised)
    if "stand_up" in gestures:
        ev += detect_stand_up(tr, p)
    return ev


def detect_phases(rows: pd.DataFrame, people: pd.DataFrame, frames: pd.DataFrame,
                  min_phase_s: float, fps: float) -> pd.DataFrame:
    """presentation = a presenter is visible; qa = panel visible without presenter."""
    cols = ["phase", "start_s", "end_s"]
    if frames.empty:
        return pd.DataFrame(columns=cols)
    t = frames.sort_values("frame").set_index("frame").t
    key_col = "role_key" if "role_key" in people.columns else "role"
    keys = dict(zip(people.person_id, people[key_col])) if not people.empty else {}
    r = rows.assign(role_key=rows.pid.map(keys)) if not rows.empty else rows
    pres = set(r[r.role_key == "presenter"].frame) if len(r) else set()
    panel = set(r[r.role_key == "panel"].frame) if len(r) else set()
    if not pres and not panel:
        return pd.DataFrame([{"phase": "all", "start_s": float(t.iloc[0]), "end_s": float(t.iloc[-1])}])
    lab = np.array(["presentation" if f in pres else "qa" if f in panel else "other" for f in t.index], dtype=object)
    # smooth: absorb short segments into the previous one
    min_len = max(1, int(min_phase_s * fps))
    segs = []
    for v in lab:
        if segs and segs[-1][0] == v:
            segs[-1][1] += 1
        else:
            segs.append([v, 1])
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, (v, n) in enumerate(segs):
            if n < min_len:
                j = i - 1 if i > 0 else i + 1
                segs[j][1] += n
                segs.pop(i)
                changed = True
                break
        merged = []
        for v, n in segs:
            if merged and merged[-1][0] == v:
                merged[-1][1] += n
            else:
                merged.append([v, n])
        segs = merged
    out, i = [], 0
    tv = t.to_numpy()
    for v, n in segs:
        out.append({"phase": v, "start_s": round(float(tv[i]), 2), "end_s": round(float(tv[i + n - 1]), 2)})
        i += n
    return pd.DataFrame(out, columns=cols)


def phase_at(phases: pd.DataFrame, t: float) -> str:
    if phases.empty:
        return "all"
    hit = phases[(phases.start_s <= t + 1e-6) & (phases.end_s >= t - 1e-6)]
    return str(hit.iloc[0].phase) if len(hit) else str(phases.iloc[-1].phase)


def detect_hand_events(hand_df: pd.DataFrame, person_id: int, wanted_gestures: set[str],
                       frame_t: pd.Series, p: dict, fps: float) -> list[dict]:
    if hand_df.empty or not wanted_gestures:
        return []
    p_hands = hand_df[hand_df.person_id == person_id]
    if p_hands.empty or "extra" not in p_hands.columns:
        return []

    n_frames = len(frame_t)
    frame_to_idx = {int(fr): i for i, fr in enumerate(frame_t.index)}
    frame_gestures: dict[int, list[tuple[str, float]]] = {}
    for _, row in p_hands.iterrows():
        extra_str = row.get("extra")
        if not isinstance(extra_str, str) or not extra_str:
            continue
        try:
            extra = json.loads(extra_str)
        except Exception:  # noqa: BLE001
            continue
        g = extra.get("gesture")
        if g in wanted_gestures:
            fr = int(row.get("frame", -1))
            score = float(row.get("score") or 0.8)
            frame_gestures.setdefault(fr, []).append((g, score))

    out = []
    for g_name in wanted_gestures:
        mask = np.zeros(n_frames, dtype=bool)
        confs = np.zeros(n_frames, dtype=float)
        for fr, matches in frame_gestures.items():
            if fr in frame_to_idx:
                idx = frame_to_idx[fr]
                scores = [s for (name, s) in matches if name == g_name]
                if scores:
                    mask[idx] = True
                    confs[idx] = max(scores)

        min_len = max(1, int(round(p.get(f"{g_name}_min_s", 0.5) * fps)))
        merge_gap = max(0, int(round(p.get("merge_gap_s", 0.5) * fps)))
        rr = runs(mask, min_len, merge_gap)
        for s, e in rr:
            mean_conf = float(np.mean(confs[s:e + 1])) if np.any(confs[s:e + 1]) else 0.8
            out.append({
                "gesture": g_name,
                "s": s,
                "e": e,
                "confidence": round(mean_conf, 3),
            })
    return out


def detect_events(rows: pd.DataFrame, people: pd.DataFrame, frames: pd.DataFrame, cfg: EventsConfig,
                  fps: float, video_id: str,
                  hand_landmarks: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """rows: one row per person per frame with a `pid` column (people.assign_people)."""
    p = cfg.resolved_params()
    phases = detect_phases(rows, people, frames, cfg.min_phase_s, fps)
    if rows.empty or people.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS), phases
    frame_t = frames.set_index("frame").t
    by_gesture_roles = {g: set(r) for g, r in cfg.gestures.items()}
    out = []
    for _, person in people.iterrows():
        key = person.get("role_key", person.role)
        if key in cfg.exclude_roles:
            continue
        wanted = {g for g, roles in by_gesture_roles.items() if key in roles}
        if not wanted:
            continue
        g = rows[rows.pid == person.person_id]
        if len(g) < 2:
            continue
        tr = build_track(g, frame_t, fps, p["kp_conf"])
        for ev in detect_person(tr, wanted, p):
            s, e = ev["s"], ev["e"]
            t0, t1 = float(tr.t[s]), float(tr.t[e]) + 1.0 / fps
            out.append({"video_id": video_id, "person_id": int(person.person_id), "role": person.role,
                        "phase": phase_at(phases, t0), "gesture": ev["gesture"],
                        "start_s": round(t0, 2), "end_s": round(t1, 2), "duration_s": round(t1 - t0, 2),
                        "confidence": ev["confidence"]})

        # Process fine hand gestures if hand landmarks are available
        if hand_landmarks is not None and not hand_landmarks.empty:
            hand_wanted = wanted & {"pointing", "open_palm"}
            if hand_wanted:
                h_events = detect_hand_events(hand_landmarks, int(person.person_id), hand_wanted, frame_t, p, fps)
                for h_ev in h_events:
                    s, e = h_ev["s"], h_ev["e"]
                    t0 = float(frame_t.iloc[s])
                    t1 = float(frame_t.iloc[e]) + 1.0 / fps
                    out.append({"video_id": video_id, "person_id": int(person.person_id), "role": person.role,
                                "phase": phase_at(phases, t0), "gesture": h_ev["gesture"],
                                "start_s": round(t0, 2), "end_s": round(t1, 2), "duration_s": round(t1 - t0, 2),
                                "confidence": h_ev["confidence"]})

    events = pd.DataFrame(out, columns=EVENT_COLUMNS)
    if len(events):
        events = events.sort_values(["start_s", "person_id"]).reset_index(drop=True)
        log.info("%d events: %s", len(events), events.gesture.value_counts().to_dict())
    return events, phases
