"""Stage: turn raw tracker ids into consistent people, and give each person a role.

1. Tracklet merging. A tracker loses people during occlusions and gives them a
   new id when they reappear. With a static camera most people stay put, so a
   tracklet that starts close to where an earlier one ended (relative to the
   person's own size), after it ended, is the same person.

2. Role assignment, from what the person looks like over the whole video:
     transient  - visible only briefly (walking past)
     on_screen  - a small figure inside the projection screen (a photo on a slide)
     audience   - mostly seen from behind (shoulders visible, face not)
     other      - standing at the far left/right of the frame (staff), or unclear
     presenter  - standing, or facing the camera on their own
     panel      - facing the camera, seated in a row with others
   Manual zones in the config override these rules for a known venue.

3. Duplicate boxes. The detector occasionally draws two boxes on one person;
   ids whose heads coincide most of the time they are both visible are merged.

A "person" is one continuous presence in one place. If someone walks from the
podium to the panel table they appear as two people (presenter, then panel);
linking them would need face recognition, which this footage is too small for.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PeopleConfig
from .keypoints import L_EYE, L_SHOULDER, NOSE, R_EYE, R_SHOULDER, masked, mic_mask, stack, standing_mask
from .scene import fraction_inside

log = logging.getLogger(__name__)


@dataclass
class Tracklet:
    track_id: int
    t_start: float
    t_end: float
    n: int
    start_xy: np.ndarray  # box centre near the start
    end_xy: np.ndarray    # box centre near the end
    height: float         # median box height (0-1)


def _tracklets(pose: pd.DataFrame, min_det: int) -> list[Tracklet]:
    out = []
    for tid, g in pose[pose.person_id >= 0].groupby("person_id"):
        if len(g) < min_det:
            continue
        g = g.sort_values("t")
        b = np.stack(g["bbox"].to_numpy())
        c = np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]
        k = min(5, len(g))
        out.append(Tracklet(int(tid), float(g.t.iloc[0]), float(g.t.iloc[-1]), len(g),
                            np.median(c[:k], 0), np.median(c[-k:], 0), float(np.median(b[:, 3] - b[:, 1]))))
    return sorted(out, key=lambda t: t.t_start)


def merge_tracklets(pose: pd.DataFrame, cfg: PeopleConfig) -> dict[int, int]:
    """Map raw tracker id -> person id (1, 2, 3 ... in order of first appearance)."""
    people: list[list[Tracklet]] = []
    for tr in _tracklets(pose, cfg.min_detections):
        best, best_d = None, np.inf
        for idx, chain in enumerate(people):
            last = chain[-1]
            gap = tr.t_start - last.t_end
            if gap < -0.5 or gap > cfg.merge_max_gap_s:
                continue  # overlapping in time -> different people; or gone too long
            size = max(tr.height, last.height)
            if not 0.6 <= tr.height / max(last.height, 1e-6) <= 1.6:
                continue
            d = float(np.linalg.norm(tr.start_xy - last.end_xy)) / max(size, 1e-6)
            if d <= cfg.merge_max_dist and d < best_d:
                best, best_d = idx, d
        if best is None:
            people.append([tr])
        else:
            people[best].append(tr)
    mapping = {}
    for pid, chain in enumerate(people, 1):
        for tr in chain:
            mapping[tr.track_id] = pid
    return mapping


def _person_features(g: pd.DataFrame, face_thr: float, sample_dt: float) -> dict:
    xy, cf, bbox = stack(g)
    face = (cf[:, NOSE] >= face_thr) & (cf[:, L_EYE] >= face_thr) & (cf[:, R_EYE] >= face_thr)
    shoulders = (cf[:, L_SHOULDER] >= face_thr) & (cf[:, R_SHOULDER] >= face_thr)
    back = shoulders & (cf[:, NOSE] < 0.3) & (cf[:, L_EYE] < 0.3) & (cf[:, R_EYE] < 0.3)

    m = masked(xy, cf, face_thr)
    standing = standing_mask(m, bbox)
    mic = mic_mask(m)

    cx = (bbox[:, 0] + bbox[:, 2]) / 2
    cy = (bbox[:, 1] + bbox[:, 3]) / 2
    shw = np.abs(m[:, L_SHOULDER, 0] - m[:, R_SHOULDER, 0])
    head_y = np.where(np.isfinite(m[:, NOSE, 1]), m[:, NOSE, 1], bbox[:, 1] + 0.08 * (bbox[:, 3] - bbox[:, 1]))
    return {
        "first_seen_s": round(float(g.t.min()), 2),
        "last_seen_s": round(float(g.t.max()), 2),
        "visible_s": round(len(g) * sample_dt, 1),
        "face_ratio": round(float(face.mean()), 3),
        "back_ratio": round(float(back.mean()), 3),
        "standing_ratio": round(float(np.mean(standing)), 3),
        "mic_ratio": round(float(mic.mean()), 3),
        "edge": bool(np.median(bbox[:, 0]) < 0.015 or np.median(bbox[:, 2]) > 0.985),
        "bottom_edge": bool(np.median(bbox[:, 3]) > 0.97),
        "center_x": round(float(np.median(cx)), 3),
        "center_y": round(float(np.median(cy)), 3),
        "height": round(float(np.median(bbox[:, 3] - bbox[:, 1])), 3),
        "shoulder_w": round(float(np.nanmedian(shw)), 4) if np.isfinite(shw).any() else None,
        "head_y": round(float(np.nanmedian(head_y)), 4),
        "box": [round(float(v), 4) for v in np.median(bbox, axis=0)],
    }


def _in_row(f: dict, others: list[dict], aspect: float) -> bool:
    """Seated panel members sit side by side: heads at the same height (within ~1
    shoulder width) and similar size. A presenter stands apart."""
    S = f["shoulder_w"]
    if not S:
        return False
    for o in others:
        if o is f or not o["shoulder_w"] or o["face_ratio"] < 0.3:
            continue
        if not 0.6 <= o["shoulder_w"] / S <= 1.6:
            continue
        if abs(o["head_y"] - f["head_y"]) * aspect <= 1.0 * (S + o["shoulder_w"]) / 2:
            return True
    return False


def assign_roles(feats: list[dict], cfg: PeopleConfig, aspect: float = 9 / 16,
                 screen: list[float] | None = None) -> list[tuple[str, str]]:
    """Decide every person's role, looking at the whole scene (rules in docs/roles.md)."""
    # Size reference: the biggest person facing the camera (a real presenter or panelist).
    # Pictures on a slide are much smaller than the people standing in front of it.
    facing = [f["shoulder_w"] for f in feats if f["visible_s"] >= cfg.min_presence_s
              and f["shoulder_w"] and f["face_ratio"] >= 0.3]
    ref_S = max(facing) if facing else None

    def on_screen(f: dict) -> bool:
        if not screen or fraction_inside(f["box"], screen) <= 0.9:
            return False
        small = bool(ref_S and f["shoulder_w"] and f["shoulder_w"] < 0.6 * ref_S)
        floating = f["box"][3] < screen[3] - 0.05     # feet/body don't reach below the screen
        return small or floating

    # pass 1: things decided by the person alone
    first: list[tuple[str, str] | None] = []
    for f in feats:
        if f["visible_s"] < cfg.min_presence_s:
            first.append(("transient", f"visible only {f['visible_s']}s"))
        elif on_screen(f):
            first.append(("on_screen", "small figure entirely inside the projection screen (a picture on a slide)"))
        elif f["back_ratio"] >= 0.4 and f["face_ratio"] < 0.3:
            first.append(("audience", f"seen from behind {f['back_ratio']:.0%} of the time"))
        elif f["bottom_edge"] and f["face_ratio"] < 0.3:
            first.append(("audience", "cut off by the bottom of the frame, face not visible"))
        elif (f["edge"] or not cfg.edge_margin <= f["center_x"] <= 1 - cfg.edge_margin) and f["standing_ratio"] >= 0.5:
            first.append(("other", "standing at the side of the frame (staff / moderator)"))
        else:
            first.append(None)

    # pass 2: people facing (or side-on to) the camera, judged against each other
    candidates = [f for f, r in zip(feats, first) if r is None]
    out = []
    for f, r in zip(feats, first):
        if r is not None:
            out.append(r)
        elif f["standing_ratio"] >= 0.8:
            out.append(("presenter", f"standing {f['standing_ratio']:.0%} of the time, not at the side"))
        elif f["face_ratio"] >= 0.3 and _in_row(f, candidates, aspect):
            out.append(("panel", "seated in a row with others facing the camera"))
        elif f["face_ratio"] >= 0.3 or f["standing_ratio"] >= 0.5:
            out.append(("presenter", "facing the camera on their own (not part of a seated row)"))
        else:
            out.append(("other", "no clear cue (face and back both rarely visible)"))
    return out


def _zone_role(f: dict, zones: list[dict]) -> str | None:
    for z in zones:
        x1, y1, x2, y2 = z["box"]
        if x1 <= f["center_x"] <= x2 and y1 <= f["center_y"] <= y2:
            return z["role"]
    return None


def _head_points(pose: pd.DataFrame) -> np.ndarray:
    xy, cf, bbox = stack(pose)
    head = xy[:, NOSE].copy()
    weak = cf[:, NOSE] < 0.3
    head[weak, 0] = (bbox[weak, 0] + bbox[weak, 2]) / 2   # fall back to top-centre of the box
    head[weak, 1] = bbox[weak, 1] + 0.08 * (bbox[weak, 3] - bbox[weak, 1])
    return head


def find_duplicates(pose: pd.DataFrame, min_shared: int = 8) -> dict[int, int]:
    """The detector sometimes gives one person two boxes (e.g. upper body + full body).
    Two ids whose heads coincide in most shared frames are the same person.
    Returns {duplicate pid: kept pid}."""
    if pose.empty:
        return {}
    df = pose[["frame", "pid", "score"]].copy()
    df["hx"], df["hy"] = _head_points(pose).T
    b = np.stack(pose["bbox"].to_numpy())
    df["h"] = b[:, 3] - b[:, 1]
    counts = df.pid.value_counts()
    score = df.groupby("pid").score.mean()
    merged = df.merge(df, on="frame", suffixes=("_a", "_b"))
    merged = merged[merged.pid_a < merged.pid_b]
    if merged.empty:
        return {}
    d = np.hypot(merged.hx_a - merged.hx_b, merged.hy_a - merged.hy_b)
    merged["close"] = d < 0.12 * np.minimum(merged.h_a, merged.h_b)
    dup = {}
    for (a, b_), g in merged.groupby(["pid_a", "pid_b"]):
        if len(g) >= min_shared and g.close.mean() >= 0.6:
            keep, drop = (a, b_) if (counts[a], score[a]) >= (counts[b_], score[b_]) else (b_, a)
            dup[int(drop)] = int(keep)
    # resolve chains a->b->c
    for k in list(dup):
        while dup[k] in dup:
            dup[k] = dup[dup[k]]
    return dup


def assign_people(pose: pd.DataFrame, mapping: dict[int, int]) -> pd.DataFrame:
    """Pose rows with a `pid` column: one row per person per frame (best detection kept)."""
    if pose.empty or not mapping:
        return pose.assign(pid=pd.Series(dtype=int)).iloc[0:0]
    out = pose[pose.person_id.isin(mapping)].copy()
    out["pid"] = out.person_id.map(mapping).astype(int)
    return (out.sort_values("score", ascending=False).drop_duplicates(["pid", "frame"])
               .sort_values(["frame", "pid"]).reset_index(drop=True))


def build_people(pose: pd.DataFrame, cfg: PeopleConfig, video_id: str, sample_fps: float,
                 aspect: float = 9 / 16, screen: list[float] | None = None) -> tuple[pd.DataFrame, dict[int, int]]:
    cols = ["video_id", "person_id", "role", "role_key", "role_reason", "track_ids"]
    if pose.empty:
        return pd.DataFrame(columns=cols), {}
    mapping = merge_tracklets(pose, cfg)
    tmp = pose[pose.person_id.isin(mapping)].copy()
    tmp["pid"] = tmp.person_id.map(mapping)
    dup = find_duplicates(tmp.reset_index(drop=True))
    if dup:
        log.info("merged duplicate boxes: %s", dup)
        mapping = {t: dup.get(p, p) for t, p in mapping.items()}
    # renumber 1..N in order of first appearance
    first = assign_people(pose, mapping).groupby("pid").t.min().sort_values()
    renum = {old: new for new, old in enumerate(first.index, 1)}
    mapping = {t: renum[p] for t, p in mapping.items() if p in renum}
    rows_df = assign_people(pose, mapping)

    pids, feats = [], []
    for pid, g in rows_df.groupby("pid"):
        pids.append(int(pid))
        feats.append(_person_features(g, cfg.face_conf, 1.0 / sample_fps))
    roles = assign_roles(feats, cfg, aspect, screen)
    rows = []
    for pid, f, (role, why) in zip(pids, feats, roles):
        zr = _zone_role(f, cfg.zones) if role not in ("transient", "on_screen") else None
        if zr:
            role, why = zr, "inside configured zone"
        tracks = sorted(t for t, p in mapping.items() if p == pid)
        f = {k: v for k, v in f.items() if k != "box"}
        rows.append({"video_id": video_id, "person_id": pid,
                     "role": cfg.labels.get(role, role), "role_key": role, "role_reason": why,
                     "track_ids": json.dumps(tracks), **f})
    people = pd.DataFrame(rows).sort_values("person_id").reset_index(drop=True)
    log.info("%d tracker ids -> %d people (%s)", len(mapping), len(people),
             people.role.value_counts().to_dict())
    return people, mapping
