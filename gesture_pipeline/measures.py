"""Facing / turning-away and hand-gesture measures, in a form ready for analysis.

Two tables per video, built from events.csv:

facing_and_hands.csv - one row per presenter / panelist:
    visible_min                 minutes the person was on screen
    facing_s, facing_share      time facing the audience (camera), share of visible time
    turned_away_n/_s/_share     how often / how long the head was turned away (look_away)
    head_down_n/_s/_share       how often / how long the head was down (notes, phone)
    pointing_n, pointing_per_min, open_palm_n, open_palm_per_min   hand shapes (experimental,
                                only with --detect-hands; empty otherwise)

timeline.csv - one row per presenter / panelist per second:
    state   facing | turned_away | head_down | not_visible | unclear
    pointing, open_palm   1 if that hand shape was seen during the second
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FOCUS_ROLES = {"presenter", "panel"}
STATE_GESTURES = {"facing_audience", "look_away", "head_down", "holding_mic", "gesturing", "speaking"}

SUMMARY_COLUMNS = ["video_id", "person_id", "role", "visible_min",
                   "facing_s", "facing_share",
                   "turned_away_n", "turned_away_s", "turned_away_share",
                   "head_down_n", "head_down_s", "head_down_share",
                   "pointing_n", "pointing_per_min", "open_palm_n", "open_palm_per_min",
                   "hands_tracked"]
TIMELINE_COLUMNS = ["video_id", "person_id", "role", "second", "clock", "state", "pointing", "open_palm"]


def _focus(people: pd.DataFrame) -> pd.DataFrame:
    key = "role_key" if "role_key" in people.columns else "role"
    return people[people[key].isin(FOCUS_ROLES)] if len(people) else people


def facing_and_hands(events: pd.DataFrame, people: pd.DataFrame, visibility: pd.DataFrame,
                     hands_tracked: bool) -> pd.DataFrame:
    rows = []
    vis = visibility.groupby("person_id").visible_s.sum() if len(visibility) else pd.Series(dtype=float)
    for _, p in _focus(people).iterrows():
        ev = events[events.person_id == p.person_id] if len(events) else events
        visible = float(vis.get(p.person_id, p.get("visible_s", 0.0)))

        def tot(g):
            e = ev[ev.gesture == g] if len(ev) else ev
            return len(e), float(e.duration_s.sum()) if len(e) else 0.0

        f_n, f_s = tot("facing_audience")
        a_n, a_s = tot("look_away")
        d_n, d_s = tot("head_down")
        pt_n, _ = tot("pointing")
        op_n, _ = tot("open_palm")
        share = (lambda x: round(min(1.0, x / visible), 3) if visible else None)
        per_min = (lambda n: round(n / (visible / 60), 2) if visible else None)
        rows.append({
            "video_id": p.video_id, "person_id": int(p.person_id), "role": p.role,
            "visible_min": round(visible / 60, 2),
            "facing_s": round(f_s, 1), "facing_share": share(f_s),
            "turned_away_n": a_n, "turned_away_s": round(a_s, 1), "turned_away_share": share(a_s),
            "head_down_n": d_n, "head_down_s": round(d_s, 1), "head_down_share": share(d_s),
            "pointing_n": pt_n if hands_tracked else None,
            "pointing_per_min": per_min(pt_n) if hands_tracked else None,
            "open_palm_n": op_n if hands_tracked else None,
            "open_palm_per_min": per_min(op_n) if hands_tracked else None,
            "hands_tracked": hands_tracked,
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _cover(intervals: pd.DataFrame, n_sec: int, t0: int) -> np.ndarray:
    """Seconds (float, 0-1) of each 1-s bin [t0+i, t0+i+1) covered by the intervals."""
    cov = np.zeros(n_sec)
    for s, e in zip(intervals.start_s, intervals.end_s):
        a, b = max(s, t0), min(e, t0 + n_sec)
        i = int(np.floor(a - t0))
        while a < b and i < n_sec:
            nxt = min(b, t0 + i + 1)
            cov[i] += nxt - a
            a, i = nxt, i + 1
    return np.minimum(cov, 1.0)


def timeline(events: pd.DataFrame, people: pd.DataFrame, rows: pd.DataFrame, video_id: str) -> pd.DataFrame:
    out = []
    for _, p in _focus(people).iterrows():
        mine = rows[rows.pid == p.person_id] if len(rows) else rows
        if mine.empty:
            continue
        t0, t1 = int(np.floor(mine.t.min())), int(np.ceil(mine.t.max()))
        n = max(1, t1 - t0)
        seen = np.zeros(n, bool)
        seen[np.clip(np.floor(mine.t.to_numpy() - t0).astype(int), 0, n - 1)] = True
        ev = events[events.person_id == p.person_id] if len(events) else events

        def cov(g):
            e = ev[ev.gesture == g] if len(ev) else ev
            return _cover(e, n, t0) if len(e) else np.zeros(n)

        away, down, facing = cov("look_away"), cov("head_down"), cov("facing_audience")
        point, palm = cov("pointing"), cov("open_palm")
        for i in range(n):
            if away[i] >= 0.5:
                st = "turned_away"
            elif down[i] >= 0.5:
                st = "head_down"
            elif facing[i] >= 0.5:
                st = "facing"
            elif not seen[i]:
                st = "not_visible"
            else:
                st = "unclear"
            sec = t0 + i
            out.append({"video_id": video_id, "person_id": int(p.person_id), "role": p.role,
                        "second": sec, "clock": f"{sec // 60:02d}:{sec % 60:02d}", "state": st,
                        "pointing": int(point[i] > 0), "open_palm": int(palm[i] > 0)})
    return pd.DataFrame(out, columns=TIMELINE_COLUMNS)
