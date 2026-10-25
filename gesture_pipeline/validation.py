"""Accuracy check: compare the program's events with gestures marked by a person.

Two commands:

  gestures validate-prepare   picks short clips from processed videos and creates, in
                              validation/: one "coding video" per clip (every person boxed
                              with their id, a running clock), a coding sheet (Excel) to fill
                              in, and instructions.
  gestures validate           reads the filled-in sheets and reports, per gesture, how many
                              of the person's marks the program found (recall) and how many of
                              the program's events the person agrees with (precision). With two
                              or more coders it also reports how much the coders agree.

Matching rule: a program event matches a marked one when it is the same person and gesture
and the two time ranges overlap, allowing `tolerance_s` (default 1 s) slack on each side.
Each event can match at most once.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .tables import read_table

from .config import DEFAULT_ROLE_GESTURES
from .debug_video import _Writer
from .frames import FrameSampler
from .measures import STATE_GESTURES
from .people import assign_people
from .store import read_landmarks

log = logging.getLogger(__name__)

CODABLE = list(DEFAULT_ROLE_GESTURES) + ["speaking"]
CLIP_COLUMNS = ["clip_id", "video_id", "video_path", "phase", "start_s", "end_s", "people"]


# ----------------------------------------------------------------------------- prepare

def usable_videos(output_dir: Path) -> tuple[list[Path], list[str]]:
    """Videos fully processed by the current version (roles, gestures and talk time done)."""
    ready, stale = [], []
    if not Path(output_dir).exists():
        return ready, stale
    for d in sorted(p for p in Path(output_dir).iterdir() if (p / "status.json").exists()):
        stages = json.loads((d / "status.json").read_text()).get("stages", {})
        done = {k for k, v in stages.items() if v.get("state") == "done"}
        cols = read_table(d / "people.csv", nrows=0).columns if (d / "people.csv").exists() else []
        if {"people", "events", "talk"} <= done and "role_key" in cols:
            ready.append(d)
        else:
            stale.append(d.name)
    return ready, stale


class NotReadyError(RuntimeError):
    pass


def choose_clips(output_dir: Path, n_clips: int = 10, length_s: float = 120.0, seed: int = 1) -> pd.DataFrame:
    """Spread clips over videos and over presentation / Q&A, chosen at random positions
    (not by looking at results, so the check is not biased towards easy moments)."""
    rng = np.random.default_rng(seed)
    per_video = []
    ready, stale = usable_videos(output_dir)
    for d in ready:
        meta = json.loads((d / "metadata.json").read_text()) if (d / "metadata.json").exists() else {}
        phases = read_table(d / "phases.csv") if (d / "phases.csv").exists() else pd.DataFrame()
        if phases.empty:
            phases = pd.DataFrame([{"phase": "all", "start_s": 0.0, "end_s": meta.get("duration_s", 0.0)}])
        options = []
        for _, ph in phases.iterrows():
            room = ph.end_s - ph.start_s - length_s
            if room < 0:
                continue
            start = float(ph.start_s + rng.uniform(0, room))
            options.append({"video_id": d.name, "video_path": meta.get("path", ""), "phase": ph.phase,
                            "start_s": round(start, 1), "end_s": round(start + length_s, 1)})
        dur = float(meta.get("duration_s") or phases.end_s.max())
        if not options and dur >= length_s:              # no single part long enough: span parts
            start = float(rng.uniform(0, dur - length_s))
            options.append({"video_id": d.name, "video_path": meta.get("path", ""), "phase": "mixed",
                            "start_s": round(start, 1), "end_s": round(start + length_s, 1)})
        rng.shuffle(options)
        if options:
            per_video.append(options)
    clips = []
    while len(clips) < n_clips and any(per_video):
        for opts in per_video:                      # round-robin over videos
            if opts and len(clips) < n_clips:
                clips.append(opts.pop(0))
    df = pd.DataFrame(clips)
    if df.empty:
        return pd.DataFrame(columns=CLIP_COLUMNS)
    df.insert(0, "clip_id", [f"C{i + 1:02d}" for i in range(len(df))])
    df["people"] = ""
    return df


def render_coding_video(video: Path, out: Path, rows: pd.DataFrame, start: float, end: float,
                        max_width: int = 1280) -> None:
    """Full-frame-rate clip with each person's box and id, and the video clock. No gesture
    labels: the coder must not see what the program found."""
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    times = np.sort(rows.t.unique()) if len(rows) else np.array([])
    by_t = {t: g for t, g in rows.groupby("t")} if len(rows) else {}
    writer = None
    try:
        for fr in FrameSampler(video, src_fps, max_width, start, end):
            img = fr.image
            h, w = img.shape[:2]
            if writer is None:
                writer = _Writer(out, src_fps, (w, h))
            if len(times):
                i = int(np.clip(np.searchsorted(times, fr.t), 0, len(times) - 1))
                j = i - 1 if i > 0 and abs(times[i - 1] - fr.t) < abs(times[i] - fr.t) else i
                if abs(times[j] - fr.t) <= 0.3:
                    for _, r in by_t[times[j]].iterrows():
                        x1, y1, x2, y2 = (np.array(r.bbox) * [w, h, w, h]).astype(int)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 255), 1)
                        lab = f"P{int(r.pid)}"
                        cv2.putText(img, lab, (x1 + 2, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                        cv2.putText(img, lab, (x1 + 2, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
            mm, ss = divmod(fr.t, 60)
            clock = f"{int(mm):02d}:{ss:04.1f}"
            cv2.rectangle(img, (w - 150, h - 44), (w - 6, h - 8), (0, 0, 0), -1)
            cv2.putText(img, clock, (w - 142, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(img)
    finally:
        if writer is not None:
            writer.close()


INSTRUCTIONS = """# Gesture coding: instructions

Thank you for helping check the program's accuracy. This takes about 15-20 minutes per clip.

## What you have

* `clips/C01.mp4`, `clips/C02.mp4`, ... short clips (about 2 minutes). Every person has a
  yellow box with an id (P1, P2, ...). A clock at the bottom right shows the time.
* `coding_sheet.xlsx` - where you write what you see. **Make a copy named with your name first**,
  e.g. `coding_sheet_anna.xlsx`, so several people can code the same clips.

## What to do

Watch each clip (pause and rewind as much as you like). Every time someone in a box does one
of the gestures below, add one row to the sheet:

| column | what to write |
|---|---|
| clip_id | the clip, e.g. C01 |
| person_id | the number in the box, e.g. 3 for P3 |
| gesture | one of the names below (choose from the drop-down list) |
| start | clock time when it starts, e.g. 25:12 (minutes:seconds) or 25:12.5 |
| end | clock time when it ends |
| notes | optional, anything unclear |

Mark only what you can clearly see. If you are unsure, write it anyway and put "unsure" in notes.
Do not try to guess what the program found.

## Gestures

| gesture | mark it when |
|---|---|
| hand_raise | a hand goes above the top of the head (e.g. to ask a question) for at least about 1 second |
| hand_above_shoulder | a hand is above shoulder height but not above the head (touching head, holding up a phone, pointing up) |
| holding_mic | someone holds a microphone up to their mouth |
| nod | the head goes down and back up quickly, once or several times in a row (count a series as one) |
| head_shake | the head turns left-right-left (or right-left-right) |
| head_down | the head is clearly lowered (reading, writing, looking at a phone) for 3 seconds or more |
| look_away | the head is clearly turned away from where the person usually looks, for 2 seconds or more |
| facing_audience | the person is facing towards the audience/camera (not looking away at slides or down at notes) for at least 1 second |
| gesturing | hands move while talking or explaining (one row per continuous stretch) |
| stand_up | a seated person stands up |
| speaking | the person is talking (one row per stretch of talking; short pauses can stay in one row) |

You do not have to code every gesture type. If you only code some (for example only hand_raise,
nod and speaking), that is fine. Write the types you coded in the "coded_gestures" sheet so that
gestures you did not look for are not counted as misses.

When done, save the file in this folder and run `gestures validate`.
"""


def prepare(output_dir: Path, val_dir: Path, n_clips: int = 10, length_s: float = 120.0, seed: int = 1) -> pd.DataFrame:
    from openpyxl import Workbook
    from openpyxl.worksheet.datavalidation import DataValidation

    val_dir = Path(val_dir)
    (val_dir / "clips").mkdir(parents=True, exist_ok=True)
    ready, stale = usable_videos(output_dir)
    if stale:
        log.warning("Skipping %d video(s) not fully processed by this version: %s. "
                    "Run `gestures run` on them first.", len(stale), ", ".join(stale))
    if not ready:
        raise NotReadyError(
            f"No fully processed videos in '{output_dir}'. Process your videos first, e.g.\n"
            f"    gestures run data/raw/\n"
            f"then run `gestures validate-prepare` again."
            + (f"\n(Found results from an older version for: {', '.join(stale)}; `gestures run` updates them.)" if stale else ""))
    clips = choose_clips(output_dir, n_clips, length_s, seed)
    if clips.empty:
        raise NotReadyError(f"The processed videos are all shorter than {length_s:.0f} s. Try --length 60.")
    for i, c in clips.iterrows():
        vdir = Path(output_dir) / c.video_id
        pose = read_landmarks(vdir / "landmarks", "pose")
        mapping = {int(k): v for k, v in json.loads((vdir / "person_map.json").read_text()).items()}
        rows = assign_people(pose, mapping)
        rows = rows[(rows.t >= c.start_s) & (rows.t <= c.end_s)]
        people = read_table(vdir / "people.csv")
        key = "role_key" if "role_key" in people.columns else "role"
        excluded = set(people[people[key].isin(["transient", "on_screen"])].person_id)
        rows = rows[~rows.pid.isin(excluded)]
        clips.at[i, "people"] = " ".join(f"P{p}" for p in sorted(rows.pid.unique()))
        log.info("Rendering coding clip %s (%s, %s %.0f-%.0f s)", c.clip_id, c.video_id, c.phase, c.start_s, c.end_s)
        render_coding_video(Path(c.video_path), val_dir / "clips" / f"{c.clip_id}.mp4", rows, c.start_s, c.end_s)
    clips.to_csv(val_dir / "clips.csv", index=False, encoding="utf-8-sig")

    wb = Workbook()
    ws = wb.active
    ws.title = "annotations"
    ws.append(["clip_id", "person_id", "gesture", "start", "end", "notes"])
    for col, width in zip("ABCDEF", (9, 10, 22, 10, 10, 40)):
        ws.column_dimensions[col].width = width
    for col in "DE":                                       # keep "25:12" as typed, not a time of day
        for r in range(2, 2001):
            ws[f"{col}{r}"].number_format = "@"
    dv_g = DataValidation(type="list", formula1='"' + ",".join(CODABLE) + '"', allow_blank=True)
    dv_c = DataValidation(type="list", formula1='"' + ",".join(clips.clip_id) + '"', allow_blank=True)
    ws.add_data_validation(dv_g)
    ws.add_data_validation(dv_c)
    dv_g.add("C2:C2000")
    dv_c.add("A2:A2000")
    ws.freeze_panes = "A2"
    cs = wb.create_sheet("clips")
    cs.append(["clip_id", "video", "part", "from", "to", "people in clip"])
    for _, c in clips.iterrows():
        cs.append([c.clip_id, c.video_id, c.phase, _clock(c.start_s), _clock(c.end_s), c.people])
    gs = wb.create_sheet("coded_gestures")
    gs.append(["gesture", "coded? (yes/no)"])
    for g in CODABLE:
        gs.append([g, "yes"])
    wb.save(val_dir / "coding_sheet.xlsx")
    (val_dir / "INSTRUCTIONS.md").write_text(INSTRUCTIONS, encoding="utf-8")
    return clips


def _clock(t: float) -> str:
    mm, ss = divmod(float(t), 60)
    return f"{int(mm):02d}:{ss:04.1f}"


# ----------------------------------------------------------------------------- validate

def parse_time(v) -> float | None:
    """Accepts 25:12, 25:12.5, 1:25:12, 1512.5 (seconds), or an Excel time value."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return None
    if isinstance(v, dt.time):
        return v.hour * 3600 + v.minute * 60 + v.second + v.microsecond / 1e6   # typed mm:ss read as hh:mm
    if isinstance(v, dt.timedelta):
        return v.total_seconds()
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip().replace("：", ":")
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    parts = s.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    sec = 0.0
    for n in nums:
        sec = sec * 60 + n
    return sec


def parse_person(v) -> int | None:
    """Accepts 3, 3.0 (Excel stores numbers as floats), "3", "P3", "p 3"."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return int(round(float(v)))
    m = re.fullmatch(r"\s*[Pp]?\s*(\d+)(\.0+)?\s*", str(v))
    return int(m.group(1)) if m else None


def load_annotations(val_dir: Path) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    rows, coded = [], {}
    for f in sorted(Path(val_dir).glob("coding_sheet*.xlsx")):
        coder = f.stem.replace("coding_sheet", "").strip("_- ") or "coder"
        sheets = pd.read_excel(f, sheet_name=None, dtype={"start": object, "end": object})
        ann = sheets.get("annotations", pd.DataFrame()).dropna(how="all")
        if ann.empty:
            continue                                     # the blank template, or an unused copy
        cg = sheets.get("coded_gestures")
        coded[coder] = set(cg[cg.iloc[:, 1].astype(str).str.lower().str.startswith("y")].iloc[:, 0]) \
            if cg is not None and len(cg) else set(CODABLE)
        for _, r in ann.dropna(subset=["clip_id", "gesture"]).iterrows():
            s, e = parse_time(r.get("start")), parse_time(r.get("end"))
            if s is None:
                log.warning("%s: row without a start time skipped (%s)", f.name, dict(r))
                continue
            e = s if e is None else e
            pid = parse_person(r.person_id)
            if pid is None:
                log.warning("%s: row with unreadable person_id skipped (%s)", f.name, dict(r))
                continue
            rows.append({"coder": coder, "clip_id": str(r.clip_id).strip(), "person_id": pid,
                         "gesture": str(r.gesture).strip(), "start_s": min(s, e), "end_s": max(s, e),
                         "notes": r.get("notes", "")})
    return pd.DataFrame(rows, columns=["coder", "clip_id", "person_id", "gesture", "start_s", "end_s", "notes"]), coded


def match(a: pd.DataFrame, b: pd.DataFrame, tol: float) -> tuple[int, list[int], list[int]]:
    """Greedy one-to-one matching by largest overlap. Returns (matches, unmatched a idx, unmatched b idx)."""
    pairs = []
    for i, x in a.iterrows():
        for j, y in b.iterrows():
            if x.person_id != y.person_id or x.gesture != y.gesture:
                continue
            ov = min(x.end_s, y.end_s) + tol - (max(x.start_s, y.start_s) - tol)
            if ov > 0:
                pairs.append((ov, i, j))
    used_a, used_b = set(), set()
    for ov, i, j in sorted(pairs, reverse=True):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
    return len(used_a), [i for i in a.index if i not in used_a], [j for j in b.index if j not in used_b]


def _time_overlap(h: pd.DataFrame, p: pd.DataFrame, t0: float, t1: float, step: float = 0.1) -> tuple[float, float, float]:
    """Seconds marked by the person, by the program, and by both, inside [t0, t1)."""
    grid = np.arange(t0, t1, step)

    def mask(df):
        m = np.zeros(len(grid), bool)
        for s, e in zip(df.start_s, df.end_s):
            m |= (grid >= s) & (grid < e)
        return m
    mh, mp_ = mask(h), mask(p)
    return round(mh.sum() * step, 1), round(mp_.sum() * step, 1), round((mh & mp_).sum() * step, 1)


def _program_events(output_dir: Path, clip: pd.Series) -> pd.DataFrame:
    vdir = Path(output_dir) / clip.video_id
    ev = read_table(vdir / "events.csv") if (vdir / "events.csv").exists() else pd.DataFrame()
    talk = read_table(vdir / "talk.csv") if (vdir / "talk.csv").exists() else pd.DataFrame()
    parts = []
    if len(ev):
        parts.append(ev[["person_id", "gesture", "start_s", "end_s"]])
    if len(talk):
        t = talk[talk.person_id >= 0][["person_id", "start_s", "end_s"]].assign(gesture="speaking")
        parts.append(t)
    if not parts:
        return pd.DataFrame(columns=["person_id", "gesture", "start_s", "end_s"])
    allev = pd.concat(parts, ignore_index=True)
    inside = (allev.end_s > clip.start_s) & (allev.start_s < clip.end_s)
    return allev[inside].reset_index(drop=True)


def _prf(tp: int, n_prog: int, n_human: int) -> dict:
    p = tp / n_prog if n_prog else None
    r = tp / n_human if n_human else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if (p == 0 or r == 0) else None)
    return {"precision": None if p is None else round(p, 3), "recall": None if r is None else round(r, 3),
            "f1": None if f is None else round(f, 3)}


def validate(output_dir: Path, val_dir: Path, tolerance_s: float = 1.0) -> dict[str, pd.DataFrame]:
    val_dir = Path(val_dir)
    clips = read_table(val_dir / "clips.csv")
    ann, coded = load_annotations(val_dir)
    if ann.empty:
        raise RuntimeError(f"No filled-in coding sheets (coding_sheet*.xlsx with rows) found in {val_dir}")
    acc_rows, mism = [], []
    for coder, a_c in ann.groupby("coder"):
        for _, clip in clips[clips.clip_id.isin(a_c.clip_id)].iterrows():
            human = a_c[a_c.clip_id == clip.clip_id].reset_index(drop=True)
            prog = _program_events(output_dir, clip)
            prog = prog[prog.gesture.isin(coded[coder])].reset_index(drop=True)
            for g in sorted(set(human.gesture) | set(prog.gesture)):
                h, p = human[human.gesture == g], prog[prog.gesture == g]
                tp, miss_h, miss_p = match(h, p, tolerance_s)
                acc_rows.append({"coder": coder, "clip_id": clip.clip_id, "gesture": g,
                                 "marked_by_person": len(h), "found_by_program": len(p), "matched": tp})
                for i in miss_h:
                    mism.append({"coder": coder, "clip_id": clip.clip_id, "kind": "missed by program",
                                 **h.loc[i, ["person_id", "gesture", "start_s", "end_s"]].to_dict()})
                for j in miss_p:
                    mism.append({"coder": coder, "clip_id": clip.clip_id, "kind": "extra (person did not mark)",
                                 **p.loc[j, ["person_id", "gesture", "start_s", "end_s"]].to_dict()})
    per_clip = pd.DataFrame(acc_rows)

    # Ongoing states (facing, turned away, head down, mic, gesturing, speaking) are compared
    # by time: a person marking one long stretch and the program reporting it in pieces
    # should agree, not count as one match plus several "extras".
    time_rows = []
    for coder, a_c in ann.groupby("coder"):
        for _, clip in clips[clips.clip_id.isin(a_c.clip_id)].iterrows():
            human = a_c[a_c.clip_id == clip.clip_id]
            prog = _program_events(output_dir, clip)
            for g in sorted((STATE_GESTURES & coded[coder]) & (set(human.gesture) | set(prog.gesture))):
                for pid in sorted(set(human[human.gesture == g].person_id) | set(prog[prog.gesture == g].person_id)):
                    h = human[(human.gesture == g) & (human.person_id == pid)]
                    pr = prog[(prog.gesture == g) & (prog.person_id == pid)]
                    hs, ps, both = _time_overlap(h, pr, clip.start_s, clip.end_s)
                    time_rows.append({"coder": coder, "clip_id": clip.clip_id, "gesture": g, "person_id": pid,
                                      "person_s": hs, "program_s": ps, "both_s": both})
    time_df = pd.DataFrame(time_rows, columns=["coder", "clip_id", "gesture", "person_id", "person_s",
                                               "program_s", "both_s"])
    if len(time_df):
        tt = time_df.groupby("gesture")[["person_s", "program_s", "both_s"]].sum().reset_index()
        tt["time_precision"] = (tt.both_s / tt.program_s).where(tt.program_s > 0).round(3)
        tt["time_recall"] = (tt.both_s / tt.person_s).where(tt.person_s > 0).round(3)
    else:
        tt = pd.DataFrame(columns=["gesture", "person_s", "program_s", "both_s", "time_precision", "time_recall"])
    tot = per_clip.groupby("gesture")[["marked_by_person", "found_by_program", "matched"]].sum().reset_index()
    tot = pd.concat([tot, tot[["marked_by_person", "found_by_program", "matched"]].sum().to_frame().T.assign(gesture="ALL")],
                    ignore_index=True)
    stats = pd.DataFrame([_prf(int(r.matched), int(r.found_by_program), int(r.marked_by_person)) for r in tot.itertuples()])
    accuracy = pd.concat([tot, stats], axis=1)

    # agreement between coders on clips both coded (same matching rule)
    agree = []
    coders = sorted(ann.coder.unique())
    for i, c1 in enumerate(coders):
        for c2 in coders[i + 1:]:
            both = set(ann[ann.coder == c1].clip_id) & set(ann[ann.coder == c2].clip_id)
            gs = coded[c1] & coded[c2]
            for g in sorted(gs):
                a = ann[(ann.coder == c1) & ann.clip_id.isin(both) & (ann.gesture == g)].reset_index(drop=True)
                b = ann[(ann.coder == c2) & ann.clip_id.isin(both) & (ann.gesture == g)].reset_index(drop=True)
                if len(a) + len(b) == 0:
                    continue
                tp = sum(match(a[a.clip_id == c], b[b.clip_id == c], tolerance_s)[0] for c in both)
                agree.append({"coder_a": c1, "coder_b": c2, "gesture": g, "clips": len(both),
                              "marks_a": len(a), "marks_b": len(b), "agreed": tp,
                              "agreement_f1": round(2 * tp / (len(a) + len(b)), 3)})
    accuracy["measure"] = accuracy.gesture.map(lambda g: "time: see 'Time agreement'" if g in STATE_GESTURES else "events")
    return {"Accuracy": accuracy, "Time agreement": tt, "Time per person": time_df,
            "Per clip": per_clip, "Mismatches": pd.DataFrame(mism),
            "Coder agreement": pd.DataFrame(agree), "Annotations": ann}


def write_report(tables: dict[str, pd.DataFrame], path: Path) -> Path:
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in tables.items():
            (df if len(df) else pd.DataFrame({"(empty)": []})).to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "A2"
            if ws.max_row > 1:
                ws.auto_filter.ref = ws.dimensions
    return path
