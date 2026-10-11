"""Per-video result files and the combined, filterable Excel workbook."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

EVENT_COLUMNS = ["video_id", "person_id", "role", "phase", "gesture", "start_s", "end_s",
                 "duration_s", "confidence"]
SUMMARY_COLUMNS = ["video_id", "person_id", "role", "gesture", "count", "total_duration_s",
                   "visible_min", "rate_per_min", "share_of_time"]
EXCEL_MAX_ROWS = 1_048_575


def empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def validate_events(events: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in EVENT_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"events table missing columns: {missing}")
    ev = events[EVENT_COLUMNS].copy()
    bad = ev["end_s"] < ev["start_s"]
    if bad.any():
        raise ValueError(f"{int(bad.sum())} events end before they start")
    return ev.sort_values(["video_id", "start_s", "person_id"], kind="stable").reset_index(drop=True)


def summarize(events: pd.DataFrame, people: pd.DataFrame, visibility: pd.DataFrame,
              role_gestures: dict[str, list[str]], gestures: list[str] | None = None) -> pd.DataFrame:
    """Counts per person x gesture, INCLUDING zeros for gestures that apply to the person's
    role, plus a rate per minute visible so long and short appearances are comparable."""
    if people.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    vis = visibility.groupby(["video_id", "person_id"]).visible_s.sum() if not visibility.empty else None
    rows = []
    for _, p in people.iterrows():
        key = p.get("role_key", p.role)
        applicable = [g for g, roles in role_gestures.items() if key in roles]
        if gestures:
            applicable = [g for g in applicable if g in gestures]
        visible_s = float(vis.get((p.video_id, p.person_id), 0.0)) if vis is not None else float(p.visible_s)
        mine = events[(events.video_id == p.video_id) & (events.person_id == p.person_id)] if len(events) else events
        for g in applicable:
            ge = mine[mine.gesture == g] if len(mine) else mine
            n = int(len(ge))
            rows.append({"video_id": p.video_id, "person_id": int(p.person_id), "role": p.role, "gesture": g,
                         "count": n, "total_duration_s": round(float(ge.duration_s.sum()) if n else 0.0, 2),
                         "visible_min": round(visible_s / 60, 2),
                         "rate_per_min": round(n / (visible_s / 60), 3) if visible_s > 0 else None,
                         # for ongoing behaviours (holding_mic, gesturing, head_down) the share of
                         # time is usually more meaningful than the number of events
                         "share_of_time": round(min(1.0, float(ge.duration_s.sum()) / visible_s), 3)
                         if n and visible_s > 0 else 0.0})
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def by_role(summary: pd.DataFrame) -> pd.DataFrame:
    cols = ["video_id", "role", "gesture", "people", "events", "mean_rate_per_min", "mean_share_of_time",
            "total_duration_s"]
    if summary.empty:
        return pd.DataFrame(columns=cols)
    g = summary.groupby(["video_id", "role", "gesture"])
    out = g.agg(people=("person_id", "nunique"), events=("count", "sum"),
                mean_rate_per_min=("rate_per_min", "mean"), mean_share_of_time=("share_of_time", "mean"),
                total_duration_s=("total_duration_s", "sum")).reset_index()
    out["mean_rate_per_min"] = out["mean_rate_per_min"].round(3)
    out["mean_share_of_time"] = out["mean_share_of_time"].round(3)
    return out[cols]


def write_video_outputs(events: pd.DataFrame, people: pd.DataFrame, visibility: pd.DataFrame,
                        role_gestures: dict[str, list[str]], events_csv: Path, summary_csv: Path) -> pd.DataFrame:
    ev = validate_events(events) if not events.empty else empty_events()
    # utf-8-sig so Excel shows Chinese text correctly when opening the CSV directly
    ev.to_csv(events_csv, index=False, encoding="utf-8-sig")
    summ = summarize(ev, people, visibility, role_gestures)
    summ.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    return summ


@dataclass
class ExportFilters:
    roles: list[str] | None = None                       # keep only these role labels
    exclude_roles: list[str] = field(default_factory=lambda: ["transient", "other", "on_screen"])
    gestures: list[str] | None = None
    min_confidence: float = 0.0
    phase: str | None = None                            # presentation | qa | all | other

    def apply_people(self, people: pd.DataFrame) -> pd.DataFrame:
        if people.empty:
            return people
        keep = ~people.role.isin(self.exclude_roles) & ~people.get("role_key", people.role).isin(self.exclude_roles)
        if self.roles:
            keep &= people.role.isin(self.roles) | people.get("role_key", people.role).isin(self.roles)
        return people[keep]

    def apply_events(self, events: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
        if events.empty:
            return events
        ev = events.merge(people[["video_id", "person_id"]], on=["video_id", "person_id"])
        if self.gestures:
            ev = ev[ev.gesture.isin(self.gestures)]
        if self.min_confidence:
            ev = ev[ev.confidence >= self.min_confidence]
        if self.phase:
            ev = ev[ev.phase == self.phase]
        return ev


def _read(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths if p.exists() and p.stat().st_size > 0]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def export_batch(output_dir: Path, xlsx_path: Path | None = None, filters: ExportFilters | None = None,
                 role_gestures: dict[str, list[str]] | None = None) -> Path:
    """One workbook for every processed video: Summary, By role, Events, People, Phases,
    Videos, Filters (a record of exactly which filters produced the numbers)."""
    from .config import DEFAULT_ROLE_GESTURES

    filters = filters or ExportFilters()
    role_gestures = role_gestures or DEFAULT_ROLE_GESTURES
    output_dir = Path(output_dir)
    video_dirs = sorted(d for d in output_dir.iterdir() if d.is_dir() and (d / "status.json").exists())

    meta_rows = []
    for d in video_dirs:
        row = {"video_id": d.name}
        if (d / "metadata.json").exists():
            row.update(json.loads((d / "metadata.json").read_text(encoding="utf-8")))
        st = json.loads((d / "status.json").read_text(encoding="utf-8")).get("stages", {})
        row["stages_done"] = ",".join(k for k, v in st.items() if v.get("state") == "done")
        row["errors"] = "; ".join(f"{k}: {v.get('error')}" for k, v in st.items() if v.get("state") == "failed")
        meta_rows.append(row)
    videos = pd.DataFrame(meta_rows)

    people_all = _read([d / "people.csv" for d in video_dirs])
    events_all = _read([d / "events.csv" for d in video_dirs])
    phases = _read([d / "phases.csv" for d in video_dirs])
    visibility = _read([d / "visibility.csv" for d in video_dirs])

    people = filters.apply_people(people_all)
    events = filters.apply_events(events_all, people) if len(people) else empty_events()
    vis = visibility
    if filters.phase and not visibility.empty:
        vis = visibility[visibility.phase == filters.phase]
    summary = summarize(events, people, vis, role_gestures, filters.gestures)
    roles = by_role(summary)

    filt = pd.DataFrame([{"setting": k, "value": json.dumps(v, ensure_ascii=False)} for k, v in asdict(filters).items()] +
                        [{"setting": "exported_at", "value": datetime.now().isoformat(timespec="seconds")},
                         {"setting": "videos", "value": len(video_dirs)}])

    xlsx_path = Path(xlsx_path or output_dir / "results.xlsx")
    tmp = xlsx_path.with_suffix(".partial.xlsx")
    with pd.ExcelWriter(tmp, engine="openpyxl") as xw:
        for name, df in [("Summary", summary), ("By role", roles), ("Events", events),
                         ("People", people_all), ("Phases", phases), ("Videos", videos), ("Filters", filt)]:
            if len(df) > EXCEL_MAX_ROWS:
                log.warning("%s exceeds Excel's row limit; truncated (full data in the CSV files)", name)
                df = df.head(EXCEL_MAX_ROWS)
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "A2"
            if ws.max_row > 1:
                ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 50)
    tmp.replace(xlsx_path)
    log.info("Wrote %s (%d videos, %d events after filters)", xlsx_path, len(video_dirs), len(events))
    return xlsx_path
