import json

import pandas as pd
import pytest

from gesture_pipeline.config import load_config
from gesture_pipeline.export import ExportFilters, export_batch, summarize, validate_events, write_video_outputs
from gesture_pipeline.paths import video_id_for
from gesture_pipeline.status import StageStatus
from gesture_pipeline.store import LandmarkWriter, available_models, read_frames, read_landmarks


def test_landmark_roundtrip(tmp_path):
    lm, fr = tmp_path / "landmarks", tmp_path / "frames.parquet"
    with LandmarkWriter(lm, fr, rows_per_part=3) as w:
        for i in range(7):
            w.add_frame(i, i * 3, i / 8)
            w.add("pose", {"frame": i, "t": i / 8, "person_id": 1, "x": [0.1] * 33, "y": [0.2] * 33})
        w.add("face", {"frame": 0, "t": 0.0, "x": [0.5], "y": [0.5], "extra": json.dumps({"yaw": 3})})
    assert available_models(lm) == ["face", "pose"]
    pose = read_landmarks(lm, "pose")
    assert len(pose) == 7 and len(pose.iloc[0]["x"]) == 33
    assert len(list((lm / "pose").glob("*.parquet"))) == 3
    assert list(read_frames(fr)["source_index"]) == [0, 3, 6, 9, 12, 15, 18]
    assert read_landmarks(lm, "hand").empty


def test_crash_leaves_no_partial_output(tmp_path):
    lm, fr = tmp_path / "landmarks", tmp_path / "frames.parquet"
    with pytest.raises(RuntimeError):
        with LandmarkWriter(lm, fr, rows_per_part=1) as w:
            w.add("pose", {"frame": 0, "t": 0.0, "x": [0.1], "y": [0.1]})
            raise RuntimeError("boom")
    assert not lm.exists() and not fr.exists()
    assert not (tmp_path / "landmarks.partial").exists()


def test_status_resume_and_fingerprints(tmp_path):
    s = StageStatus(tmp_path / "status.json")
    s.mark("probe", "done", "abc")
    s2 = StageStatus(tmp_path / "status.json")
    assert s2.is_done("probe", "abc")
    assert not s2.is_done("probe", "changed")
    (tmp_path / "status.json").write_text("{corrupt")
    assert not StageStatus(tmp_path / "status.json").is_done("probe", "abc")


def test_config_overrides_and_typos(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("sampling:\n  fps: 5\n")
    cfg = load_config(p)
    assert cfg.sampling.fps == 5 and cfg.sampling.max_width == 1280
    p.write_text("sampling:\n  fsp: 5\n")
    with pytest.raises(KeyError, match="sampling.fsp"):
        load_config(p)
    p.write_text("sampling:\n  fps: 0\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_video_id_keeps_chinese():
    from pathlib import Path
    assert video_id_for(Path("115.10.08--春源(2010)法人說明會.mp4")) == "115.10.08--春源_2010_法人說明會"


def _events():
    return pd.DataFrame([
        ["v1", 1, "panel", "qa", "nod", 1.0, 2.0, 1.0, 0.9],
        ["v1", 1, "panel", "qa", "nod", 5.0, 5.5, 0.5, 0.8],
        ["v1", 2, "audience", "qa", "hand_raise", 3.0, 6.0, 3.0, 1.0],
    ], columns=["video_id", "person_id", "role", "phase", "gesture", "start_s", "end_s", "duration_s", "confidence"])


def _people():
    return pd.DataFrame([
        {"video_id": "v1", "person_id": 1, "role": "panel", "role_key": "panel", "visible_s": 120.0},
        {"video_id": "v1", "person_id": 2, "role": "audience", "role_key": "audience", "visible_s": 60.0},
        {"video_id": "v1", "person_id": 3, "role": "transient", "role_key": "transient", "visible_s": 2.0},
    ])


ROLE_GESTURES = {"nod": ["panel"], "hand_raise": ["panel", "audience"]}


def test_summary_counts_zeros_and_rates():
    s = summarize(_events(), _people(), pd.DataFrame(), ROLE_GESTURES)
    nod = s[(s.person_id == 1) & (s.gesture == "nod")].iloc[0]
    assert nod["count"] == 2 and nod["total_duration_s"] == 1.5 and nod["rate_per_min"] == 1.0
    zero = s[(s.person_id == 1) & (s.gesture == "hand_raise")].iloc[0]
    assert zero["count"] == 0                     # zeros are reported, not missing
    assert not ((s.person_id == 2) & (s.gesture == "nod")).any()  # nod not applicable to audience


def test_validate_rejects_bad_events():
    ev = _events()
    ev.loc[0, "end_s"] = 0.0
    with pytest.raises(ValueError):
        validate_events(ev)
    with pytest.raises(ValueError):
        validate_events(ev.drop(columns=["gesture"]))


def _video_dir(tmp_path):
    vdir = tmp_path / "v1"
    vdir.mkdir()
    (vdir / "status.json").write_text(json.dumps({"stages": {"probe": {"state": "done"}}}))
    (vdir / "metadata.json").write_text(json.dumps({"duration_s": 10}))
    _people().to_csv(vdir / "people.csv", index=False)
    write_video_outputs(_events(), _people(), pd.DataFrame(), ROLE_GESTURES, vdir / "events.csv", vdir / "summary.csv")
    return vdir


def test_export_workbook(tmp_path):
    _video_dir(tmp_path)
    xlsx = export_batch(tmp_path, role_gestures=ROLE_GESTURES)
    sheets = pd.read_excel(xlsx, sheet_name=None)
    assert set(sheets) == {"Facing & hands", "Summary", "By role", "Talk time", "Events", "People", "Phases",
                           "Videos", "Filters"}
    assert len(sheets["Events"]) == 3
    assert 3 not in set(sheets["Summary"].person_id)          # transient excluded by default


def test_export_filters(tmp_path):
    _video_dir(tmp_path)
    xlsx = export_batch(tmp_path, filters=ExportFilters(roles=["panel"], min_confidence=0.85),
                        role_gestures=ROLE_GESTURES)
    ev = pd.read_excel(xlsx, sheet_name="Events")
    assert list(ev.gesture) == ["nod"] and ev.confidence.iloc[0] == 0.9
    f = pd.read_excel(xlsx, sheet_name="Filters")
    assert "min_confidence" in set(f.setting)
