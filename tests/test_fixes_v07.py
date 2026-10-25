"""Regression tests for the fixes in 0.7.0 and the facing / hand measures."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from gesture_pipeline.measures import facing_and_hands, timeline
from gesture_pipeline.speech import clip_to_window, summarize_talk
from gesture_pipeline.tables import read_table


# --- bug 1: all-digit video names --------------------------------------------------------

def test_numeric_video_id_matches_across_files(tmp_path):
    pd.DataFrame([{"video_id": "1012", "person_id": 7, "role": "presenter", "visible_s": 60}]).to_csv(
        tmp_path / "people.csv", index=False)
    people = read_table(tmp_path / "people.csv")
    assert people.video_id.iloc[0] == "1012"                      # text, not the number 1012
    talk = pd.DataFrame([{"video_id": "1012", "person_id": 7, "role": "presenter", "start_s": 0,
                          "end_s": 30, "duration_s": 30.0, "method": "mic"}])
    s = summarize_talk(talk, people, pd.DataFrame()).set_index("person_id")
    assert s.loc[7, "talk_s"] == 30.0


def test_read_table_handles_missing_and_empty(tmp_path):
    assert read_table(tmp_path / "nope.csv").empty
    (tmp_path / "empty.csv").write_text("\n")
    assert read_table(tmp_path / "empty.csv").empty


# --- bug 3: partial runs ----------------------------------------------------------------

def test_speech_clipped_to_analysed_window():
    sp = pd.DataFrame([{"video_id": "v", "start_s": a, "end_s": b, "duration_s": b - a}
                       for a, b in [(0, 50), (100, 130), (200, 3000)]])
    out = clip_to_window(sp, 0, 120)
    assert list(out.start_s) == [0, 100] and list(out.end_s) == [50, 120]
    assert out.duration_s.sum() == 70
    assert len(clip_to_window(sp)) == 3                            # no window: unchanged


# --- bugs 4/5: hand crops ----------------------------------------------------------------

def _pt(x, y):
    return SimpleNamespace(x=x, y=y, z=0.0)


def _hand(wx, wy):
    """21 points: wrist at (wx, wy) in crop coordinates, fingers above it."""
    return [_pt(wx, wy)] + [_pt(wx, wy - 0.02 * (i % 4 + 1)) for i in range(20)]


def test_hand_crop_keeps_only_the_hand_at_this_wrist_with_its_own_label(tmp_path):
    from gesture_pipeline.config import Config
    from gesture_pipeline.extractors_hand import HandTracker
    from gesture_pipeline.store import LandmarkWriter

    cfg = Config()
    cfg.extract.model_dir = str(tmp_path)                          # no model file -> disabled
    ht = HandTracker(cfg)
    assert not ht.enabled

    # Fake MediaPipe: the crop contains a neighbour's hand (labelled Pointing_Up) far from the
    # wrist, and this person's hand (labelled Open_Palm) right at the wrist.
    def recognize(_img):
        return SimpleNamespace(
            hand_landmarks=[_hand(0.05, 0.05), _hand(0.5, 0.55)],
            gestures=[[SimpleNamespace(category_name="Pointing_Up", score=0.9)],
                      [SimpleNamespace(category_name="Open_Palm", score=0.8)]],
            handedness=[[SimpleNamespace(score=0.9)], [SimpleNamespace(score=0.9)]])
    ht.enabled, ht.recognizer = True, SimpleNamespace(recognize=recognize, close=lambda: None)
    ht.mp = SimpleNamespace(Image=lambda image_format, data: data, ImageFormat=SimpleNamespace(SRGB=1))

    img = np.zeros((720, 1280, 3), np.uint8)
    kxy = np.zeros((17, 2))
    kcf = np.zeros(17)
    kxy[8], kxy[10] = (600, 360), (600, 400)                        # right elbow, right wrist
    kcf[[8, 10]] = 0.9
    with LandmarkWriter(tmp_path / "lm", tmp_path / "f.parquet") as w:
        rows = ht.process_person_hands(img, 0, 0.0, 1, kxy, kcf, 0.05, w)
    assert len(rows) == 1
    assert json.loads(rows[0]["extra"])["gesture"] == "open_palm"
    assert json.loads(rows[0]["extra"])["mp_category"] == "Open_Palm"


def test_hand_listed_as_model_turns_on_hand_tracking():
    from gesture_pipeline.config import Config
    cfg = Config()
    cfg.extract.models = ["pose", "hand"]
    cfg.validate()
    assert cfg.extract.models == ["pose"] and cfg.extract.detect_hands


# --- facing / hands measures --------------------------------------------------------------

def _people():
    return pd.DataFrame([{"video_id": "v", "person_id": 1, "role": "presenter", "role_key": "presenter", "visible_s": 60},
                         {"video_id": "v", "person_id": 2, "role": "audience", "role_key": "audience", "visible_s": 60}])


def _events():
    E = lambda p, g, a, b: {"video_id": "v", "person_id": p, "gesture": g, "start_s": a, "end_s": b, "duration_s": b - a}
    return pd.DataFrame([E(1, "facing_audience", 0, 20), E(1, "look_away", 20, 30), E(1, "facing_audience", 30, 50),
                         E(1, "head_down", 50, 60), E(1, "pointing", 5, 6), E(1, "open_palm", 40, 42),
                         E(2, "hand_raise", 1, 3)])


def test_facing_and_hands_summary():
    vis = pd.DataFrame([{"video_id": "v", "person_id": 1, "phase": "all", "visible_s": 60.0}])
    s = facing_and_hands(_events(), _people(), vis, hands_tracked=True)
    assert list(s.person_id) == [1]                                 # audience not included
    r = s.iloc[0]
    assert r.facing_s == 40 and abs(r.facing_share - 0.667) < 0.01
    assert r.turned_away_n == 1 and r.turned_away_s == 10 and r.head_down_s == 10
    assert r.pointing_n == 1 and r.open_palm_n == 1 and r.pointing_per_min == 1.0


def test_hands_columns_empty_when_not_tracked():
    s = facing_and_hands(_events(), _people(), pd.DataFrame(), hands_tracked=False)
    assert s.pointing_n.isna().all()


def test_timeline_states_per_second():
    rows = pd.DataFrame([{"pid": 1, "t": t / 8} for t in range(0, 60 * 8)])
    tl = timeline(_events(), _people(), rows, "v").set_index("second")
    assert tl.loc[10, "state"] == "facing" and tl.loc[25, "state"] == "turned_away"
    assert tl.loc[55, "state"] == "head_down"
    assert tl.loc[5, "pointing"] == 1 and tl.loc[6, "pointing"] == 0 and tl.loc[41, "open_palm"] == 1
    assert len(tl) == 60 and tl.loc[59, "clock"] == "00:59"
