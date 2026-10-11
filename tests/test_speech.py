"""Who-is-talking attribution, on hand-made timelines (no audio model needed)."""

import numpy as np
import pandas as pd

from gesture_pipeline.config import SpeechConfig
from gesture_pipeline.speech import attribute_speech, summarize_talk

from .conftest import needs_ffmpeg


def _people():
    return pd.DataFrame([
        {"video_id": "v", "person_id": 1, "role": "presenter", "role_key": "presenter", "visible_s": 60},
        {"video_id": "v", "person_id": 2, "role": "panel", "role_key": "panel", "visible_s": 60},
        {"video_id": "v", "person_id": 3, "role": "panel", "role_key": "panel", "visible_s": 60},
    ])


def _rows(pids_until):
    """Everyone visible every 1/8 s; pids_until = {pid: last second visible}."""
    return pd.DataFrame([{"pid": p, "t": t} for p, last in pids_until.items()
                         for t in np.arange(0, last, 0.125)])


def _speech(*spans):
    return pd.DataFrame([{"video_id": "v", "start_s": a, "end_s": b, "duration_s": b - a} for a, b in spans])


def _mic(*items):
    return pd.DataFrame([{"video_id": "v", "person_id": p, "gesture": "holding_mic", "start_s": a, "end_s": b,
                          "duration_s": b - a, "confidence": 0.9} for p, a, b in items])


PHASES = pd.DataFrame([{"phase": "presentation", "start_s": 0, "end_s": 30}, {"phase": "qa", "start_s": 30, "end_s": 60}])


def test_presenter_then_two_panelists_with_mic():
    talk = attribute_speech(_speech((0, 20), (32, 40), (41, 50)), _mic((2, 31, 40.5), (3, 40.6, 51)),
                            _people(), PHASES, _rows({1: 30, 2: 60, 3: 60}), SpeechConfig(), "v")
    by = talk.groupby("person_id").duration_s.sum()
    assert abs(by[1] - 20) < 0.5                 # presenter during presentation (no mic needed)
    assert abs(by[2] - 8) < 0.5 and abs(by[3] - 9) < 0.5
    assert set(talk.method) == {"presenter", "mic"}


def test_speech_with_nobody_identifiable_is_unattributed():
    talk = attribute_speech(_speech((35, 45)), _mic(), _people(), PHASES, _rows({2: 60, 3: 60}),
                            SpeechConfig(), "v")
    assert list(talk.person_id.unique()) == [-1] and abs(talk.duration_s.sum() - 10) < 0.5


def test_presenter_not_credited_when_off_screen():
    talk = attribute_speech(_speech((5, 15)), _mic(), _people(), PHASES, _rows({1: 3, 2: 60}),
                            SpeechConfig(), "v")
    assert (talk.person_id == -1).all()


def test_short_pause_does_not_split_a_turn_and_is_not_counted_as_talk():
    talk = attribute_speech(_speech((0, 5), (5.6, 10)), _mic(), _people(), PHASES, _rows({1: 30}),
                            SpeechConfig(), "v")
    assert len(talk) == 1 and talk.iloc[0].person_id == 1
    assert abs(talk.iloc[0].duration_s - 9.4) < 0.15      # speech only, not the 0.6 s pause
    assert talk.iloc[0].end_s - talk.iloc[0].start_s >= 9.9


def test_attributed_never_exceeds_detected_speech():
    sp = _speech((0, 3), (3.5, 7), (8, 12), (31, 33), (33.8, 39), (40, 52))
    talk = attribute_speech(sp, _mic((2, 30, 45), (3, 45.2, 60)), _people(), PHASES,
                            _rows({1: 30, 2: 60, 3: 60}), SpeechConfig(), "v")
    assert talk.duration_s.sum() <= sp.duration_s.sum() + 0.01


def test_summary_includes_silent_people_and_unattributed():
    talk = attribute_speech(_speech((0, 10), (35, 40)), _mic(), _people(), PHASES, _rows({1: 30, 2: 60}),
                            SpeechConfig(), "v")
    s = summarize_talk(talk, _people(), pd.DataFrame())
    assert set(s.person_id) == {1, 2, 3, -1}
    assert s.set_index("person_id").loc[2, "talk_s"] == 0
    assert abs(s.set_index("person_id").loc[-1, "talk_s"] - 5) < 0.5


def test_no_speech():
    assert attribute_speech(_speech(), _mic(), _people(), PHASES, _rows({1: 10}), SpeechConfig(), "v").empty


@needs_ffmpeg
def test_vad_runs_on_real_audio_file(av_video, tmp_path):
    from gesture_pipeline.ingest import extract_audio
    from gesture_pipeline.speech import detect_speech
    wav = extract_audio(av_video, tmp_path / "a.wav")
    out = detect_speech(wav, SpeechConfig(), "v")       # a pure tone: little or no speech
    assert list(out.columns) == ["video_id", "start_s", "end_s", "duration_s"]
    assert out.duration_s.sum() < 1.5
