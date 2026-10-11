"""Each gesture rule on synthetic keypoint sequences with realistic jitter.

Scale is similar to a panelist in the TWSE footage: shoulder width S = 0.05 of the
frame width (~64 px at 1280), nose-to-shoulder H = 0.035. Jitter: 2% of S per frame.
"""

import numpy as np
import pandas as pd
import pytest

from gesture_pipeline.config import DEFAULT_GESTURE_PARAMS, EventsConfig
from gesture_pipeline.gestures import build_track, detect_events, detect_person, detect_phases, runs
from gesture_pipeline.people import build_people

FPS = 8
S, H = 0.05, 0.035
RNG = np.random.default_rng(0)

BASE = np.array([
    [0.500, 0.400], [0.495, 0.395], [0.505, 0.395], [0.488, 0.400], [0.512, 0.400],  # nose eyes ears
    [0.475, 0.435], [0.525, 0.435],                                                 # shoulders
    [0.470, 0.480], [0.530, 0.480],                                                 # elbows
    [0.480, 0.500], [0.520, 0.500],                                                 # wrists (on table)
    [0.485, 0.510], [0.515, 0.510],                                                 # hips
    [0.485, 0.520], [0.515, 0.520],                                                 # knees (seated)
    [0.485, 0.560], [0.515, 0.560],                                                 # ankles
])


def make(seconds=20, modify=None, jitter=0.02, conf=0.9, pid=1, face=True):
    rows = []
    for f in range(int(seconds * FPS)):
        t = f / FPS
        kp = BASE.copy()
        if modify:
            kp = modify(t, kp)
        kp = kp + RNG.normal(0, jitter * S, kp.shape)
        v = np.full(17, conf)
        if not face:
            v[:5] = 0.05
        x1, y1 = kp[:, 0].min() - 0.01, kp[:, 1].min() - 0.02
        x2, y2 = kp[:, 0].max() + 0.01, kp[:, 1].max() + 0.01
        rows.append({"frame": f, "t": t, "person_id": pid, "pid": pid, "score": 0.9,
                     "bbox": [x1, y1, x2, y2], "x": kp[:, 0].tolist(), "y": kp[:, 1].tolist(),
                     "visibility": v.tolist()})
    return pd.DataFrame(rows)


def detect(df, gestures=None):
    frames = df[["frame", "t"]].drop_duplicates().set_index("frame").t
    tr = build_track(df, frames, FPS, DEFAULT_GESTURE_PARAMS["kp_conf"])
    wanted = set(gestures or EventsConfig().gestures)
    return pd.DataFrame(detect_person(tr, wanted, DEFAULT_GESTURE_PARAMS), columns=["gesture", "s", "e", "confidence"])


def counts(ev):
    return ev.gesture.value_counts().to_dict() if len(ev) else {}


def window(t, a, b):
    return a <= t < b


# ----------------------------------------------------------------------------- tests

def test_still_person_produces_nothing():
    assert counts(detect(make(60))) == {}


def test_nod():
    def nod(t, kp):
        if window(t, 8, 9.5):           # two dips of 0.35 H, 0.75 s each
            kp[:5, 1] += 0.35 * H * np.sin(np.pi * ((t - 8) % 0.75) / 0.75) ** 2
        return kp
    c = counts(detect(make(20, nod)))
    assert c.get("nod") == 1 and "head_shake" not in c


def test_two_separate_nods():
    def nod(t, kp):
        for a in (5, 12):
            if window(t, a, a + 0.75):
                kp[:5, 1] += 0.35 * H * np.sin(np.pi * (t - a) / 0.75) ** 2
        return kp
    assert counts(detect(make(20, nod))).get("nod") == 2


def test_tiny_head_jitter_is_not_a_nod():
    def wobble(t, kp):
        kp[:5, 1] += 0.04 * H * np.sin(2 * np.pi * 1.5 * t)   # dips of 0.08 H: well below 0.15 H
        return kp
    assert "nod" not in counts(detect(make(20, wobble)))


def test_head_shake():
    def shake(t, kp):
        if window(t, 6, 7.5):
            kp[:5, 0] += 0.25 * S * np.sin(2 * np.pi * 1.5 * (t - 6))
        return kp
    c = counts(detect(make(20, shake)))
    assert c.get("head_shake") == 1 and "nod" not in c


def test_hand_raise_not_also_above_shoulder():
    def raise_(t, kp):
        if window(t, 5, 8):
            kp[8] = [0.540, 0.420]                 # elbow up
            kp[10] = [0.545, 0.400 - 1.2 * H]      # wrist above head
        return kp
    c = counts(detect(make(15, raise_)))
    assert c.get("hand_raise") == 1 and "hand_above_shoulder" not in c


def test_short_hand_flick_is_not_a_raise():
    def flick(t, kp):
        if window(t, 5, 5.5):
            kp[8] = [0.540, 0.420]
            kp[10] = [0.545, 0.400 - 1.2 * H]
        return kp
    assert "hand_raise" not in counts(detect(make(15, flick)))


def test_hand_above_shoulder():
    def up(t, kp):
        if window(t, 5, 7):
            kp[8] = [0.545, 0.440]
            kp[10] = [0.560, 0.435 - 0.5 * S]      # above shoulder, below head top
        return kp
    c = counts(detect(make(15, up)))
    assert c.get("hand_above_shoulder") == 1 and "hand_raise" not in c


def test_holding_mic():
    def mic(t, kp):
        if window(t, 4, 10):
            kp[7] = [0.490, 0.470]                 # elbow
            kp[9] = [0.497, 0.425]                 # wrist under the chin
        return kp
    c = counts(detect(make(15, mic)))
    assert c.get("holding_mic") == 1
    assert "hand_raise" not in c and "hand_above_shoulder" not in c


def test_back_view_guessed_wrist_near_shoulder_is_ignored():
    def guess(t, kp):
        kp[8] = [0.535, 0.435 + 0.63 * S]          # elbow hanging down
        kp[10] = [0.530, 0.435 - 0.27 * S]         # wrist "guessed" just above the shoulder
        return kp
    assert counts(detect(make(20, guess, face=False))) == {}


def test_back_view_real_hand_to_head_counts():
    def phone(t, kp):
        if window(t, 5, 8):
            kp[8] = [0.540, 0.435 + 0.3 * S]
            kp[10] = [0.530, 0.435 - 0.7 * S]      # holding a phone up to take a photo
        return kp
    assert counts(detect(make(15, phone, face=False))).get("hand_above_shoulder") == 1


def test_mic_with_brief_dropouts_is_one_event():
    def mic(t, kp):
        if window(t, 2, 18) and not window(t, 7, 8) and not window(t, 12, 12.6):
            kp[7] = [0.490, 0.470]
            kp[9] = [0.497, 0.425]
        return kp
    assert counts(detect(make(20, mic))).get("holding_mic") == 1


def test_head_down():
    def down(t, kp):
        if window(t, 5, 11):
            kp[:5, 1] += 0.6 * H                    # nose much closer to shoulders
        return kp
    assert counts(detect(make(20, down))).get("head_down") == 1


def test_look_away():
    def turn(t, kp):
        if window(t, 5, 9):
            kp[:5, 0] += 0.5 * S
        return kp
    c = counts(detect(make(20, turn)))
    assert c.get("look_away") == 1 and "head_shake" not in c


def test_gesturing():
    def move(t, kp):
        if window(t, 5, 8):
            a = 2 * np.pi * 1.5 * t
            kp[10] = [0.52 + 0.4 * S * np.cos(a), 0.48 + 0.4 * S * np.sin(a)]
        return kp
    assert counts(detect(make(15, move))).get("gesturing") == 1


def test_stand_up_from_behind_and_sit_down():
    def stand(t, kp):
        if 6 <= t < 12:
            kp[:13, 1] -= 1.5 * S                 # upper body rises; legs hidden behind chairs
        return kp
    ev = detect(make(20, stand, face=False), {"stand_up"})
    assert counts(ev).get("stand_up") == 1
    dur = (ev.iloc[0].e - ev.iloc[0].s + 1) / FPS
    assert 5 <= dur <= 7                          # ends when they sit back down


def test_seated_audience_with_tall_box_is_not_standing():
    def lean(t, kp):
        kp[13:, 1] += 0.08                        # tall box, but shoulders never rise
        return kp
    assert counts(detect(make(20, lean, face=False), {"stand_up"})) == {}


def test_noisy_distant_person_needs_bigger_nod():
    rng_state = RNG.bit_generator.state
    noisy = make(30, jitter=0.12)                 # 6x the usual jitter, no real movement
    RNG.bit_generator.state = rng_state
    assert "nod" not in counts(detect(noisy))


def test_missing_frames_are_tolerated():
    df = make(20).sample(frac=0.6, random_state=1).sort_values("frame")
    assert counts(detect(df)) == {}


def test_runs_helper():
    m = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1], bool)
    assert runs(m, min_len=2, merge_gap=1) == [(1, 6)]
    assert runs(m, min_len=1, merge_gap=0) == [(1, 2), (4, 6), (10, 10)]


def test_phases_and_end_to_end_roles():
    # presenter standing on the right for 40 s, then 4 panelists seated for 40 s
    def presenter(t, kp):
        kp = kp.copy()
        kp[:, 0] += 0.3
        kp[13:15, 1] = 0.58
        kp[15:17, 1] = 0.70
        return kp
    pres = make(40, presenter, pid=1)
    pres["person_id"] = 1
    panel = []
    for i in range(4):
        d = make(40, lambda t, kp, i=i: kp + np.array([-0.3 + 0.12 * i, 0.0]), pid=10 + i)
        d["frame"] += 320
        d["t"] += 40
        d["person_id"] = 10 + i
        panel.append(d)
    pose = pd.concat([pres, *panel], ignore_index=True).drop(columns="pid")
    people, mapping = build_people(pose, __import__("gesture_pipeline.config", fromlist=["PeopleConfig"]).PeopleConfig(), "v", FPS)
    assert sorted(people.role_key) == ["panel"] * 4 + ["presenter"]
    from gesture_pipeline.people import assign_people
    rows = assign_people(pose, mapping)
    frames = rows[["frame", "t"]].drop_duplicates().sort_values("frame")
    phases = detect_phases(rows, people, frames, 20, FPS)
    assert list(phases.phase) == ["presentation", "qa"]
    events, _ = detect_events(rows, people, frames, EventsConfig(), FPS, "v")
    assert events.empty  # nobody moved


@pytest.mark.parametrize("bad", [{"params": {"nod_amp": 1}}, {"gestures": {"wave": ["panel"]}},
                                 {"gestures": {"nod": ["teacher"]}}])
def test_events_config_rejects_typos(bad):
    with pytest.raises((KeyError, ValueError)):
        EventsConfig(**bad).validate()
