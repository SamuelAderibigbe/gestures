"""Role and identity logic, on synthetic keypoints (no model needed)."""

import numpy as np
import pandas as pd

from gesture_pipeline.config import PeopleConfig
from gesture_pipeline.people import build_people, merge_tracklets


def _row(frame, tid, cx, cy, h, face=True, standing=False, mic=False):
    """A plausible COCO-17 pose for a person centred at (cx, cy) with box height h."""
    top = cy - h / 2
    sh_y, hip_y = top + 0.25 * h, top + 0.55 * h
    knee_y = hip_y + (0.25 * h if standing else 0.03 * h)
    x = [cx] * 17
    y = [top + 0.1 * h] * 5 + [sh_y] * 2 + [sh_y + 0.15 * h] * 2 + [sh_y + 0.3 * h] * 2 + \
        [hip_y] * 2 + [knee_y] * 2 + [knee_y + 0.2 * h] * 2
    x[5], x[6] = cx - 0.1 * h, cx + 0.1 * h
    if mic:  # forearm up, wrist at the chin
        x[7], y[7] = cx - 0.05 * h, sh_y + 0.12 * h
        x[9], y[9] = cx - 0.03 * h, top + 0.14 * h
    v = [0.9] * 17
    if not face:
        v[0] = v[1] = v[2] = 0.05
    w = 0.4 * h
    return {"frame": frame, "t": frame / 8, "person_id": tid, "score": 0.9,
            "bbox": [cx - w / 2, top, cx + w / 2, top + h], "x": x, "y": y, "visibility": v}


def _video(seconds=60):
    rows = []
    for f in range(seconds * 8):
        rows.append(_row(f, 1, 0.80, 0.45, 0.30, standing=True, mic=(f % 2 == 0)))  # presenter
        rows.append(_row(f, 2, 0.35, 0.60, 0.15))                                   # panel, in a row
        rows.append(_row(f, 5, 0.50, 0.60, 0.15))                                   # with this panelist
        rows.append(_row(f, 3 if f < 200 else 9, 0.30, 0.90, 0.25, face=False))     # audience, id switch
        if f < 30:
            rows.append(_row(f, 4, 0.10, 0.5, 0.3, standing=True))                  # walks past
    return pd.DataFrame(rows)


def test_roles_and_merge():
    people, mapping = build_people(_video(), PeopleConfig(), "v", 8)
    assert mapping[3] == mapping[9], "id switch should be merged back into one person"
    roles = dict(zip(people.person_id, people.role))
    assert roles[mapping[1]] == "presenter"
    assert roles[mapping[2]] == "panel"
    assert roles[mapping[3]] == "audience"
    assert roles[mapping[5]] == "panel"
    assert roles[mapping[4]] == "transient"
    assert len(people) == 5


def test_duplicate_box_merged_and_mic_detected():
    rows = []
    for f in range(160):
        rows.append(_row(f, 1, 0.5, 0.5, 0.2, mic=True))
        dup = _row(f, 2, 0.5, 0.5, 0.2)
        dup["bbox"] = [0.46, 0.4, 0.54, 0.75]  # same head, taller box
        dup["score"] = 0.5
        rows.append(dup)
    people, mapping = build_people(pd.DataFrame(rows), PeopleConfig(), "v", 8)
    assert len(people) == 1 and mapping[1] == mapping[2]
    assert people.iloc[0].mic_ratio > 0.9


def test_lone_seated_facing_person_is_presenter():
    rows = [_row(f, 1, 0.5, 0.5, 0.2) for f in range(160)]       # e.g. presenter behind a podium
    people, _ = build_people(pd.DataFrame(rows), PeopleConfig(), "v", 8)
    assert people.iloc[0].role_key == "presenter"


def test_presenter_in_profile_is_presenter():
    rows = []
    for f in range(160):
        r = _row(f, 1, 0.8, 0.45, 0.3, standing=True)
        r["visibility"][1] = r["visibility"][2] = 0.1            # eyes not both visible: side-on
        rows.append(r)
    people, _ = build_people(pd.DataFrame(rows), PeopleConfig(), "v", 8)
    assert people.iloc[0].role_key == "presenter"


def test_picture_on_slide_is_on_screen():
    rows = []
    for f in range(160):
        rows.append(_row(f, 1, 0.80, 0.55, 0.30, standing=True))   # real presenter
        rows.append(_row(f, 2, 0.30, 0.80, 0.25, face=False))       # audience
        rows.append(_row(f, 3, 0.50, 0.25, 0.08))                   # tiny figure on the slide
    screen = [0.2, 0.05, 0.75, 0.5]
    people, mapping = build_people(pd.DataFrame(rows), PeopleConfig(), "v", 8, screen=screen)
    roles = dict(zip(people.person_id, people.role_key))
    assert roles[mapping[3]] == "on_screen" and roles[mapping[1]] == "presenter"


def test_presenter_at_central_podium_inside_screen_area_is_kept():
    # 聲寶: presenter in front of the screen, lower body hidden by the podium, so the box is
    # entirely inside the screen area and does not reach below it. Must stay a presenter.
    rows = []
    for f in range(160):
        rows.append(_row(f, 1, 0.80, 0.35, 0.20))                    # facing, upper body only
        rows.append(_row(f, 2, 0.20, 0.90, 0.25, face=False))        # audience
    screen = [0.2, 0.05, 0.85, 0.6]
    people, mapping = build_people(pd.DataFrame(rows), PeopleConfig(), "v", 8, screen=screen)
    roles = dict(zip(people.person_id, people.role_key))
    assert roles[mapping[1]] == "presenter"


def test_panel_seated_below_screen_is_kept_while_slide_photo_is_not():
    # Q&A: panel at a table in front of the screen (inside the screen area), presenter
    # standing at a side podium outside it, and a small photo of a person on the slide.
    rows = []
    for f in range(160):
        rows.append(_row(f, 1, 0.92, 0.45, 0.30, standing=True))    # side presenter, outside screen
        for i, x in enumerate((0.35, 0.50, 0.65)):
            rows.append(_row(f, 10 + i, x, 0.42, 0.18))              # panel (0.6x presenter), inside screen
        rows.append(_row(f, 20, 0.45, 0.18, 0.05))                   # photo on the slide (tiny)
    screen = [0.2, 0.05, 0.85, 0.6]
    people, mapping = build_people(pd.DataFrame(rows), PeopleConfig(), "v", 8, screen=screen)
    roles = dict(zip(people.person_id, people.role_key))
    assert [roles[mapping[t]] for t in (10, 11, 12)] == ["panel"] * 3
    assert roles[mapping[20]] == "on_screen"


def test_staff_at_edge_is_other():
    rows = [_row(f, 1, 0.96, 0.5, 0.3, standing=True) for f in range(160)]
    for r in rows:
        r["bbox"][2] = 1.0
    people, _ = build_people(pd.DataFrame(rows), PeopleConfig(), "v", 8)
    assert people.iloc[0].role == "other"


def test_simultaneous_people_never_merged():
    df = pd.DataFrame([_row(f, t, 0.5 + 0.01 * t, 0.5, 0.3) for f in range(50) for t in (1, 2)])
    m = merge_tracklets(df, PeopleConfig())
    assert m[1] != m[2]


def test_zone_override_and_labels():
    cfg = PeopleConfig(zones=[{"role": "panel", "box": [0.7, 0.2, 0.9, 0.7]}],
                       labels={**PeopleConfig().labels, "panel": "participant"})
    people, mapping = build_people(_video(), cfg, "v", 8)
    roles = dict(zip(people.person_id, people.role))
    assert roles[mapping[1]] == "participant"  # presenter forced into zone, then relabelled


def test_empty():
    people, mapping = build_people(pd.DataFrame(), PeopleConfig(), "v", 8)
    assert people.empty and mapping == {}
