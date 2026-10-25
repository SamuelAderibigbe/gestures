# People and roles

## What counts as a "person"

The tracker gives each detected body an id. Ids break when someone is hidden for a moment, and
the detector occasionally draws two boxes on one body. The `people` stage cleans this up **[done]**:

1. **Joining broken tracks.** With a fixed camera most people stay where they are. A new track
   that starts after an old one ended, close to where it ended (within `merge_max_dist` × the
   person's height) and within `merge_max_gap_s`, is the same person. Two tracks visible at the
   same time are never joined.
2. **Removing duplicate boxes.** Two ids whose heads are in the same spot in most frames where
   both are visible are one person; the better-detected box is kept.
3. **Dropping noise.** Tracks shorter than `min_detections` frames are discarded.

A person is **one continuous presence in one place**. If the presenter walks from the podium to
the panel table, they appear as two people (presenter, then panel). Linking them would need face
recognition, which 25-pixel faces do not allow. This is a known limitation; per-role results are
unaffected.

## Roles

Each person gets one role, decided from how they appear over the whole video **and how they
relate to the other people in the scene**. The reason is written to `people.csv`
(`role_reason`) so every decision can be checked.

| Role | Rule (checked in this order) | In TWSE footage |
|---|---|---|
| transient | visible less than `min_presence_s` (10 s) | people walking past |
| on_screen | entirely inside the projection screen area **and** either tiny (shoulders < 0.6% of frame width) or much smaller (< 0.45×) than the typical person facing the camera outside the screen area. Being inside the screen area alone is never enough: presenters at a central podium and panels at a table below the screen are inside it too | people in photos on slides |
| audience | seen from behind ≥ 40% of the time and face < 30%, **or** cut off by the bottom of the frame with no face | investors/analysts in the front rows |
| other | standing within `edge_margin` (8%) of the left/right edge | TWSE staff, moderators |
| presenter | standing ≥ 80% of the time (any facing direction except from behind) | speaker at the podium, also when turned to the slides |
| panel | facing the camera and **seated in a row** with at least one other person facing the camera | executives in the Q&A |
| presenter | facing the camera **on their own** (not part of a row) | speaker behind a podium that hides the legs |
| other | none of the above | unclear cases |

"In a row" means another facing person whose head is at the same height (within one shoulder
width) and whose shoulders are of similar size (0.6–1.6×). This is what separates a seated
panel from a presenter when a table or podium hides the legs.

Cues are computed from the 17 body keypoints:

* **facing camera** – nose and both eyes detected with confidence ≥ 0.5
* **seen from behind** – both shoulders detected but no nose or eyes
* **standing** – thigh length ≥ 0.6 × torso length; if knees are hidden, box height ≥ 2.6 × width
* **microphone at mouth** – a forearm points up (wrist above elbow), the wrist is in front of the
  face, between chin and chest height, and the face is visible
* **projection screen** – found automatically: the largest very bright (> 230 of 255) area that
  stays in the same place across 9 frames spread over the video, between 8% and 60% of the
  frame. Saved to `scene.json`. Set `people.detect_screen: false` to turn it off.

### Checked on all six downloaded videos

Two 40-second clips per video (from the presentation at 30% and from the Q&A at 85% of each
video) plus two Chun Yuan clips, 12 clips in total, checked by eye in the debug videos:

| Video | Presentation clip | Q&A clip |
|---|---|---|
| 春源 2010 | presenter, 6 audience, 1 staff ✔ | 4 panel, 3 audience, 2 staff ✔ |
| 華夏 1305 (wide shot) | presenter (turned to the slides), 2 audience ✔ | standing moderator with mic → presenter, 2 panel ✔ |
| 新纖 1409 | presenter, 5 audience, **4 people in a slide photo → on_screen** ✔ | 4 panel, 2 audience, 1 staff ✔ |
| 嘉里大榮 2608 | presenter behind podium (legs hidden), 2 audience ✔ | same ✔ |
| 永新-KY 4557 (different venue) | 1 audience (presenter off camera) ✔ | 3 panel ✔ |
| 富爾特 6136 | presenter, 2 audience ✔ | same ✔ |

Small, distant audience members at the back of a wide shot are often not detected at all.
Results for the audience are therefore about the front rows.

Errors found and fixed during this check: a presenter in profile labelled "other"; a presenter
behind a podium labelled "panel"; a seated panelist labelled "presenter"; people in a slide photo
counted as real people.

## Overriding roles for a known venue

If a venue is always filmed the same way, draw fixed zones in `config.yaml`. A person whose median
position is inside a zone gets that role, overriding the rules:

```yaml
people:
  zones:
    - {role: panel,    box: [0.25, 0.50, 0.75, 0.75]}   # x1, y1, x2, y2 as fractions of the frame
    - {role: audience, box: [0.00, 0.75, 1.00, 1.00]}
```

Tip: open a frame in Preview, read pixel positions, divide by width/height.

## Changing role rules

| Setting | Default | Effect |
|---|---|---|
| people.min_presence_s | 10 | shorter appearances become `transient` |
| people.edge_margin | 0.08 | standing people this close to the side become `other` |
| people.detect_screen | true | find the projection screen and mark pictures on it `on_screen` |

## Renaming roles for other kinds of video

The role names fit conferences. For a class or meeting, rename them in the output:

```yaml
people:
  labels: {presenter: teacher, panel: student, audience: student_back}
```
