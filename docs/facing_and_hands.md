# Facing / turning away, and hand gestures

Two measures added for the "small sample test" requested in October 2026:

1. **Facing vs. turning away:** how much of the time each presenter / panelist faces the
   audience (the camera), and how often and how long they turn away or look down.
2. **Hand gestures (experimental):** `pointing` and `open_palm` hand shapes.

## Output

| File | One row per | Contents |
|---|---|---|
| `outputs/facing_and_hands.csv` (and sheet **Facing & hands** in `results.xlsx`) | presenter / panelist per video | `visible_min`, `facing_s`, `facing_share`, `turned_away_n`, `turned_away_s`, `turned_away_share`, `head_down_n`, `head_down_s`, `head_down_share`, `pointing_n`, `pointing_per_min`, `open_palm_n`, `open_palm_per_min`, `hands_tracked` |
| `outputs/timeline_all.csv` (per video: `timeline.csv`) | presenter / panelist per **second** | `state` = facing / turned_away / head_down / not_visible / unclear, plus `pointing` and `open_palm` (1 if seen that second) |

Use the **shares** (fraction of visible time) to compare people and videos, not the counts:
facing is a continuous state that the program reports in pieces, so `facing_audience` "event"
counts mostly reflect how often the stretch was interrupted.

## How they are measured

* **facing_audience:** face visible (nose + both eyes), not turned away and not head-down,
  for at least 1 s. Only for people whose face is usually visible.
* **turned_away** (`look_away` in events.csv): head turned more than 0.3 shoulder-widths
  from the person's usual direction, or face lost while the shoulders stay visible, for at least 2 s
  (e.g. turning to the slides).
* **head_down:** nose-to-shoulder height below 60% of usual for at least 3 s (reading notes).
* **pointing / open_palm:** see below.

In the per-second timeline a second is `turned_away` / `head_down` / `facing` if that state
covers at least half of it (checked in that order), `not_visible` if the person wasn't detected,
otherwise `unclear`.

## Hand gestures (experimental)

Turned on with `--detect-hands`. For every person and each wrist found by the body model, a
crop around the hand is enlarged to 224 px and given to Google MediaPipe's Gesture Recognizer,
which returns 21 hand points and a label; finger geometry is used as a second opinion. Only the
hand whose wrist matches the body's wrist is kept, so a neighbour's hand in the crop isn't counted.

Setup, once:

```bash
pip install -e ".[hands]"            # installs MediaPipe
# download the model (8 MB) into models/:
curl -L -o models/gesture_recognizer.task \
  https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
gestures run data/raw/ --detect-hands
```

**Reliability warning.** In the TWSE wide shot a hand is about 15–25 pixels. MediaPipe was made
for close-up hands, so many readings are noise (on 嘉里大榮, "pinch", a shape nobody makes at a
podium, was the most frequent raw label). Treat pointing / open-palm numbers as indicative only,
until checked against hand-coded clips. Hand tracking also roughly halves processing speed.
