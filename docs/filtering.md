# Filtering

"Filtering" happens at two levels.

## 1. Automatic noise filtering (built in)

| Filter | Where | Default | Status |
|---|---|---|---|
| Low-confidence people dropped | extract | person confidence < 0.3 | [done] |
| Ignore zones (e.g. a TV showing people) | extract | none | [done] |
| Very short tracks discarded | people | < 3 frames | [done] |
| Broken tracks joined, duplicate boxes merged | people | see [roles.md](roles.md) | [done] |
| Brief appearances marked `transient` | people | < 10 s visible | [done] |
| Pictures of people on slides marked `on_screen` | people | see roles.md | [done] |
| Stricter hand rules for people seen from behind (guessed hands) | events | see gestures.md | [done] |
| Low-confidence keypoints ignored | events | < 0.4 | [done] |
| Signals smoothed before rules apply | events | ~0.4 s window | [done] |
| Thresholds scaled to each person's keypoint jitter (nods, shakes) | events | 4 × jitter | [done] |
| Events shorter than the gesture's minimum duration dropped | events | per gesture | [done] |
| Same gesture repeated within 0.5 s merged | events | 0.5 s | [done] |

## 2. Filtering results for analysis [done]

Every event row carries `video_id, person_id, role, phase, gesture, start_s, end_s, duration_s,
confidence`, so any filter is a simple column filter in Excel or pandas. The export also has
built-in options:

```bash
gestures export --roles presenter,panel            # only these roles
gestures export --gestures nod,head_shake          # only these gestures
gestures export --min-confidence 0.6               # only clear events
gestures export --phase qa                         # only the Q&A part of each video
gestures export --exclude-roles transient,other,on_screen   # default
gestures export --exclude-roles ""                 # keep everyone
gestures export --roles panel --min-confidence 0.6 --xlsx outputs/panel_clear.xlsx   # combine
```

The workbook contains (every sheet has filter buttons and a frozen header row):

* **Summary** – per video × person × gesture: count, total duration, minutes visible, rate per
  minute, share of visible time. Gestures that apply to a person's role but never happened are listed with **0**, so
  averages are not biased by missing rows.
* **By role** – per video × role × gesture: number of people, events, mean rate per minute, mean
  share of time
* **Events** – every event with person, role, phase, start/end, confidence
* **People** – everyone detected (before filtering), with role and the reason for it
* **Phases** – presentation / Q&A segments per video
* **Videos** – metadata and processing status
* **Filters** – exactly which filters produced this workbook, and when (for reproducibility)

With `--phase`, rates use only the minutes each person was visible in that phase.

Rates are given per minute visible as well as raw counts, so people who are on screen longer are
not over-counted.

## Phases

TWSE videos have a presentation part (one standing presenter) and a Q&A part (seated panel).
The phase is detected automatically from who is visible: a presenter on screen → `presentation`,
panel on screen without a presenter → `qa`. Videos without this structure get `phase = all`.
