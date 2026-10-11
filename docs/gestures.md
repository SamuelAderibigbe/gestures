# Gesture definitions

Status: **[done]**, implemented in `gesture_pipeline/gestures.py` and tested on synthetic
movements (one test per gesture, plus "sitting still produces nothing") and on real TWSE clips.
Thresholds are calibrated on real footage so far by eye; they will be confirmed against
hand-coded clips (see [validation.md](validation.md)). All thresholds can be changed in
`config.yaml` under `events.params` without touching code.

## Ground rules

* Gestures are computed from the 17 body keypoints of each person, sampled 8 times per second.
  Keypoints below confidence 0.4 are ignored. Signals are smoothed over ~0.4 s before rules apply.
* Sizes are measured relative to the person's own body (shoulder width, nose-to-shoulder height),
  so rules work the same for someone near or far from the camera.
* Every event has a start, end and **confidence** (0-1: how clearly the rule was met × how
  confidently the keypoints were seen). Two events of the same gesture by the same person closer
  than 0.5 s are merged.
* A gesture is only evaluated when the body parts it needs are visible. E.g. nods are not
  evaluated for audience members seen from behind; their face is not visible.

## Gestures

S = the person's shoulder width, H = their nose-to-shoulder height (medians over the video).

| Gesture | Who | Rule | Min. duration |
|---|---|---|---|
| **hand_raise** | all | A wrist above the top of the head (above the nose by > 0.5 H; from behind, > 1 S above the shoulders) with the elbow lifted to shoulder level | 1.0 s |
| **hand_above_shoulder** | all | A wrist above the shoulder line by > 0.25 S, but not a hand raise and not holding a mic (touching the head, pointing, reaching, holding a phone up to photograph a slide). **Seen from behind:** wrist > 0.4 S above the shoulder and elbow no lower than 0.6 S below it | 0.5 s |
| **holding_mic** | all | Forearm pointing up (wrist above elbow), wrist in front of the face, between chin and chest; face visible. Pauses < 2 s do not split the event. A turn-of-talk cue | 1.0 s |
| **nod** | presenter, panel | The head dips relative to the shoulders and comes back up. Dip depth ≥ 0.15 H **and** ≥ 4 × the person's own keypoint jitter; dip width ≤ 0.8 s at half depth. Dips < 1 s apart form one nod event | – |
| **head_shake** | presenter, panel | Head swings side to side relative to the shoulders, each swing ≥ 0.10 S and ≥ 4 × jitter, ≥ 2 direction changes, ≤ 0.75 s between extremes | – |
| **head_down** | presenter, panel | Nose-to-shoulder height < 0.6 × usual (reading notes, looking at a phone, writing) | 3.0 s |
| **look_away** | presenter, panel | Head turned > 0.3 S from the person's usual direction, or face no longer visible while the shoulders still are (only for people whose face is normally visible) | 2.0 s |
| **gesturing** | presenter, panel | A hand above the waist moving faster than 0.8 S per second (not during a hand raise) | 0.5 s |
| **stand_up** | audience | Shoulders rise ≥ 0.8 S above where they were in the preceding 2 s, and stay up; the event lasts until they sit back down | 1.0 s |

All rules first ignore keypoints with confidence < 0.4, remove single-frame flicker, and merge
repeats of the same gesture less than 0.5 s apart.

### Decisions made while calibrating on real footage

* **Nods adapt to each person's jitter.** On panelists (~50 px shoulders) head keypoints jitter
  by 1–4 % of H between frames; on the distant presenter by up to 7 %. A fixed threshold would
  produce false nods for small people, so a nod must also exceed 4 × that person's own measured
  jitter (robust estimate from frame-to-frame differences).
* **A nod is quick.** Real dips from Chun Yuan were 0.6–1.5 s wide. The slower ones were the
  presenter glancing down at the podium and a panelist bobbing while writing, not nods. Dips wider
  than 0.8 s at half depth are therefore not nods (long ones are caught by head_down).
* **Standing up uses the shoulders only.** Audience members are seen from behind with legs hidden
  by chairs; a rule based on box shape wrongly flagged a seated person. Shoulder rise is visible
  from any angle.
* **Hidden hands get guessed.** For audience members seen from behind, the pose model guesses
  where hidden hands are, often just above the shoulder. On 永新 this produced 5 false
  hand_above_shoulder events in 40 s for a man sitting still. Real cases checked on 新纖 and
  嘉里大榮 (photographing a slide with a phone, hand to face/head) all had the wrist ≥ 0.45 S above
  the shoulder; the false ones 0.27 S with the elbow hanging. Hence the stricter rule from behind.
  Trade-off: a light touch of the ear from behind is no longer counted.
* **Ongoing behaviours are measured as time, not counts.** Holding a mic, gesturing and head_down
  are reported with `share_of_time` (fraction of the person's visible time) in the Summary; brief
  keypoint dropouts would otherwise split one continuous mic hold into several "events".
* **Hand-to-head counts as hand_above_shoulder, not hand_raise.** Two audience members touching
  their head were correctly reported as hand_above_shoulder; a hand raise needs the hand above the
  head with the elbow up.

Per-person totals are also reported without being "gestures": time visible (screen time), time
facing the camera, and later talk time.

## Why these, and what they are not

* **Hand raise** is the classic bid for a turn (asking a question). It was named in the project
  brief: a hand that rises above the shoulder probably means something. We count the
  above-head raise as the main event and the weaker above-shoulder movement separately.
* **Nods and head shakes** are the main listener feedback signals in conversation
  ("backchannels", Yngve 1970). In a Q&A they show agreement or disagreement with what is said.
* **Gesturing while speaking** follows McNeill (1992, *Hand and Mind*) and Kendon (2004,
  *Gesture: Visible Action as Utterance*): hand movements produced with speech. At this resolution
  we can measure *how much* someone gestures, not the gesture type (iconic, beat, etc.).
* **Head down / look away** are observable attention cues. We deliberately do **not** label
  anyone "bored" or "uninterested": that is an interpretation, left to the analysis.
* **Not measurable in the Taiwan footage:** facial expressions (smiles, frowns), gaze direction,
  and finger-level hand shapes (thumbs up, pointing). Faces are ~25 px wide and hands smaller.
  These become possible on close-up video (see [custom_videos.md](custom_videos.md)).
