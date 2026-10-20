# How the pipeline works

```
video ─▶ probe ─▶ audio ─▶ speech ─▶ extract ─▶ people ─▶ events ─▶ talk ─▶ export
                              │          │          │
                     landmarks/      people.csv   events.csv / summary.csv
                     (slow, once)    (roles)      (gestures)       └─▶ results.xlsx
                                                   └─▶ debug.mp4 (optional, for checking)
```

| Stage | Status | What it does | Output (in `outputs/<video>/`) |
|---|---|---|---|
| probe | [done] | Reads length, resolution, frame rate, audio format | `metadata.json` |
| audio | [done] | Extracts 16 kHz mono audio for speech analysis | `audio.wav` |
| extract | [done] | Samples 8 frames/s; YOLO26-pose finds and tracks every person and gives 17 body keypoints each | `landmarks/pose/*.parquet`, `frames.parquet` |
| people | [done] | Joins broken tracks, removes duplicate boxes, assigns a role to each person | `people.csv`, `person_map.json` |
| events | [done] | Applies the gesture rules in [gestures.md](gestures.md); splits the video into presentation / Q&A | `events.csv`, `summary.csv`, `phases.csv`, `visibility.csv` |
| speech | [done] | Speech detection on the audio (Silero VAD) | `speech.csv` |
| talk | [done] | Who is talking: microphone holder / presenter / unattributed ([talk_time.md](talk_time.md)) | `talk.csv`, `talk_summary.csv` |
| debug | [done] | Draws boxes, ids, roles, skeletons and detected gestures on the video | `debug.mp4` |
| export | [done] | Combines all videos into one workbook | `outputs/results.xlsx` |

## Design principles

* **Expensive work runs once.** Video decoding and the neural network (the `extract` stage) take
  most of the time. Their results are saved, so every later stage, including gesture rules,
  can be changed and re-run in seconds.
* **Resumable and safe.** Each video has a `status.json`. A stage is skipped when its inputs,
  its settings and the program version are unchanged; otherwise it re-runs automatically.
  Files are written to a temporary name and renamed at the end, so a crash never leaves a
  half-written result that looks complete. One broken video never stops a batch.
* **Every number is traceable.** Each event has a person id, a time and a confidence, and
  `debug.mp4` shows exactly what the program saw at that moment.
* **Rules, not black boxes.** Gestures are defined as explicit, documented rules on body
  keypoints, so a reviewer can read exactly what "a nod" means and challenge it.

## Why these tools

| Need | Tool | Why |
|---|---|---|
| Find, track and pose people | Ultralytics YOLO26-pose | One model does detection, tracking and keypoints; works on small, distant people in wide shots; runs on CPU or the Mac GPU |
| Read video / audio | FFmpeg, OpenCV | Standard, fast, handles every format |
| Store landmarks | Parquet (pyarrow) | Compact, fast to reload, opens in pandas |
| Speech detection | Silero VAD | Accurate, multilingual (Mandarin), CPU-only, model bundled |
| Results | pandas, openpyxl | CSV + Excel for analysis |

MediaPipe (face mesh, hand landmarks) was evaluated but not used for the Taiwan footage: faces
there are ~25 px wide, below what face-mesh models handle. It can be added as an extra
extractor for close-up video (see [custom_videos.md](custom_videos.md)).

## Speed

Measured on an M4 Pro MacBook (GPU via `mps`): 2 minutes of 720p video in 22 seconds,
i.e. ~6× faster than real time. A 36-minute conference takes ~7 minutes; 100 hours of video
~17 hours. (On a 2-core cloud CPU it is ~0.25 s per frame, ~2× slower than real time.)
Everything after `extract` takes seconds. Ways to go faster: `sampling.fps: 5`, or
`extract.imgsz: 960` (less accurate for small people).

## Output files

`people.csv` — one row per person per video:

| Column | Meaning |
|---|---|
| person_id | 1, 2, 3 … in order of first appearance (unique within a video) |
| role | presenter / panel / audience / other / transient (see [roles.md](roles.md)) |
| role_reason | plain-language reason the role was chosen |
| first_seen_s, last_seen_s, visible_s | when and how long the person was visible |
| face_ratio | share of frames with face (nose + both eyes) visible = facing camera |
| back_ratio | share of frames seen from behind |
| standing_ratio | share of frames standing |
| mic_ratio | share of frames holding a microphone to the mouth |
| center_x, center_y, height | median position and size (0-1 of frame) |
| track_ids | raw tracker ids merged into this person |

`events.csv` — one row per detected gesture: `video_id, person_id, role, phase, gesture,
start_s, end_s, duration_s, confidence`.

`summary.csv` — per person × gesture: count, total duration, minutes visible, rate per minute
(zeros included).

`phases.csv` — presentation / qa segments with start and end times.

`visibility.csv` — seconds each person was visible in each phase.

`outputs/results.xlsx` — sheets **Videos**, **Summary**, **Events** for all processed videos.
