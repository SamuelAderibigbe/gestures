# Configuration reference

All settings live in `config.yaml`. Anything left out uses the default shown here. Misspelled
keys stop the program with an error instead of being silently ignored. Changing a setting
re-runs only the stages that depend on it.

## paths
| Key | Default | Meaning |
|---|---|---|
| output_dir | `outputs` | One sub-folder per video is created here |

## sampling
| Key | Default | Meaning |
|---|---|---|
| fps | 8 | Frames analysed per second. Gestures last 0.3–3 s; 5–10 is enough |
| max_width | 1280 | Wider frames are scaled down first |

## audio
| Key | Default | Meaning |
|---|---|---|
| extract | true | Extract audio for talk-time analysis |
| sample_rate | 16000 | Hz, mono |

## extract
| Key | Default | Meaning |
|---|---|---|
| models | `[pose]` | Extractors to run |
| model_dir | `models` | Where model weights are stored (downloaded automatically) |
| pose_model | `yolo26n-pose.pt` | `yolo26s-pose.pt` = more accurate, ~2.5× slower |
| imgsz | 1280 | Detector input size. Lower is faster but misses small people |
| conf | 0.3 | Minimum detection confidence |
| device | `auto` | `auto` picks the Mac GPU (`mps`) if available, else CPU |
| tracker | `conference` | Bundled ByteTrack settings for static cameras, or a path to your own YAML |
| ignore_zones | `[]` | Rectangles `[x1,y1,x2,y2]` (0–1) where people are ignored |
| detect_hands | false | Experimental hand shapes with MediaPipe (same as `--detect-hands`); needs `pip install -e ".[hands]"` and `models/gesture_recognizer.task`, see [facing_and_hands.md](facing_and_hands.md) |
| hand_model | `gesture_recognizer.task` | Hand model file inside `model_dir` |
| hand_conf | 0.2 | Minimum hand detection confidence |

## people
| Key | Default | Meaning |
|---|---|---|
| min_presence_s | 10 | Visible less than this → `transient` |
| min_detections | 3 | Shorter tracks are noise |
| merge_max_gap_s | 300 | Max time a person can disappear and still be re-joined |
| merge_max_dist | 0.6 | Max distance (× body height) for re-joining |
| face_conf | 0.5 | Keypoint confidence for "face visible" |
| edge_margin | 0.08 | Standing people this close to the left/right edge → `other` (staff) |
| detect_screen | true | Detect the projection screen; pictures of people on it → `on_screen` |
| zones | `[]` | Fixed roles by position, see [roles.md](roles.md) |
| labels | identity | Rename roles in the output |

## events
| Key | Default | Meaning |
|---|---|---|
| exclude_roles | `[transient, on_screen]` | Roles for which no gestures are computed |
| gestures | see [gestures.md](gestures.md) | Which gestures apply to which role, e.g. `nod: [presenter, panel]` |
| params | `{}` | Override any rule threshold, e.g. `{nod_min_amp: 0.2, hand_raise_min_s: 1.5}`. All names and defaults are in `DEFAULT_GESTURE_PARAMS` in `gesture_pipeline/config.py` |
| min_phase_s | 20 | Presentation/Q&A switches shorter than this are smoothed away |

Changing any of these re-runs only the events stage (seconds), not the video analysis.

## speech
See [talk_time.md](talk_time.md#settings-configyaml--speech) for `enabled, threshold, min_speech_ms,
min_silence_ms, merge_gap_s, min_turn_s, presenter_fallback`.

## batch
| Key | Default | Meaning |
|---|---|---|
| video_extensions | mp4, avi, mkv, mov, webm, ts | Files picked up from folders |
| continue_on_error | true | Keep going when one video fails |

## Command line

```
gestures doctor                          check installation
gestures probe <videos/folders>          show video details
gestures run <videos/folders> [options]  process
    --config FILE      --output DIR       --fps N
    --start SEC        --end SEC          (process only part of each video)
    --limit N          (first N videos)   --stages a,b,c   --force
    --debug-video      (also render debug.mp4)
    --detect-hands     (experimental hand shapes: pointing, open palm)
gestures demo [--open]                   synthetic conference, no video needed
gestures validate-prepare [--clips N --length S --seed K]   clips + coding sheet
gestures validate [--tolerance S]        accuracy report from filled-in coding sheets
gestures export [options]                rebuild results.xlsx
    --roles a,b   --exclude-roles a,b   --gestures a,b
    --min-confidence X   --phase presentation|qa   --xlsx FILE
python scripts/download_webpro.py --list data/webpro_links.txt
```
