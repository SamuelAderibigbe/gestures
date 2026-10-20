# Using your own videos

The pipeline is not tied to the Taiwan footage. Any video file works: a lecture recording, a
meeting filmed with a phone, a Zoom/Teams recording.

```bash
gestures run path/to/my_video.mp4 --debug-video
```

Then open `outputs/my_video/debug.mp4` to check people and roles, and `people.csv` for the reasons.

## What works best

| Factor | Good | Problematic |
|---|---|---|
| Camera | fixed (tripod, wall camera, screen recording) | handheld, panning, zooming, frequent cuts |
| Person size | ≥ 80 px tall in the frame | tiny figures at the back of a hall |
| Resolution | 720p or more | below 480p |
| Lighting | even | strong backlight (window behind people) |
| View | people facing camera for nods/head shakes | only backs visible (only hand raises then) |

## Adapting to a new setting

1. **Run once with `--debug-video` and check roles.** The automatic rules use generic cues
   (facing camera, seated/standing, seen from behind), so they often work unchanged.
2. **Rename roles** if conference names don't fit, e.g. for a classroom:
   ```yaml
   people:
     labels: {presenter: teacher, panel: student, audience: student_back}
   ```
3. **Fix roles with zones** if the rules get a venue wrong (see [roles.md](roles.md)).
4. **Ignore zones** for areas that contain people who should not be counted (a TV screen, a
   poster with people on it):
   ```yaml
   extract:
     ignore_zones: [[0.80, 0.05, 0.99, 0.30]]
   ```
5. **Close-up video** (one or two people filling the frame, e.g. the AMI meeting corpus or a webcam
   recording) allows finer detail. A MediaPipe face/hand extractor can be added for smiles,
   gaze and finger gestures **[planned]**.

## Video-call recordings (gallery view)

A Zoom/Teams gallery recording shows each person in their own tile. The current pipeline treats
it as one picture with several people, which works for hand raises and head movement.
Splitting tiles and reading names from them **[planned]** would add per-tile names and
active-speaker detection from the highlighted tile border.

## Use your own config per project

Keep one config file per kind of video and pass it explicitly:

```bash
gestures run data/classroom/ --config configs/classroom.yaml --output outputs_classroom
```
