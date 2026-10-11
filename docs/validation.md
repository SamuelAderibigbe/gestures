# Accuracy check (validation)

**Why:** until results are compared with what a person sees, "the program found 40 nods" is an
unverified claim. The check produces numbers like *"found 9 of 10 hand raises (recall 0.9); 11 of
its 12 nods were real (precision 0.92)"*, which is what a reviewer will ask for.

**What a person does:** watch a few short clips and write down each gesture they see (who, what,
when) in a spreadsheet. The program then compares. No special software and no programming needed.

## Steps

**1. Prepare** (after the videos have been processed; takes a minute or two):

```bash
gestures validate-prepare                 # 10 clips of 2 minutes, picked at random
gestures validate-prepare --clips 6 --length 90 --seed 2     # other choices
```

This creates `validation/`:

| File | What it is |
|---|---|
| `clips/C01.mp4` ... | the clips, full frame rate, everyone boxed with their id (P1, P2 ...) and a clock. **No gesture labels**, so the coder isn't influenced |
| `coding_sheet.xlsx` | where gestures are written down: drop-down lists for clip and gesture, plus a sheet listing who is in each clip |
| `INSTRUCTIONS.md` | plain-language instructions for the coder, with the definition of every gesture |
| `clips.csv` | which video and time range each clip is |

Clips are spread over all videos and over the presentation and Q&A parts, at random positions,
chosen **without looking at the program's results** so the check is not biased to easy moments.

**2. Code** (a person, ~15–20 minutes per 2-minute clip). Each coder saves a copy named
`coding_sheet_<name>.xlsx` and adds one row per gesture: clip, person number, gesture, start and
end time read off the clock (e.g. `25:12`). Coders can limit themselves to some gesture types and
tick them in the `coded_gestures` sheet; types they didn't look for aren't counted as misses.
`speaking` can also be coded to check talk time.

**3. Compare:**

```bash
gestures validate
```

Prints a table per gesture and writes `validation/accuracy_report.xlsx`:

| Sheet | Contents |
|---|---|
| Accuracy | per gesture: marked by person, found by program, matched, **precision, recall, F1** |
| Per clip | the same per clip |
| Mismatches | every miss and every extra, with person and time, to look up in the clip and tune rules |
| Coder agreement | when 2+ people coded the same clips: how much they agree (agreement F1) |
| Annotations | everything that was coded |

**Matching rule:** a program event and a marked gesture match if they are the same person and
gesture and their times overlap, allowing 1 s slack on each side (`--tolerance`). Each can match
only once.

**What the numbers mean:**

* **Recall:** of the gestures the person saw, the share the program found. Low = misses things.
* **Precision:** of the program's events, the share the person agrees with. Low = false alarms.
* **Coder agreement:** if two people agree only 70% on nods, the program can't be expected to
  do better than that; it shows how clear-cut a gesture is.

**Recommended amount:** 10 clips × 2 minutes (~3–4 hours of coding), with 3–4 of the clips
coded by a second person for agreement. Use half the clips to adjust thresholds if needed and
report accuracy on the other half, so the numbers aren't flattered by tuning.

## Development checks done so far

| Date | What | Result |
|---|---|---|
| 2026-10-11 | Unit and end-to-end tests (synthetic videos, keypoints, demo, validation loop) | 81/81 pass |
| 2026-10-11 | Microphone cue, Chun Yuan Q&A | Speaking panelist 100%, other three 0% |
| 2026-10-11 | Roles on 12 clips from all 6 videos (see [roles.md](roles.md)) | all correct after 4 fixes |
| 2026-10-11 | Head keypoint jitter on real panelists / presenter | 1–4% / 7% of head height per frame; nod threshold made jitter-adaptive |
| 2026-10-11 | Gesture events spot-checked in debug videos | false stand_up, false nods (glance down / writing) and false back-view hand events found and fixed; see [gestures.md](gestures.md) |
| 2026-10-11 | Talk time on the same 12 clips | see [talk_time.md](talk_time.md) |

These are development checks by eye. The formal check above has not been done yet.
