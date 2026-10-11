# Validation

Automatic counts are only useful if we know how accurate they are. Plan **[planned]**:

1. **Pick clips.** 10–15 clips of 2 minutes each from different videos, covering presentation
   and Q&A, chosen before looking at the program's output.
2. **Hand-code them in ELAN** (free, https://archive.mpi.nl/tla/elan). One tier per person, one
   annotation per gesture with start/end, using the definitions in [gestures.md](gestures.md).
   Ideally two coders code a subset independently to check the definitions are clear
   (agreement, Cohen's kappa).
3. **Compare automatically.** A `gestures validate` command will read the ELAN `.eaf` exports and
   match them with `events.csv` (an event counts as found if it overlaps a hand-coded one).
4. **Report per gesture:** precision (of the events the program found, how many are real),
   recall (of the real ones, how many it found), and count error per person.
5. **Tune thresholds** on half of the clips, report final accuracy on the other half, so the
   reported numbers are not flattered by tuning.

Also checked: role assignment accuracy (compare `people.csv` with what a human sees in
`debug.mp4`) and identity stability (how often one person's id changes).

## Checks done so far

| Date | What | Result |
|---|---|---|
| 2026-10-11 | Unit tests (synthetic videos and keypoints) | 26/26 pass |
| 2026-10-11 | Chun Yuan, 30 s presentation + 30 s Q&A: roles and ids by eye in debug video | All visible people correctly labelled; ids stable across both clips; 2 duplicate boxes correctly merged per clip |
| 2026-10-11 | Microphone cue, Chun Yuan Q&A | Speaking panelist 100%, other three 0% |
| 2026-10-11 | Unit tests incl. one per gesture with realistic jitter, and "still person = no events" | 53/53 pass |
| 2026-10-11 | Roles on 12 clips from all 6 videos (see [roles.md](roles.md)) | all correct after 4 fixes |
| 2026-10-11 | Head keypoint jitter on real panelists / presenter | 1–4% / 7% of head height per frame; nod threshold made jitter-adaptive |
| 2026-10-11 | Gesture events spot-checked in debug videos | false stand_up, false nods (glance down / writing) and false back-view hand events found and fixed; see [gestures.md](gestures.md) |

These are development checks by eye, not the formal accuracy measurement described above.
