# Talk time

Status: **[done]**. For every person: seconds spent talking, number of speaking turns, and talk
time relative to the time they were on screen. Sheet **Talk time** in `results.xlsx`; per video
`speech.csv` (when anyone talks), `talk.csv` (who, turn by turn) and `talk_summary.csv`.

## How it works

1. **When is anyone talking?** Speech detection (Silero VAD) on the audio track. It works for
   Mandarin and English, ignores most background noise, and takes about 1% of the video's length
   (~2 minutes for the six TWSE videos). The model is part of the installed package; nothing is
   downloaded.
2. **Who is talking?** For every 1/8 second of speech, in this order:

| Method | Rule |
|---|---|
| `mic` | Someone is holding a microphone to their mouth (the `holding_mic` gesture). If several people hold one, the clearest one |
| `presenter` | During the presentation part, a presenter is on screen (podium microphones aren't hand-held) |
| `unattributed` | Otherwise. Reported as its own row, **never guessed** |

3. **Turns.** Consecutive speech by the same person, with pauses under 1 s, is one turn. A turn's
   `start_s`/`end_s` include short pauses; `duration_s` and `talk_s` count speech only.

## Reading the results

`Videos` sheet → `speech_attributed_pct` tells you how much of each video's speech was assigned
to a person. Check it before comparing talk time between videos.

Checked on 12 TWSE clips (2 per video):

| Situation | Result |
|---|---|
| Presenter at a podium (5 videos) | 99–100% attributed to the presenter |
| Q&A with a hand-held microphone (春源, 華夏, 永新) | 100% attributed, to the person holding it |
| Q&A with **table-stand microphones** (新纖) | 16% attributed; the rest `unattributed` |
| Presenter off camera (永新 presentation) | 0% attributed: correct, nobody visible is speaking |

## Limitations and possible extensions

* **Table-stand microphones.** Panelists speaking into a fixed microphone give no visible cue
  at this resolution (faces ~25 px, so no lip movement). That speech stays unattributed.
  Possible extension: speaker diarization (grouping speech by voice, e.g. pyannote) so
  unattributed speech is at least split into "voice A / voice B", linked to a person whenever that
  voice is once heard with a visible microphone.
* Audience questions asked off camera, or by people seen from behind without a hand-held mic,
  are unattributed.
* Overlapping speech is credited to one person at a time.

## Settings (`config.yaml` → `speech:`)

| Key | Default | Meaning |
|---|---|---|
| enabled | true | Run speech detection |
| threshold | 0.5 | Speech probability needed (higher = stricter) |
| min_speech_ms | 250 | Shorter sounds are ignored |
| min_silence_ms | 300 | Shorter pauses don't split speech |
| merge_gap_s | 1.0 | Same speaker again within this gap = same turn |
| min_turn_s | 0.5 | Shorter attributed pieces are dropped |
| presenter_fallback | true | Credit unclaimed speech to the visible presenter during the presentation |
