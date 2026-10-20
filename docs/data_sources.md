# Data sources

## TWSE WebPro investor conferences (法人說明會)

Source: Taiwan Stock Exchange WebPro video portal, category 101
(`https://webpro.twse.com.tw/WebPortal/vod/101/...`). Each page has a video id; the stream
address is derived from it, so downloading needs no browser.

```bash
brew install yt-dlp ffmpeg                                   # once
python3 scripts/download_webpro.py --list data/webpro_links.txt
```

* Files are saved as `data/raw/<YYYYMMDD>_<stock code>_<company>.mp4` (ROC year 115 = 2026).
* `data/webpro_manifest.csv` records every video: id, date, company, page, stream, status.
* Already-downloaded videos are skipped; interrupted downloads resume when re-run.

Observed so far (Chun Yuan 2010, 2026-10-08): 1280×720, 29.97 fps, AAC audio, ~36 min, one fixed
wide camera in the TWSE auditorium: presentation at a podium, then a seated Q&A panel, audience
seen from behind, stock ticker across the top.

**Permissions:** confirm with the project owner that use of these recordings for research is
permitted under TWSE's terms, and whether ethics approval is required.

## AMI Meeting Corpus (reference data)

`data/reference/amicorpus/` – meeting ES2016a, close-up videos and headset mix. Licence CC BY 4.0
(https://groups.inf.ed.ac.uk/ami/corpus/). Useful for close-up head movement and, later, for
validating against AMI's own gesture annotations.
