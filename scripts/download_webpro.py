#!/usr/bin/env python3
"""Download TWSE WebPro investor-conference videos (法人說明會).

Usage:
    python scripts/download_webpro.py URL [URL ...]
    python scripts/download_webpro.py --list links.txt          # one page URL per line
    python scripts/download_webpro.py --list links.txt --dry-run

Each WebPro page URL contains a video id, and the stream address can be built
from that id, so no browser is needed. Files are named
    data/raw/<YYYYMMDD>_<stock code>_<company>.mp4
(ROC year 115 -> 2026). A manifest CSV records every attempt; videos already
downloaded are skipped, and interrupted downloads resume when re-run.

Requires yt-dlp and ffmpeg (macOS: brew install yt-dlp ffmpeg).
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

ID_RE = re.compile(r"/vod/\d+/([0-9A-F]{12}-[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4})", re.I)
TITLE_RE = re.compile(r"<title>([^<]*)</title>", re.I)
# e.g. "115.09.10--富爾特(6136)法人說明會"
EVENT_RE = re.compile(r"(\d{2,3})\.(\d{1,2})\.(\d{1,2})\s*-+\s*(.+?)\((\w+)\)\s*(.*)$")
STREAM = "https://webprovod.twse.com.tw/vod/_definst_/{a}/{b}/{id}/{id}_ATM.mp4/playlist.m3u8"
MANIFEST_FIELDS = ["video_id", "date", "stock_code", "company", "event", "page_url",
                   "stream_url", "file", "status", "checked_at"]


def video_id(page_url: str) -> str:
    m = ID_RE.search(page_url)
    if not m:
        raise ValueError(f"Not a WebPro video page URL: {page_url}")
    return m.group(1).upper()


def stream_url(vid: str) -> str:
    tail = vid[-4:]
    return STREAM.format(a=tail[:2], b=tail[2:], id=vid)


def parse_title(title: str) -> dict:
    """'臺灣證券交易所｜WebPro 影音傳播網 - 115.09.10--富爾特(6136)法人說明會' -> fields."""
    name = html.unescape(title).split(" - ", 1)[-1].strip()
    m = EVENT_RE.search(name)
    if not m:
        return {"date": "", "stock_code": "", "company": "", "event": name}
    roc, mo, d, company, code, event = m.groups()
    year = int(roc) + 1911 if int(roc) < 1000 else int(roc)
    return {"date": f"{year:04d}{int(mo):02d}{int(d):02d}", "stock_code": code,
            "company": company.strip(), "event": event.strip() or "法人說明會"}


def fetch_title(page_url: str) -> str:
    req = urllib.request.Request(page_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8", "replace")
    m = TITLE_RE.search(text)
    return m.group(1) if m else ""


def safe(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", s).strip("_")


def output_name(info: dict, vid: str) -> str:
    if info["date"] and info["stock_code"]:
        return f"{info['date']}_{info['stock_code']}_{safe(info['company'])}.mp4"
    return f"webpro_{vid}.mp4"


def is_complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1_000_000:
        return False
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return True
    r = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() not in ("", "N/A")


def download(stream: str, out: Path) -> bool:
    cmd = ["yt-dlp", "--quiet", "--progress", "--newline",
           "--retries", "infinite", "--fragment-retries", "infinite",
           "--concurrent-fragments", "4", "--socket-timeout", "30",
           "-o", str(out), stream]
    return subprocess.run(cmd).returncode == 0


def load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return {row["video_id"]: row for row in csv.DictReader(fh)}


def save_manifest(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for row in rows.values():
            w.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", nargs="*", help="WebPro video page URLs")
    ap.add_argument("--list", type=Path, help="text file with one page URL per line")
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--manifest", type=Path, default=Path("data/webpro_manifest.csv"))
    ap.add_argument("--dry-run", action="store_true", help="show what would be downloaded")
    args = ap.parse_args(argv)

    urls = list(args.urls)
    if args.list:
        urls += [l.strip() for l in args.list.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.lstrip().startswith("#")]
    if not urls:
        ap.error("give at least one URL or --list")
    if not args.dry_run:
        for tool in ("yt-dlp", "ffmpeg"):
            if not shutil.which(tool):
                sys.exit(f"{tool} not found. Install with: brew install yt-dlp ffmpeg")

    manifest = load_manifest(args.manifest)
    failures = 0
    for i, url in enumerate(dict.fromkeys(urls), 1):
        vid = video_id(url)
        try:
            info = parse_title(fetch_title(url))
        except Exception as exc:  # page unreachable: still downloadable by id
            print(f"  (could not read page title: {exc})")
            info = {"date": "", "stock_code": "", "company": "", "event": ""}
        prev = manifest.get(vid, {})
        out = Path(prev["file"]) if prev.get("file") else args.out / output_name(info, vid)
        row = {"video_id": vid, **info, "page_url": url, "stream_url": stream_url(vid), "file": str(out)}
        print(f"[{i}/{len(urls)}] {info['date']} {info['stock_code']} {info['company']} -> {out}")

        if is_complete(out):
            row["status"] = "done"
            print("  already downloaded, skipping")
        elif args.dry_run:
            row["status"] = prev.get("status", "pending")
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            ok = download(row["stream_url"], out) and is_complete(out)
            row["status"] = "done" if ok else "failed"
            failures += not ok
            print("  OK" if ok else "  FAILED (re-run the same command to resume)")
        row["checked_at"] = datetime.now().isoformat(timespec="seconds")
        manifest[vid] = row
        save_manifest(args.manifest, manifest)
    print(f"Manifest: {args.manifest}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
