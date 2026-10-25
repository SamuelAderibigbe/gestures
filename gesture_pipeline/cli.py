"""Command line entry point.

    gestures doctor                     check the installation
    gestures probe VIDEO_OR_DIR ...     show video metadata
    gestures run VIDEO_OR_DIR ...       process videos (resumes where it stopped)
    gestures export                     combine results into one Excel workbook
    gestures demo                       try it on a synthetic conference (no video needed)
    gestures validate-prepare           make clips + a coding sheet for an accuracy check
    gestures validate                   compare people's coding with the program
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path

from . import __version__
from .config import load_config
from .export import ExportFilters, export_batch
from .ingest import probe
from .log import setup_logging
from .pipeline import DEFAULT_STAGES, STAGES, discover_videos, run_batch

log = logging.getLogger("gesture_pipeline")


def _default_config() -> str | None:
    return "config.yaml" if Path("config.yaml").exists() else None


def cmd_doctor(_args) -> int:
    ok = True
    print(f"gesture-pipeline {__version__} | Python {platform.python_version()} | {platform.platform()}")
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        print(f"  {'OK ' if path else 'MISSING'} {tool:<22} {path or 'install FFmpeg'}")
        ok &= bool(path)
    for mod, required in [("cv2", True), ("numpy", True), ("pandas", True), ("pyarrow", True),
                          ("openpyxl", True), ("yaml", True), ("ultralytics", True), ("lap", True), ("torch", True),
                          ("mediapipe", False)]:
        try:
            m = importlib.import_module(mod)
            print(f"  OK  {mod:<22} {getattr(m, '__version__', '')}")
        except Exception as exc:  # noqa: BLE001 - report any import failure
            print(f"  {'MISSING' if required else 'optional'} {mod:<18} {type(exc).__name__}: {exc}")
            ok &= not required
    print("All required dependencies present." if ok else "Some required dependencies are missing.")
    return 0 if ok else 1


def cmd_probe(args) -> int:
    cfg = load_config(args.config)
    videos = discover_videos(args.inputs, cfg.batch.video_extensions)
    if not videos:
        print("No videos found.")
        return 1
    rc = 0
    for v in videos:
        try:
            m = probe(v)
            print(f"{v.name}\n  {m.duration_s/60:.1f} min | {m.width}x{m.height} @ {m.fps} fps | "
                  f"{m.video_codec} | audio: {m.audio_codec or 'none'} | {m.size_bytes/1e6:.0f} MB")
            if args.json:
                print(json.dumps(m.to_dict(), ensure_ascii=False, indent=2))
        except Exception as exc:  # noqa: BLE001
            print(f"{v.name}\n  ERROR {exc}")
            rc = 1
    return rc


def cmd_run(args) -> int:
    cfg = load_config(args.config)
    if args.output:
        cfg.paths.output_dir = args.output
    if args.fps:
        cfg.sampling.fps = args.fps
        cfg.validate()
    if getattr(args, "detect_hands", False):
        cfg.extract.detect_hands = True
    out = Path(cfg.paths.output_dir)
    setup_logging(cfg.logging.level, out / "logs" / f"run_{datetime.now():%Y%m%d_%H%M%S}.log")
    stages = args.stages.split(",") if args.stages else list(DEFAULT_STAGES)
    if args.debug_video and "debug" not in stages:
        stages.append("debug")
    if stages:
        bad = [s for s in stages if s not in STAGES]
        if bad:
            log.error("Unknown stages %s. Valid: %s", bad, STAGES)
            return 2
    videos = discover_videos(args.inputs, cfg.batch.video_extensions)
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        log.error("No videos found in %s", args.inputs)
        return 1
    log.info("Processing %d video(s) -> %s", len(videos), out.resolve())
    report = run_batch(videos, cfg, stages, args.force, args.start, args.end)
    failed = int((~report["ok"]).sum())
    log.info("Done: %d ok, %d failed. Report: %s", len(report) - failed, failed, out / "run_report.csv")
    if not args.no_export:
        export_batch(out, role_gestures=cfg.events.gestures)
    return 0 if failed == 0 else 1


def cmd_export(args) -> int:
    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)
    out = Path(args.output_dir or cfg.paths.output_dir)
    split = lambda v: [x.strip() for x in v.split(",") if x.strip()] if v else None  # noqa: E731
    filters = ExportFilters(
        roles=split(args.roles),
        exclude_roles=split(args.exclude_roles) if args.exclude_roles is not None else ["transient", "other", "on_screen"],
        gestures=split(args.gestures), min_confidence=args.min_confidence, phase=args.phase)
    path = export_batch(out, Path(args.xlsx) if args.xlsx else None, filters, cfg.events.gestures)
    print(path)
    return 0


def cmd_demo(args) -> int:
    from .demo import run_demo

    setup_logging("INFO")
    ok, table, xlsx = run_demo(Path(args.output))
    vdir = Path(args.output) / "demo_conference"
    print("\nScripted vs found:\n" + table.to_string(index=False))
    print(f"\nDebug video : {vdir / 'debug.mp4'}")
    print(f"Results     : {xlsx}")
    print("\nDemo PASSED: every scripted gesture was found, nothing extra." if ok else
          "\nDemo FAILED: see the table above.")
    if args.open and sys.platform == "darwin":
        import subprocess
        subprocess.run(["open", str(vdir / "debug.mp4")])
        subprocess.run(["open", str(xlsx)])
    return 0 if ok else 1


def cmd_validate_prepare(args) -> int:
    from .validation import prepare

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)
    clips = prepare(Path(args.output_dir or cfg.paths.output_dir), Path(args.dir), args.clips, args.length, args.seed)
    print(clips[["clip_id", "video_id", "phase", "start_s", "end_s", "people"]].to_string(index=False))
    print(f"\nCreated {len(clips)} coding clips in {Path(args.dir) / 'clips'}")
    print(f"Next: read {Path(args.dir) / 'INSTRUCTIONS.md'}, fill in a copy of coding_sheet.xlsx, then run: gestures validate")
    return 0


def cmd_validate(args) -> int:
    from .validation import validate, write_report

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)
    tables = validate(Path(args.output_dir or cfg.paths.output_dir), Path(args.dir), args.tolerance)
    path = write_report(tables, Path(args.dir) / "accuracy_report.xlsx")
    pd_opts = tables["Accuracy"][["gesture", "marked_by_person", "found_by_program", "matched",
                                  "precision", "recall", "f1"]]
    print(pd_opts.to_string(index=False))
    if len(tables["Time agreement"]):
        print("\nOngoing states, compared by time (seconds):\n" + tables["Time agreement"].to_string(index=False))
    if len(tables["Coder agreement"]):
        print("\nAgreement between coders:\n" + tables["Coder agreement"].to_string(index=False))
    print(f"\nFull report (incl. every miss and extra, for tuning): {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gestures", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="check FFmpeg and Python packages")
    d.set_defaults(func=cmd_doctor)

    pr = sub.add_parser("probe", help="show video metadata")
    pr.add_argument("inputs", nargs="+")
    pr.add_argument("--config", default=_default_config())
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(func=cmd_probe)

    r = sub.add_parser("run", help="process videos")
    r.add_argument("inputs", nargs="+", help="video files and/or folders")
    r.add_argument("--config", default=_default_config())
    r.add_argument("--output", help="output folder (overrides config)")
    r.add_argument("--stages", help=f"comma list from {STAGES} (default: all except debug)")
    r.add_argument("--debug-video", action="store_true", help="also render debug.mp4 for checking")
    r.add_argument("--force", action="store_true", help="re-run stages even if already done")
    r.add_argument("--fps", type=float, help="analysis frame rate (overrides config)")
    r.add_argument("--detect-hands", action="store_true", help="enable 21-point hand tracking & gestures")
    r.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    r.add_argument("--end", type=float, default=None, help="end time in seconds (for quick tests)")
    r.add_argument("--limit", type=int, help="only the first N videos")
    r.add_argument("--no-export", action="store_true", help="skip writing results.xlsx")
    r.set_defaults(func=cmd_run)

    dm = sub.add_parser("demo", help="run the pipeline on a synthetic conference (no video needed)")
    dm.add_argument("--output", default="outputs_demo")
    dm.add_argument("--open", action="store_true", help="open the debug video and Excel file (macOS)")
    dm.set_defaults(func=cmd_demo)

    vp = sub.add_parser("validate-prepare", help="create clips + coding sheet for an accuracy check")
    vp.add_argument("--config", default=_default_config())
    vp.add_argument("--output-dir", help="processed results folder (default from config)")
    vp.add_argument("--dir", default="validation", help="where to put clips and sheets")
    vp.add_argument("--clips", type=int, default=10)
    vp.add_argument("--length", type=float, default=120.0, help="clip length in seconds")
    vp.add_argument("--seed", type=int, default=1, help="change to get a different random selection")
    vp.set_defaults(func=cmd_validate_prepare)

    va = sub.add_parser("validate", help="compare filled-in coding sheets with the program's results")
    va.add_argument("--config", default=_default_config())
    va.add_argument("--output-dir")
    va.add_argument("--dir", default="validation")
    va.add_argument("--tolerance", type=float, default=1.0, help="seconds of slack when matching times")
    va.set_defaults(func=cmd_validate)

    e = sub.add_parser("export", help="combine results into an Excel workbook")
    e.add_argument("--config", default=_default_config())
    e.add_argument("--output-dir")
    e.add_argument("--xlsx", help="output workbook (default outputs/results.xlsx)")
    e.add_argument("--roles", help="only these roles, e.g. presenter,panel")
    e.add_argument("--exclude-roles", help="default: transient,other,on_screen  (pass '' to keep everyone)")
    e.add_argument("--gestures", help="only these gestures, e.g. nod,head_shake")
    e.add_argument("--min-confidence", type=float, default=0.0, help="drop events below this confidence")
    e.add_argument("--phase", choices=["presentation", "qa", "all", "other"], help="only this part of each video")
    e.set_defaults(func=cmd_export)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001
        from .validation import NotReadyError
        from .ingest import ToolMissingError
        if isinstance(exc, (NotReadyError, ToolMissingError, FileNotFoundError, KeyError, ValueError)) \
                and not getattr(args, "traceback", False):
            # Expected problems (wrong order, missing file, typo in config): a clear message, no traceback.
            print(f"\nError: {exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
