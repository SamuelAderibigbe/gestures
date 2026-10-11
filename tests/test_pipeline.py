import wave

import pandas as pd

from gesture_pipeline.cli import main
from gesture_pipeline.config import Config
from gesture_pipeline.ingest import probe
from gesture_pipeline.pipeline import discover_videos, process_video, run_batch
from gesture_pipeline.store import read_frames

from .conftest import needs_ffmpeg


@needs_ffmpeg
def test_probe(av_video):
    m = probe(av_video)
    assert (m.width, m.height) == (640, 360)
    assert abs(m.duration_s - 3) < 0.2 and m.has_audio and abs(m.fps - 25) < 0.01


@needs_ffmpeg
def test_full_run_then_resume(av_video, tmp_path):
    cfg = Config()
    cfg.paths.output_dir = str(tmp_path / "out")
    cfg.extract.models = []  # no ML model: tests the plumbing only
    r1 = process_video(av_video, cfg)
    assert r1.ok, r1.error
    assert r1.stages_run == ["probe", "audio", "extract", "people", "events"]
    vdir = tmp_path / "out" / r1.video_id
    with wave.open(str(vdir / "audio.wav")) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
    frames = read_frames(vdir / "frames.parquet")
    assert 23 <= len(frames) <= 25  # 3 s at 8 fps
    from gesture_pipeline.export import EVENT_COLUMNS
    assert list(pd.read_csv(vdir / "events.csv").columns) == EVENT_COLUMNS
    for f in ("people.csv", "phases.csv", "visibility.csv", "summary.csv"):
        assert (vdir / f).exists()

    r2 = process_video(av_video, cfg)  # nothing changed -> everything skipped
    assert r2.ok and r2.stages_run == [] and len(r2.stages_skipped) == 5

    cfg.sampling.fps = 4  # changing a setting re-runs only what depends on it
    r3 = process_video(av_video, cfg)
    assert r3.stages_run == ["extract", "people", "events"]


def test_broken_video_does_not_stop_batch(tmp_path, numbered_video):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a video")
    cfg = Config()
    cfg.paths.output_dir = str(tmp_path / "out")
    cfg.audio.extract = False
    cfg.extract.models = []
    report = run_batch([bad, numbered_video], cfg, stages=["extract", "events"])
    assert list(report["ok"]) == [False, True]


def test_discover(tmp_path, numbered_video):
    (tmp_path / "notes.txt").write_text("x")
    found = discover_videos([tmp_path, numbered_video], [".avi", ".mp4"])
    assert found == [numbered_video]


@needs_ffmpeg
def test_cli_end_to_end(av_video, tmp_path):
    out = tmp_path / "cli_out"
    conf = tmp_path / "c.yaml"
    conf.write_text("extract:\n  models: []\n")
    assert main(["run", str(av_video), "--config", str(conf), "--output", str(out), "--end", "2",
                 "--debug-video"]) == 0
    assert (out / av_video.stem.replace(" ", "_") / "debug.mp4").exists()
    assert (out / "results.xlsx").exists() and (out / "run_report.csv").exists()
