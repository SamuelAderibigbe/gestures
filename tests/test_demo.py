"""The demo doubles as an end-to-end test of roles, gestures, debug video and export."""

from gesture_pipeline.demo import run_demo

from .conftest import needs_ffmpeg


@needs_ffmpeg
def test_demo_finds_exactly_what_was_scripted(tmp_path):
    ok, table, xlsx = run_demo(tmp_path / "demo")
    assert ok, "\n" + table.to_string()
    assert xlsx.exists()
    assert (tmp_path / "demo" / "demo_conference" / "debug.mp4").exists()
