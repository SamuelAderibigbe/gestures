import pytest

from gesture_pipeline.frames import FrameSampler

from .conftest import frame_number_of


def test_samples_at_requested_rate(numbered_video):
    frames = list(FrameSampler(numbered_video, fps=8))  # 72 frames @24fps = 3 s
    assert len(frames) == 24
    assert [f.source_index for f in frames] == list(range(0, 72, 3))
    for f in frames:
        assert frame_number_of(f.image) == f.source_index
        assert f.t == pytest.approx(f.source_index / 24, abs=0.02)
    assert [f.index for f in frames] == list(range(24))


def test_rate_above_source_returns_every_frame(numbered_video):
    frames = list(FrameSampler(numbered_video, fps=100))
    assert len(frames) == 72


def test_time_window(numbered_video):
    frames = list(FrameSampler(numbered_video, fps=4, start=1.0, end=2.0))
    ts = [f.t for f in frames]
    assert ts and min(ts) >= 1.0 - 0.05 and max(ts) <= 2.0 + 1e-6
    assert len(frames) in (4, 5)


def test_downscale_keeps_aspect(numbered_video):
    f = next(iter(FrameSampler(numbered_video, fps=1, max_width=160)))
    assert f.image.shape[:2] == (120, 160)
    assert f.scale == pytest.approx(0.5)


def test_expected_count_close(numbered_video):
    s = FrameSampler(numbered_video, fps=8)
    assert abs(s.expected_count() - len(list(s))) <= 1


def test_bad_inputs(tmp_path):
    with pytest.raises(ValueError):
        FrameSampler(tmp_path / "x.mp4", fps=0)
    with pytest.raises(ValueError):
        list(FrameSampler(tmp_path / "missing.mp4", fps=5))
