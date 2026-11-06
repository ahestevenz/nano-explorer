"""
Tests for the frame timestamps in lib/camera.py — read_timed(), read_stamped()
and the `map camera-check` analysis — with a fake VideoCapture and clock (no
camera needed), and for the `map camera-check` command in commands/mapping.py.

Run with:
    pytest tests/camera/ -v
"""

import argparse
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from lib import camera
from lib.camera import Camera, CameraConfig, FrameTimes, analyze_frame_times
from lib.settings import NanoSettings

_MS = 1_000_000
_PERIOD = 33_333_333  # 30 fps, ns


class FakeCap:
    """VideoCapture stand-in: returns a frame per read() and a scripted POS_MSEC."""

    def __init__(self, pts_ms):
        self._pts_ms = list(pts_ms)
        self._current = 0.0

    def isOpened(self):  # pylint: disable=invalid-name
        return True

    def read(self):
        self._current = self._pts_ms.pop(0)
        return True, np.zeros((4, 4, 3), np.uint8)

    def get(self, prop):
        assert prop == cv2.CAP_PROP_POS_MSEC
        return self._current

    def release(self):
        pass


def _camera(pts_ms, read_ns, timestamps="read"):
    """A Camera whose reads return `pts_ms` buffer timestamps at `read_ns` times."""
    cam = Camera(timestamps=timestamps)
    cam._cap = FakeCap(pts_ms)  # pylint: disable=protected-access
    clock = iter(read_ns)
    patcher = patch.object(camera, "monotonic_ns", lambda: next(clock))
    patcher.start()
    return cam, patcher


@pytest.fixture(name="make_camera")
def _make_camera_fixture():
    patchers = []

    def _make(*args, **kwargs):
        cam, p = _camera(*args, **kwargs)
        patchers.append(p)
        return cam

    yield _make
    for p in patchers:
        p.stop()


# CameraConfig
class TestConfig:
    def test_default_is_read(self):
        assert CameraConfig().timestamps == "read"

    def test_rejects_unknown(self):
        with pytest.raises(ValidationError):
            CameraConfig(timestamps="wall")


# read_timed() / read_stamped()
class TestReadTimed:
    def test_both_clocks(self, make_camera):
        cam = make_camera([100.5], [7 * _MS])
        _frame, times = cam.read_timed()
        assert times == FrameTimes(read_ns=7 * _MS, pts_ns=100_500_000)

    def test_zero_pts_is_valid(self, make_camera):
        cam = make_camera([0.0], [7 * _MS])
        assert cam.read_timed()[1].pts_ns == 0

    def test_negative_pts_is_none(self, make_camera):
        cam = make_camera([-1.0], [7 * _MS])
        assert cam.read_timed()[1].pts_ns is None


class TestReadStamped:
    def test_read_mode_uses_read_time(self, make_camera):
        cam = make_camera([10.0, 43.3], [5_000 * _MS, 5_040 * _MS])
        assert cam.read_stamped()[1] == 5_000 * _MS
        assert cam.read_stamped()[1] == 5_040 * _MS

    def test_pts_mode_maps_onto_monotonic_with_min_lag(self, make_camera):
        # Pipeline clock starts at 0, monotonic at 9 s; latency 50, 40, 60 ms.
        pts = [0.0, 33.0, 66.0]
        reads = [(9_000 + p + lag) * _MS for p, lag in zip(pts, [50, 40, 60])]
        cam = make_camera(pts, [int(r) for r in reads], timestamps="pts")
        stamps = [cam.read_stamped()[1] for _ in pts]
        # Offset tightens to the smallest lag (40 ms) on the second frame.
        assert stamps == [9_050 * _MS, (9_000 + 33 + 40) * _MS, (9_000 + 66 + 40) * _MS]

    def test_pts_mode_stamps_keep_capture_spacing(self, make_camera):
        n = 30
        pts = [i * 33.333 for i in range(n)]
        lags = np.random.default_rng(0).uniform(30, 70, n)
        reads = [int((1_000 + p + lag) * _MS) for p, lag in zip(pts, lags)]
        cam = make_camera(pts, reads, timestamps="pts")
        stamps = [cam.read_stamped()[1] for _ in range(n)]
        offset = stamps[-1] - int(round(pts[-1] * _MS))
        # Once settled, every stamp is PTS + one fixed offset: the read jitter is gone.
        settled = [s - int(round(p * _MS)) for s, p in zip(stamps, pts)][int(np.argmin(lags)) :]
        assert all(o == offset for o in settled)

    def test_pts_mode_rejects_missing_pts(self, make_camera):
        cam = make_camera([-1.0], [1 * _MS], timestamps="pts")
        with pytest.raises(RuntimeError, match="timestamps='read'"):
            cam.read_stamped()

    def test_pts_mode_rejects_repeated_pts(self, make_camera):
        cam = make_camera([10.0, 10.0], [50 * _MS, 90 * _MS], timestamps="pts")
        cam.read_stamped()
        with pytest.raises(RuntimeError, match="unusable"):
            cam.read_stamped()

    def test_not_open(self):
        with pytest.raises(RuntimeError, match="not open"):
            Camera().read_stamped()


# analyze_frame_times()
def _times(n=90, read_jitter_ms=5.0, pts=True, pts_tracks_read=False, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        cap = i * _PERIOD
        read = 10**9 + cap + int((40 + rng.uniform(0, read_jitter_ms)) * _MS)
        if not pts:
            p = None
        elif pts_tracks_read:
            p = read - 10**9  # the query time: read time minus a constant
        else:
            p = cap + _MS
        out.append(FrameTimes(read, p))
    return out


class TestAnalyze:
    def test_real_capture_pts(self):
        res = analyze_frame_times(_times(), fps=30)
        assert res.verdict == "pts"
        assert res.frames == 90
        assert res.pts.std_ms == pytest.approx(0.0, abs=1e-3)
        assert res.read.std_ms > 1.0
        assert res.lag_spread_ms > 1.0

    def test_no_pts(self):
        res = analyze_frame_times(_times(pts=False), fps=30)
        assert res.verdict == "read"
        assert res.pts is None
        assert "no buffer timestamp" in res.reason

    def test_all_zero_pts(self):
        t = [FrameTimes(r.read_ns, 0) for r in _times()]
        res = analyze_frame_times(t, fps=30)
        assert res.verdict == "read"
        assert "always 0" in res.reason

    def test_pts_that_is_query_time(self):
        res = analyze_frame_times(_times(pts_tracks_read=True), fps=30)
        assert res.verdict == "read"
        assert "lock-step" in res.reason

    def test_non_increasing_pts(self):
        t = _times()
        t[10] = FrameTimes(t[10].read_ns, t[9].pts_ns)
        res = analyze_frame_times(t, fps=30)
        assert res.verdict == "read"
        assert "increase" in res.reason

    def test_pts_no_steadier(self):
        # Read time already steady: the PTS buys nothing.
        steady = _times(read_jitter_ms=1.0)
        t = [FrameTimes(r.read_ns, r.pts_ns + (i % 2) * _MS) for i, r in enumerate(steady)]
        res = analyze_frame_times(t, fps=30)
        assert res.verdict == "read"
        assert "no steadier" in res.reason

    def test_late_counts_dropped_frames(self):
        t = _times()
        del t[40:42]  # two frames dropped by appsink
        res = analyze_frame_times(t, fps=30)
        assert res.pts.late == 1
        assert res.read.late == 1

    def test_too_few_frames(self):
        with pytest.raises(ValueError):
            analyze_frame_times(_times(n=2), fps=30)


# `map camera-check` command
def _parse(args):
    from commands.mapping import register

    parser = argparse.ArgumentParser(prog="nano-explorer")
    sub = parser.add_subparsers(dest="group")
    register(sub.add_parser("map"), NanoSettings())
    return parser.parse_args(["map"] + args)


class TestCameraCheckCommand:
    def test_defaults(self):
        ns = _parse(["camera-check"])
        assert ns.seconds == pytest.approx(20.0)
        assert ns.fps == 30
        assert ns.exposure_max_ms is None
        assert ns.load is False

    def test_custom(self):
        ns = _parse(["camera-check", "-t", "60", "--fps", "20", "--max-exposure-ms", "8", "--load"])
        assert ns.seconds == pytest.approx(60.0)
        assert ns.fps == 20
        assert ns.exposure_max_ms == pytest.approx(8.0)
        assert ns.load is True

    @patch("lib.camera.CameraCheck")
    def test_run_builds_config_and_runs(self, mock_impl_cls):
        from commands.mapping import _run_camera_check

        _run_camera_check(_parse(["camera-check", "-t", "5"]))

        kwargs = mock_impl_cls.call_args[1]
        assert kwargs["seconds"] == pytest.approx(5.0)
        assert kwargs["fps"] == 30
        mock_impl_cls.return_value.run.assert_called_once()
