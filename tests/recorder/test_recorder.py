"""
Tests for lib/map/recorder.py — the EuRoC writer, the frame-writer thread and
the recording loops — with a fake camera, IMU reader and teleop (no hardware
needed), and for the `map record` command in commands/mapping.py.

Run with:
    pytest tests/recorder/ -v
"""

import argparse
import json
import threading
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from lib.camera import FrameTimes
from lib.imu import ImuSample, ImuStats
from lib.map import recorder as rec
from lib.map.recorder import EurocWriter, FrameWriterThread, Recorder, RecorderConfig
from lib.settings import NanoSettings

_MS = 1_000_000
_T0 = 5_000 * _MS  # monotonic ns of the first frame


def _sample(t_ns):
    return ImuSample(t_ns, (0.1, -0.2, 0.3), (9.8, 0.0, -0.5))


class FakeImu:
    """ImuReader stand-in: each drain() returns 5 ms-spaced samples for one frame period."""

    def __init__(
        self, start_ns=_T0 - 100 * _MS, end_extra_ns=100 * _MS, stop_after=None, stop=None
    ):
        self._next = start_ns
        self._end_extra = end_extra_ns
        self._stop_after = stop_after
        self._stop = stop
        self.drains = 0
        self.started = self.stopped = False
        self.until_ns = None  # drain() covers up to here, set by the fake camera

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True
        if self.until_ns is not None:
            self.until_ns += self._end_extra

    def drain(self):
        self.drains += 1
        if self._stop_after is not None and self.drains >= self._stop_after:
            self._stop.set()
        until = self.until_ns if self.until_ns is not None else self._next + 33 * _MS
        out = []
        while self._next <= until:
            out.append(_sample(self._next))
            self._next += 5 * _MS
        return out

    def stats(self):
        return ImuStats(10, 200.0, 5.0, 0.1, 5.2, 5.5, 0, 0)


class FakeCam:
    """Camera stand-in: n frames 33.3 ms apart, then sets the recorder's stop event."""

    def __init__(self, n, stop, imu, pts=True):
        self._n, self._stop, self._imu, self._pts = n, stop, imu, pts
        self.reads = 0
        self.released = False

    def read_timed(self):
        t = _T0 + self.reads * 33_333_333
        frame = np.full((6, 8, 3), self.reads % 256, np.uint8)
        times = FrameTimes(t, (t - _T0 + _MS) if self._pts else None)
        self.reads += 1
        self._imu.until_ns = t
        if self.reads >= self._n:
            self._stop.set()
        return frame, times

    def release(self):
        self.released = True


def _record(tmp_path, n=10, pts=True, imu_kwargs=None, **cfg):
    """Run a camera recording of n frames into tmp_path/out; returns (meta, out, fakes)."""
    out = tmp_path / "out"
    r = Recorder(out=out, **cfg)
    imu = FakeImu(**(imu_kwargs or {}))
    cam = FakeCam(n, r._stop, imu, pts=pts)  # pylint: disable=protected-access
    teleop = threading.Thread(target=r._stop.wait)  # pylint: disable=protected-access
    teleop.start()
    with patch.object(rec, "ImuReader", return_value=imu), patch.object(
        Recorder, "_open_camera", return_value=cam
    ), patch.object(Recorder, "_start_teleop_thread", return_value=teleop):
        meta = r.run()
    return meta, out, (cam, imu, teleop)


def _rows(path):
    lines = path.read_text().splitlines()
    return lines[0], [line.split(",") for line in lines[1:]]


# RecorderConfig
class TestConfig:
    def test_new_dir_ok(self, tmp_path):
        assert RecorderConfig(out=tmp_path / "new").out == tmp_path / "new"

    def test_empty_dir_ok(self, tmp_path):
        RecorderConfig(out=tmp_path)

    def test_non_empty_dir_rejected(self, tmp_path):
        (tmp_path / "x").write_text("")
        with pytest.raises(ValidationError, match="not an empty directory"):
            RecorderConfig(out=tmp_path)

    def test_file_rejected(self, tmp_path):
        f = tmp_path / "f"
        f.write_text("")
        with pytest.raises(ValidationError):
            RecorderConfig(out=f)

    def test_png_compression_range(self, tmp_path):
        with pytest.raises(ValidationError):
            RecorderConfig(out=tmp_path / "o", png_compression=10)


# Camera recording
class TestCameraRecording:
    def test_layout(self, tmp_path):
        _meta, out, _ = _record(tmp_path)
        for p in (
            "mav0/cam0/data",
            "mav0/cam0/data.csv",
            "mav0/cam0/times_raw.csv",
            "mav0/imu0/data.csv",
            "times.txt",
            "meta.json",
        ):
            assert (out / p).exists(), p

    def test_frames_named_by_read_time(self, tmp_path):
        _meta, out, _ = _record(tmp_path, n=5)
        header, rows = _rows(out / "mav0/cam0/data.csv")
        assert header == "#timestamp [ns],filename"
        stamps = [_T0 + i * 33_333_333 for i in range(5)]
        assert rows == [[str(t), f"{t}.png"] for t in stamps]
        for t in stamps:
            img = cv2.imread(str(out / "mav0/cam0/data" / f"{t}.png"), cv2.IMREAD_UNCHANGED)
            assert img.ndim == 2  # grayscale
            assert img.shape == (6, 8)
        times = (out / "times.txt").read_text().split()
        assert times == [str(t) for t in stamps]

    def test_raw_times(self, tmp_path):
        _meta, out, _ = _record(tmp_path, n=3)
        header, rows = _rows(out / "mav0/cam0/times_raw.csv")
        assert header == "#frame,timestamp [ns],read [ns],pts [ns]"
        assert rows[1] == ["1", str(_T0 + 33_333_333), str(_T0 + 33_333_333), str(33_333_333 + _MS)]

    def test_raw_times_without_pts(self, tmp_path):
        meta, out, _ = _record(tmp_path, n=3, pts=False)
        _header, rows = _rows(out / "mav0/cam0/times_raw.csv")
        assert all(r[3] == "" for r in rows)
        assert meta["camera"]["timing"]["pts"] is None

    def test_imu_csv_euroc_columns(self, tmp_path):
        _meta, out, _ = _record(tmp_path, n=3)
        header, rows = _rows(out / "mav0/imu0/data.csv")
        assert header.startswith("#timestamp [ns],w_RS_S_x")
        assert "a_RS_S_z" in header
        assert [float(v) for v in rows[0][1:]] == [0.1, -0.2, 0.3, 9.8, 0.0, -0.5]
        ts = [int(r[0]) for r in rows]
        assert ts == sorted(ts)

    def test_meta(self, tmp_path):
        meta, out, _ = _record(tmp_path, n=10)
        on_disk = json.loads((out / "meta.json").read_text())
        assert on_disk["camera"]["frames_written"] == 10
        assert on_disk["camera"]["frames_dropped"] == 0
        assert on_disk["camera"]["timestamps"] == "read"
        assert on_disk["imu_covers_camera"] is True
        assert on_disk["config"]["fps"] == 30
        assert on_disk["clock"].startswith("CLOCK_MONOTONIC")
        assert meta["camera"]["timing"]["read"]["mean_ms"] == pytest.approx(33.333, abs=0.01)

    def test_imu_starting_late_is_flagged(self, tmp_path):
        meta, _out, _ = _record(tmp_path, imu_kwargs={"start_ns": _T0 + 50 * _MS})
        assert meta["imu_covers_camera"] is False

    def test_shutdown(self, tmp_path):
        _meta, _out, (cam, imu, teleop) = _record(tmp_path)
        assert cam.released
        assert imu.started and imu.stopped
        assert not teleop.is_alive()

    def test_seconds_limit(self, tmp_path):
        clock = iter(range(0, 10**6))
        with patch.object(rec.time, "monotonic", lambda: float(next(clock))):
            meta, _out, (cam, _imu, _t) = _record(tmp_path, n=1000, seconds=3)
        assert cam.reads < 10
        assert meta["camera"]["frames_written"] == cam.reads

    def test_writer_error_raises_after_saving_meta(self, tmp_path):
        with patch.object(rec.cv2, "imwrite", return_value=False), pytest.raises(
            RuntimeError, match="Failed to write"
        ):
            _record(tmp_path, n=50)
        assert (tmp_path / "out/meta.json").exists()

    def test_meta_failure_still_saves_the_recording(self, tmp_path):
        with patch.object(rec, "_git_revision", side_effect=TypeError("boom")):
            meta, out, _ = _record(tmp_path, n=5)
        assert "boom" in meta["error"]
        assert "boom" in json.loads((out / "meta.json").read_text())["error"]
        assert len((out / "times.txt").read_text().split()) == 5
        _header, rows = _rows(out / "mav0/cam0/data.csv")
        assert len(rows) == 5


# _git_revision()
class TestGitRevision:
    def test_python36_compatible_call(self):
        # text= is Python 3.7+; the Nano runs 3.6 (ruff's UP021 fix once broke this).
        with patch.object(rec.subprocess, "run") as run:
            run.return_value.stdout = "abc1234\n"
            assert rec._git_revision() == "abc1234"  # pylint: disable=protected-access
        kwargs = run.call_args[1]
        assert "text" not in kwargs
        assert "capture_output" not in kwargs
        assert kwargs["universal_newlines"] is True

    def test_no_git(self):
        with patch.object(rec.subprocess, "run", side_effect=OSError):
            assert rec._git_revision() is None  # pylint: disable=protected-access


# --imu-only
class TestImuOnly:
    def test_imu_only(self, tmp_path):
        out = tmp_path / "out"
        r = Recorder(out=out, imu_only=True)
        imu = FakeImu(stop_after=4, stop=r._stop)  # pylint: disable=protected-access
        with patch.object(rec, "ImuReader", return_value=imu), patch.object(
            Recorder, "_start_quit_listener"
        ) as quit_listener, patch.object(Recorder, "_open_camera") as open_cam, patch.object(
            rec.time, "sleep"
        ):
            meta = r.run()
        quit_listener.assert_called_once()
        open_cam.assert_not_called()
        assert not (out / "mav0/cam0").exists()
        assert not (out / "times.txt").exists()
        _header, rows = _rows(out / "mav0/imu0/data.csv")
        assert len(rows) == meta["imu"]["samples"] > 0
        assert "camera" not in meta


# FrameWriterThread
class BlockingWriter:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.written = []

    def write_frame(self, index, *_args):
        self.entered.set()
        self.release.wait(5)
        self.written.append(index)


class TestFrameWriterThread:
    def test_drops_when_full_and_flushes_on_close(self):
        w = BlockingWriter()
        fw = FrameWriterThread(w, max_frames=2)
        frame, times = np.zeros((2, 2), np.uint8), FrameTimes(0, None)
        assert fw.submit(0, 0, frame, times)
        assert w.entered.wait(5)  # frame 0 is being written; the queue is empty
        results = [fw.submit(i, 0, frame, times) for i in range(1, 6)]
        assert results == [True, True, False, False, False]
        assert fw.dropped == 3
        w.release.set()
        fw.close()
        assert w.written == [0, 1, 2]


# EurocWriter
class TestEurocWriter:
    def test_imu_only_writer_rejects_frames(self, tmp_path):
        w = EurocWriter(tmp_path, camera=False)
        with pytest.raises(RuntimeError):
            w.write_frame(0, 1, np.zeros((2, 2), np.uint8), FrameTimes(1, None))
        w.close({})


# `map record` command
def _parse(args):
    from commands.mapping import register

    parser = argparse.ArgumentParser(prog="nano-explorer")
    sub = parser.add_subparsers(dest="group")
    register(sub.add_parser("map"), NanoSettings())
    return parser.parse_args(["map"] + args)


class TestRecordCommand:
    def test_out_required(self):
        with pytest.raises(SystemExit):
            _parse(["record"])

    def test_defaults(self):
        ns = _parse(["record", "--out", "/tmp/x"])
        assert ns.out == "/tmp/x"
        assert ns.imu_only is False
        assert ns.seconds is None
        assert ns.fps == 30
        assert ns.png_compression == 1
        assert ns.imu_rate_hz == pytest.approx(200.0)
        assert ns.imu_bus == 0
        assert ns.stream is False
        assert ns.speed == pytest.approx(0.2)

    def test_custom(self):
        ns = _parse(
            ["record", "--out", "d", "--imu-only", "-t", "7200", "--rate", "100", "--stream"]
        )
        assert ns.imu_only is True
        assert ns.seconds == pytest.approx(7200.0)
        assert ns.imu_rate_hz == pytest.approx(100.0)
        assert ns.stream is True

    @patch("lib.map.recorder.Recorder")
    def test_run_builds_config_and_runs(self, mock_impl_cls, tmp_path):
        from commands.mapping import _run_record

        _run_record(_parse(["record", "--out", str(tmp_path / "new"), "--imu-only"]))

        kwargs = mock_impl_cls.call_args[1]
        assert kwargs["out"] == tmp_path / "new"
        assert kwargs["imu_only"] is True
        mock_impl_cls.return_value.run.assert_called_once()
