"""
Tests for helpers/check_recording.py, on small synthetic EuRoC recordings.

Run with:
    pytest tests/helpers/ -v
"""

import json

import pytest

from helpers import check_recording as cr

_MS = 1_000_000
_PERIOD = 33_333_333
_T0 = 10**12
_LAG = 40 * _MS  # usual read time - PTS


def _make(
    root,
    n=90,
    late=None,
    drop=(),
    stall_after=(),
    imu_start=_T0 - 500 * _MS,
    imu_end_extra=200 * _MS,
    pts=True,
    meta=None,
    skip_png=None,
):
    """
    Write a recording: n frames at 30 fps; `late` maps frame -> extra read ms;
    frame indices in `drop` are lost by the camera, those in `stall_after` are
    lost to a recording-loop stall (the next frame is read late).
    """
    late = dict(late or {})
    (root / "mav0/cam0/data").mkdir(parents=True)
    (root / "mav0/imu0").mkdir(parents=True)
    cam, raw, times = ["#timestamp [ns],filename"], ["#frame,timestamp [ns],read [ns],pts [ns]"], []
    lost = set(drop) | set(stall_after)
    k = 0
    for i in range(n):
        if i in lost:
            continue
        p = i * _PERIOD
        extra = late.get(i, 0) + (20 if i - 1 in stall_after else 0)
        read = _T0 + p + _LAG + extra * _MS
        cam.append(f"{read},{read}.png")
        raw.append(f"{k},{read},{read},{p if pts else ''}")
        times.append(str(read))
        if i != skip_png:
            (root / "mav0/cam0/data" / f"{read}.png").write_bytes(b"")
        k += 1
    (root / "mav0/cam0/data.csv").write_text("\n".join(cam) + "\n")
    (root / "mav0/cam0/times_raw.csv").write_text("\n".join(raw) + "\n")
    (root / "times.txt").write_text("\n".join(times) + "\n")
    last = int(times[-1])
    imu = ["#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z"]
    t = imu_start
    while t <= last + imu_end_extra:
        imu.append(f"{t},0,0,0,9.8,0,0")
        t += 5 * _MS
    (root / "mav0/imu0/data.csv").write_text("\n".join(imu) + "\n")
    (root / "meta.json").write_text(json.dumps(meta or {"camera": {}}))
    return root


def _levels(report, name=None):
    return [c.level for c in report.checks if name is None or c.name == name]


def _messages(report):
    return " | ".join(c.message for c in report.checks)


class TestCheckRecording:
    def test_clean_recording(self, tmp_path):
        report = cr.check_recording(_make(tmp_path))
        assert set(_levels(report)) == {cr.OK}
        assert not report.failed

    def test_startup_lateness_is_skipped(self, tmp_path):
        report = cr.check_recording(_make(tmp_path, late={0: 20, 1: 13}))
        assert set(_levels(report)) == {cr.OK}
        assert "2 late frames in the first 1 s (skipped)" in _messages(report)
        assert 0 not in report.keep and 1 not in report.keep

    def test_late_frame_after_start_is_flagged_and_trimmed(self, tmp_path):
        report = cr.check_recording(_make(tmp_path, late={60: 12}))
        assert _levels(report, "read time") == [cr.WARN]
        assert 60 not in report.keep
        assert 59 in report.keep and 61 in report.keep

    def test_camera_gap(self, tmp_path):
        report = cr.check_recording(_make(tmp_path, drop=(50, 51, 52)))
        assert "camera missed frames: ~3 frames lost in 1 gaps" in _messages(report)

    def test_stall_gap(self, tmp_path):
        report = cr.check_recording(_make(tmp_path, stall_after=(50, 51, 52)))
        assert "recording loop stalled: ~3 frames lost" in _messages(report)
        assert "camera missed" not in _messages(report)

    def test_missing_png_fails(self, tmp_path):
        report = cr.check_recording(_make(tmp_path, skip_png=10))
        assert report.failed
        assert "listed frames missing" in _messages(report)

    def test_imu_not_covering_fails(self, tmp_path):
        report = cr.check_recording(_make(tmp_path, imu_start=_T0 + 500 * _MS))
        assert report.failed
        assert "doesn't cover the frames" in _messages(report)

    def test_keep_starts_after_imu(self, tmp_path):
        root = _make(tmp_path, imu_start=_T0 + 2000 * _MS)
        report = cr.check_recording(root)
        first_imu = cr.load_recording(root).imu_ns[0]
        frames = cr.load_recording(root).frame_ns
        assert all(frames[i] > first_imu for i in report.keep)

    def test_no_pts(self, tmp_path):
        report = cr.check_recording(_make(tmp_path, pts=False))
        assert _levels(report, "read time") == [cr.WARN]
        assert not report.failed

    def test_meta_error_warns(self, tmp_path):
        report = cr.check_recording(_make(tmp_path, meta={"error": "x"}))
        assert cr.WARN in _levels(report, "layout")

    def test_times_mismatch_fails(self, tmp_path):
        root = _make(tmp_path)
        (root / "times.txt").write_text("1\n2\n")
        assert cr.check_recording(root).failed

    def test_unreadable_dataset_fails(self, tmp_path):
        report = cr.check_recording(tmp_path / "nope")
        assert report.failed
        assert "can't read" in _messages(report)


class TestMain:
    def test_write_times(self, tmp_path, capsys):
        root = _make(tmp_path / "seq", late={60: 12})
        assert cr.main([str(root), "--write-times"]) == 0
        stamps = (root / "times_trimmed.txt").read_text().split()
        report = cr.check_recording(root)
        assert len(stamps) == len(report.keep)
        assert "wrote times_trimmed.txt" in capsys.readouterr().out

    def test_exit_code_on_failure(self, tmp_path):
        bad = _make(tmp_path / "bad", skip_png=3)
        good = _make(tmp_path / "good")
        assert cr.main([str(good), str(bad), "--write-times"]) == 1
        assert not (bad / "times_trimmed.txt").exists()
        assert (good / "times_trimmed.txt").exists()

    @pytest.mark.parametrize("flag", ["--skip-start", "--max-read-error-ms"])
    def test_options_parse(self, tmp_path, flag):
        assert cr.main([str(_make(tmp_path)), flag, "2"]) == 0
