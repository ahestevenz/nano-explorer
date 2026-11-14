"""
Tests for helpers/fix_gyro_scale.py.

Run with:
    pytest tests/helpers/ -v
"""

import json

import pytest

from helpers import fix_gyro_scale as fg

_HEADER = "#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z"
_ROWS = ["1000,1.000000,-2.000000,0.500000,9.81,0.1,-0.2", "2000,0.0,0.0,3.669,9.80,0.0,0.0"]


def _make(root, meta=None):
    (root / "mav0/imu0").mkdir(parents=True)
    (root / "mav0/imu0/data.csv").write_text("\n".join([_HEADER] + _ROWS) + "\n")
    if meta is not None:
        (root / "meta.json").write_text(json.dumps(meta))
    return root


def _rows(root, name="data.csv"):
    return [
        line.split(",")
        for line in (root / "mav0/imu0" / name).read_text().splitlines()
        if not line.startswith("#")
    ]


class TestFixRecording:
    def test_scales_only_gyro(self, tmp_path):
        root = _make(tmp_path, meta={"imu": {"samples": 2}})
        fg.fix_recording(root, 2.0, "test")
        rows = _rows(root)
        assert [r[0] for r in rows] == ["1000", "2000"]
        assert [float(v) for v in rows[0][1:4]] == [2.0, -4.0, 1.0]
        assert rows[0][4:] == ["9.81", "0.1", "-0.2"]  # accelerometer untouched
        assert (root / "mav0/imu0/data.csv").read_text().startswith(_HEADER)

    def test_keeps_original_and_notes_meta(self, tmp_path):
        root = _make(tmp_path, meta={"imu": {"samples": 2}})
        fg.fix_recording(root, 2.0, "test")
        assert _rows(root, "data_raw.csv") == [r.split(",") for r in _ROWS]
        meta = json.loads((root / "meta.json").read_text())
        assert meta["gyro_fix"]["scale"] == 2.0
        assert meta["imu"]["samples"] == 2

    def test_never_applied_twice(self, tmp_path):
        root = _make(tmp_path, meta={})
        fg.fix_recording(root, 2.0, "test")
        assert "already fixed" in fg.fix_recording(root, 2.0, "test")
        assert float(_rows(root)[0][1]) == 2.0

    def test_skips_recordings_made_with_the_correction(self, tmp_path):
        root = _make(tmp_path, meta={"imu": {"gyro_scale": 1.087}})
        assert "already applied" in fg.fix_recording(root, 2.0, "test")
        assert not (root / "mav0/imu0/data_raw.csv").exists()

    def test_missing_dataset(self, tmp_path):
        assert "not found" in fg.fix_recording(tmp_path, 2.0, "test")


class TestMain:
    def test_from_range_500(self, tmp_path):
        root = _make(tmp_path / "seq", meta={})
        assert fg.main(["--from-range", "500", str(root)]) == 0
        assert float(_rows(root)[1][3]) == pytest.approx(1.0, abs=1e-6)  # 3.669 / 3.669

    def test_range_and_scale_exclusive(self, tmp_path):
        with pytest.raises(SystemExit):
            fg.main(["--from-range", "500", "--scale", "2", str(tmp_path)])

    def test_unknown_range_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            fg.main(["--from-range", "1000", str(tmp_path)])
