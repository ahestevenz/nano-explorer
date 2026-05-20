"""
Tests for commands/mapping.py and lib/map/* — argument parsing,
config validation, and algorithm logic.

Run with:
    pytest tests/map/ -v
    pytest tests/map/ -v -m "not hardware"
"""

import argparse
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

from lib.settings import PROJECT_ROOT_PATH, NanoSettings


def _make_parser(settings: NanoSettings = None) -> argparse.ArgumentParser:
    from commands.mapping import register

    if settings is None:
        settings = NanoSettings()
    parser = argparse.ArgumentParser(prog="nano-explorer")
    sub = parser.add_subparsers(dest="group")
    map_parser = sub.add_parser("map")
    register(map_parser, settings)
    return parser


def _parse(args: list, settings: NanoSettings = None) -> argparse.Namespace:
    return _make_parser(settings).parse_args(["map"] + args)


def _dummy_frame(h=480, w=640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# commands/mapping.py — sub-command structure
class TestRegister:
    def test_odometry_subcommand_exists(self):
        ns = _parse(["odometry"])
        assert ns.command == "odometry"

    def test_slam_subcommand_exists(self):
        ns = _parse(["slam"])
        assert ns.command == "slam"

    def test_missing_subcommand_exits(self):
        with pytest.raises(SystemExit):
            _make_parser().parse_args(["map"])


# odometry defaults
class TestOdometryDefaults:
    def test_default_stream_is_true(self):
        ns = _parse(["odometry"])
        assert ns.stream is True

    def test_default_stream_port_from_settings(self):
        settings = NanoSettings()
        ns = _parse(["odometry"], settings)
        assert ns.stream_port == settings.stream_port

    def test_no_stream_flag(self):
        ns = _parse(["odometry", "--no-stream"])
        assert ns.stream is False

    def test_custom_stream_port(self):
        ns = _parse(["odometry", "--stream-port", "9191"])
        assert ns.stream_port == 9191

    def test_config_path_dest(self):
        ns = _parse(["odometry", "--config", "config/models/odometry.yaml"])
        assert ns.config_path == "config/models/odometry.yaml"

    def test_func_is_set(self):
        from commands.mapping import _run_odometry

        ns = _parse(["odometry"])
        assert ns.func is _run_odometry


# slam defaults
class TestSlamDefaults:
    def test_default_stream_is_true(self):
        ns = _parse(["slam"])
        assert ns.stream is True

    def test_default_stream_port_from_settings(self):
        settings = NanoSettings()
        ns = _parse(["slam"], settings)
        assert ns.stream_port == settings.stream_port

    def test_no_stream_flag(self):
        ns = _parse(["slam", "--no-stream"])
        assert ns.stream is False

    def test_func_is_set(self):
        from commands.mapping import _run_slam

        ns = _parse(["slam"])
        assert ns.func is _run_slam


# Dispatch — _run_* functions
class TestRunOdometry:
    @patch("lib.map.odometry.VisualOdometry")
    @patch("lib.map.odometry.OdometryConfig")
    def test_run_odometry_creates_config_and_runs(self, mock_cfg_cls, mock_impl_cls):
        from commands.mapping import _run_odometry

        mock_cfg = MagicMock()
        mock_cfg.dict.return_value = {
            "config_path": str(PROJECT_ROOT_PATH / "config/models/odometry.yaml"),
            "stream": True,
            "stream_port": 8080,
        }
        mock_cfg_cls.return_value = mock_cfg
        mock_impl_cls.return_value = MagicMock()

        args = argparse.Namespace(
            config_path=str(PROJECT_ROOT_PATH / "config/models/odometry.yaml"),
            stream=True,
            stream_port=8080,
            func=_run_odometry,
            command="odometry",
        )
        _run_odometry(args)

        mock_cfg_cls.assert_called_once()
        mock_impl_cls.assert_called_once()
        mock_impl_cls.return_value.run.assert_called_once()


class TestRunSlam:
    @patch("lib.map.slam.SlamMapper")
    @patch("lib.map.slam.SlamConfig")
    def test_run_slam_creates_config_and_runs(self, mock_cfg_cls, mock_impl_cls):
        from commands.mapping import _run_slam

        mock_cfg = MagicMock()
        mock_cfg.dict.return_value = {
            "config_path": str(PROJECT_ROOT_PATH / "config/models/slam.yaml"),
            "stream": True,
            "stream_port": 8080,
        }
        mock_cfg_cls.return_value = mock_cfg
        mock_impl_cls.return_value = MagicMock()

        args = argparse.Namespace(
            config_path=str(PROJECT_ROOT_PATH / "config/models/slam.yaml"),
            stream=True,
            stream_port=8080,
            func=_run_slam,
            command="slam",
        )
        _run_slam(args)

        mock_cfg_cls.assert_called_once()
        mock_impl_cls.assert_called_once()
        mock_impl_cls.return_value.run.assert_called_once()


# lib/map/odometry.py — config validation
class TestOdometryConfig:
    def test_valid_config_loads(self):
        from lib.map.odometry import OdometryConfig

        cfg = OdometryConfig(
            config_path=str(PROJECT_ROOT_PATH / "config/models/odometry.yaml")
        )
        assert cfg.stream is False

    def test_missing_config_raises(self, tmp_path):
        from pydantic import ValidationError

        from lib.map.odometry import OdometryConfig

        with pytest.raises(ValidationError):
            OdometryConfig(config_path=str(tmp_path / "missing.yaml"))


# lib/map/odometry.py — algorithm tests
class TestVisualOdometry:
    def test_import(self):
        from lib.map.odometry import VisualOdometry

        assert VisualOdometry is not None

    def test_initial_pose_is_identity(self):
        from lib.map.odometry import OdometryConfig, VisualOdometry

        vo = VisualOdometry(
            **OdometryConfig(
                config_path=str(PROJECT_ROOT_PATH / "config/models/odometry.yaml")
            ).dict()
        )
        assert np.allclose(vo._pose, np.eye(4))

    def test_init_detector_orb(self):
        import cv2

        from lib.map.odometry import OdometryConfig, VisualOdometry

        vo = VisualOdometry(
            **OdometryConfig(
                config_path=str(PROJECT_ROOT_PATH / "config/models/odometry.yaml")
            ).dict()
        )
        vo._init_detector("orb", 500)
        assert vo._detector is not None
        assert vo._matcher is not None

    def test_init_detector_invalid_raises(self):
        from lib.map.odometry import OdometryConfig, VisualOdometry

        vo = VisualOdometry(
            **OdometryConfig(
                config_path=str(PROJECT_ROOT_PATH / "config/models/odometry.yaml")
            ).dict()
        )
        with pytest.raises(ValueError, match="feature_detector"):
            vo._init_detector("laser", 500)

    def test_match_features_empty_when_no_prev(self):
        from lib.map.odometry import OdometryConfig, VisualOdometry

        vo = VisualOdometry(
            **OdometryConfig(
                config_path=str(PROJECT_ROOT_PATH / "config/models/odometry.yaml")
            ).dict()
        )
        # With None descriptors, match_features should return empty list
        result = vo._match_features(None, None)
        assert result == []

    def test_estimate_motion_returns_none_with_too_few_matches(self):
        from lib.map.odometry import OdometryConfig, VisualOdometry

        vo = VisualOdometry(
            **OdometryConfig(
                config_path=str(PROJECT_ROOT_PATH / "config/models/odometry.yaml")
            ).dict()
        )
        result = vo._estimate_motion([], [], [], (480, 640))
        assert result is None

    def test_annotate_frame_no_keypoints(self):
        from lib.map.odometry import VisualOdometry

        frame = _dummy_frame()
        pose = np.eye(4)
        out = VisualOdometry._annotate_frame(frame.copy(), [], [], pose)
        assert out.shape == frame.shape
        assert not np.array_equal(out, frame)  # text is drawn

    def test_annotate_frame_with_keypoints(self):
        import cv2

        from lib.map.odometry import VisualOdometry

        frame = _dummy_frame()
        pose = np.eye(4)
        kps = [cv2.KeyPoint(x=320.0, y=240.0, size=5.0)]
        out = VisualOdometry._annotate_frame(frame.copy(), kps, [], pose)
        assert out.shape == frame.shape
        assert not np.array_equal(out, frame)


# lib/map/slam.py — config validation
class TestSlamConfig:
    def test_valid_config_loads(self):
        from lib.map.slam import SlamConfig

        cfg = SlamConfig(config_path=str(PROJECT_ROOT_PATH / "config/models/slam.yaml"))
        assert cfg.stream is False

    def test_missing_config_raises(self, tmp_path):
        from pydantic import ValidationError

        from lib.map.slam import SlamConfig

        with pytest.raises(ValidationError):
            SlamConfig(config_path=str(tmp_path / "missing.yaml"))


# lib/map/slam.py — algorithm tests
class TestSlamMapper:
    def test_import(self):
        from lib.map.slam import SlamMapper

        assert SlamMapper is not None

    def test_annotate_frame_ok_state(self):
        from lib.map.slam import SlamMapper

        frame = _dummy_frame()
        out = SlamMapper._annotate_frame(frame.copy(), "OK", "orbslam2")
        assert out.shape == frame.shape
        assert not np.array_equal(out, frame)

    def test_annotate_frame_lost_state(self):
        from lib.map.slam import SlamMapper

        frame = _dummy_frame()
        out = SlamMapper._annotate_frame(frame.copy(), "LOST", "rtabmap")
        assert out.shape == frame.shape
        assert not np.array_equal(out, frame)

    def test_annotate_ok_differs_from_lost(self):
        from lib.map.slam import SlamMapper

        frame = _dummy_frame()
        out_ok = SlamMapper._annotate_frame(frame.copy(), "OK", "orbslam2")
        out_lost = SlamMapper._annotate_frame(frame.copy(), "LOST", "orbslam2")
        # Different state labels should produce different colours
        assert not np.array_equal(out_ok, out_lost)
