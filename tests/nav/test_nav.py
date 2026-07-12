"""
Tests for commands/navigation.py and lib/nav/* — argument parsing,
config validation, and algorithm logic.

Run with:
    pytest tests/nav/ -v
    pytest tests/nav/ -v -m "not hardware"
"""

import argparse
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

from lib.settings import PROJECT_ROOT_PATH, NanoSettings


def _make_parser(settings: NanoSettings = None) -> argparse.ArgumentParser:
    from commands.navigation import register

    if settings is None:
        settings = NanoSettings()
    parser = argparse.ArgumentParser(prog="nano-explorer")
    sub = parser.add_subparsers(dest="group")
    nav_parser = sub.add_parser("nav")
    register(nav_parser, settings)
    return parser


def _parse(args: list, settings: NanoSettings = None) -> argparse.Namespace:
    return _make_parser(settings).parse_args(["nav"] + args)


def _dummy_frame(h=480, w=640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# commands/navigation.py — sub-command structure
class TestRegister:
    def test_line_follow_subcommand_exists(self):
        ns = _parse(["line-follow"])
        assert ns.command == "line-follow"

    def test_road_follow_subcommand_exists(self):
        ns = _parse(["road-follow", "--model", "assets/models/fake.pth"])
        assert ns.command == "road-follow"

    def test_apriltag_subcommand_exists(self):
        ns = _parse(["apriltag"])
        assert ns.command == "apriltag"

    def test_missing_subcommand_exits(self):
        with pytest.raises(SystemExit):
            _make_parser().parse_args(["nav"])


# line-follow defaults
class TestLineFollowDefaults:
    def test_default_stream_is_true(self):
        ns = _parse(["line-follow"])
        assert ns.stream is True

    def test_default_speed_from_settings(self):
        settings = NanoSettings()
        ns = _parse(["line-follow"], settings)
        assert ns.speed == settings.default_speed

    def test_default_turn_gain_from_settings(self):
        settings = NanoSettings()
        ns = _parse(["line-follow"], settings)
        assert ns.turn_gain == settings.default_turn_gain

    def test_default_stream_port_from_settings(self):
        settings = NanoSettings()
        ns = _parse(["line-follow"], settings)
        assert ns.stream_port == settings.stream_port

    def test_func_is_set(self):
        from commands.navigation import _run_line_follow

        ns = _parse(["line-follow"])
        assert ns.func is _run_line_follow


class TestLineFollowArgs:
    def test_custom_speed(self):
        ns = _parse(["line-follow", "--speed", "0.6"])
        assert ns.speed == pytest.approx(0.6)

    def test_custom_turn_gain(self):
        ns = _parse(["line-follow", "--turn-gain", "0.8"])
        assert ns.turn_gain == pytest.approx(0.8)

    def test_no_stream_flag(self):
        ns = _parse(["line-follow", "--no-stream"])
        assert ns.stream is False

    def test_custom_stream_port(self):
        ns = _parse(["line-follow", "--stream-port", "9090"])
        assert ns.stream_port == 9090

    def test_config_path_mapped(self):
        ns = _parse(["line-follow", "--config", "config/models/line_follow.yaml"])
        assert ns.config_path == "config/models/line_follow.yaml"


# road-follow defaults
class TestRoadFollowDefaults:
    def test_default_stream_is_true(self):
        ns = _parse(["road-follow", "--model", "assets/models/fake.pth"])
        assert ns.stream is True

    def test_default_speed_from_settings(self):
        settings = NanoSettings()
        ns = _parse(["road-follow", "--model", "assets/models/fake.pth"], settings)
        assert ns.speed == settings.default_speed

    def test_default_turn_gain_from_settings(self):
        settings = NanoSettings()
        ns = _parse(["road-follow", "--model", "assets/models/fake.pth"], settings)
        assert ns.turn_gain == settings.default_turn_gain

    def test_model_path_dest(self):
        ns = _parse(["road-follow", "--model", "some/model.pth"])
        assert ns.model_path == "some/model.pth"

    def test_func_is_set(self):
        from commands.navigation import _run_road_follow

        ns = _parse(["road-follow", "--model", "m.pth"])
        assert ns.func is _run_road_follow


# apriltag defaults
class TestAprilTagDefaults:
    def test_default_stream_is_true(self):
        ns = _parse(["apriltag"])
        assert ns.stream is True

    def test_default_speed_from_settings(self):
        settings = NanoSettings()
        ns = _parse(["apriltag"], settings)
        assert ns.speed == settings.default_speed

    def test_default_stream_port_from_settings(self):
        settings = NanoSettings()
        ns = _parse(["apriltag"], settings)
        assert ns.stream_port == settings.stream_port

    def test_func_is_set(self):
        from commands.navigation import _run_apriltag

        ns = _parse(["apriltag"])
        assert ns.func is _run_apriltag


# Dispatch — _run_* functions
class TestRunLineFollow:
    @patch("lib.nav.line_follower.LineFollower")
    @patch("lib.nav.line_follower.LineFollowerConfig")
    def test_run_line_follow_creates_config_and_runs(self, mock_cfg_cls, mock_impl_cls):
        from commands.navigation import _run_line_follow

        mock_cfg = MagicMock()
        mock_cfg.dict.return_value = {
            "config_path": str(PROJECT_ROOT_PATH / "config/models/line_follow.yaml"),
            "speed": 0.3,
            "turn_gain": 0.5,
            "stream": True,
            "stream_port": 8080,
        }
        mock_cfg_cls.return_value = mock_cfg
        mock_impl_cls.return_value = MagicMock()

        args = argparse.Namespace(
            config_path=str(PROJECT_ROOT_PATH / "config/models/line_follow.yaml"),
            speed=0.3,
            turn_gain=0.5,
            stream=True,
            stream_port=8080,
            func=_run_line_follow,
            command="line-follow",
        )
        _run_line_follow(args)

        mock_cfg_cls.assert_called_once()
        mock_impl_cls.assert_called_once()
        mock_impl_cls.return_value.run.assert_called_once()


class TestRunRoadFollow:
    @patch("lib.nav.road_follower.RoadFollower")
    @patch("lib.nav.road_follower.RoadFollowerConfig")
    def test_run_road_follow_creates_config_and_runs(self, mock_cfg_cls, mock_impl_cls):
        from commands.navigation import _run_road_follow

        mock_cfg = MagicMock()
        mock_cfg.dict.return_value = {
            "model_path": "/tmp/m.pth",
            "speed": 0.3,
            "turn_gain": 0.5,
            "stream": True,
            "stream_port": 8080,
        }
        mock_cfg_cls.return_value = mock_cfg
        mock_impl_cls.return_value = MagicMock()

        args = argparse.Namespace(
            model_path="/tmp/m.pth",
            speed=0.3,
            turn_gain=0.5,
            stream=True,
            stream_port=8080,
            func=_run_road_follow,
            command="road-follow",
        )
        _run_road_follow(args)

        mock_cfg_cls.assert_called_once()
        mock_impl_cls.assert_called_once()
        mock_impl_cls.return_value.run.assert_called_once()


class TestRunAprilTag:
    @patch("lib.nav.apriltag_nav.AprilTagNavigator")
    @patch("lib.nav.apriltag_nav.AprilTagNavConfig")
    def test_run_apriltag_creates_config_and_runs(self, mock_cfg_cls, mock_impl_cls):
        from commands.navigation import _run_apriltag

        mock_cfg = MagicMock()
        mock_cfg.dict.return_value = {
            "config_path": str(PROJECT_ROOT_PATH / "config/models/apriltag.yaml"),
            "speed": 0.3,
            "turn_gain": 0.5,
            "stream": True,
            "stream_port": 8080,
        }
        mock_cfg_cls.return_value = mock_cfg
        mock_impl_cls.return_value = MagicMock()

        args = argparse.Namespace(
            config_path=str(PROJECT_ROOT_PATH / "config/models/apriltag.yaml"),
            speed=0.3,
            turn_gain=0.5,
            stream=True,
            stream_port=8080,
            func=_run_apriltag,
            command="apriltag",
        )
        _run_apriltag(args)

        mock_cfg_cls.assert_called_once()
        mock_impl_cls.assert_called_once()
        mock_impl_cls.return_value.run.assert_called_once()


# lib/nav/line_follower.py — config validation
class TestLineFollowerConfig:
    def test_valid_config_loads(self):
        from lib.nav.line_follower import LineFollowerConfig

        cfg = LineFollowerConfig(
            config_path=str(PROJECT_ROOT_PATH / "config/models/line_follow.yaml")
        )
        assert cfg.stream is False

    def test_missing_config_raises(self, tmp_path):
        from pydantic import ValidationError

        from lib.nav.line_follower import LineFollowerConfig

        with pytest.raises(ValidationError):
            LineFollowerConfig(config_path=str(tmp_path / "missing.yaml"))

    def test_default_speed(self):
        from lib.nav.line_follower import LineFollowerConfig

        cfg = LineFollowerConfig(
            config_path=str(PROJECT_ROOT_PATH / "config/models/line_follow.yaml")
        )
        assert cfg.speed == pytest.approx(0.3)


# lib/nav/line_follower.py — algorithm tests
class TestLineFollower:
    def test_import(self):
        from lib.nav.line_follower import LineFollower

        assert LineFollower is not None

    def test_white_line_centroid_in_white_frame(self):
        """A frame with a central white stripe should yield a centroid near horizontal centre."""
        from lib.nav.line_follower import LineFollower, LineFollowerConfig

        follower = LineFollower(
            **LineFollowerConfig(
                config_path=str(PROJECT_ROOT_PATH / "config/models/line_follow.yaml")
            ).dict()
        )
        follower._line_color = "white"
        follower._min_area = 100

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # White stripe in the bottom third, centred horizontally
        frame[320:, 270:370] = [255, 255, 255]

        centroid, mask = follower._find_line_centroid(frame)
        assert centroid is not None
        cx, _ = centroid
        assert 220 < cx < 420, f"Expected centroid near centre, got cx={cx}"

    def test_dark_frame_returns_no_centroid(self):
        from lib.nav.line_follower import LineFollower, LineFollowerConfig

        follower = LineFollower(
            **LineFollowerConfig(
                config_path=str(PROJECT_ROOT_PATH / "config/models/line_follow.yaml")
            ).dict()
        )
        follower._line_color = "white"
        follower._min_area = 500
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        centroid, _ = follower._find_line_centroid(frame)
        assert centroid is None

    def test_annotate_no_line(self):
        from lib.nav.line_follower import LineFollower

        frame = _dummy_frame()
        out = LineFollower._annotate_frame(frame.copy(), None)
        assert out.shape == frame.shape

    def test_annotate_with_centroid(self):
        from lib.nav.line_follower import LineFollower

        frame = _dummy_frame()
        out = LineFollower._annotate_frame(frame.copy(), (320, 400))
        assert out.shape == frame.shape
        assert not np.array_equal(out, frame)

    def test_line_left_of_centre_gives_negative_steering(self):
        """Line on the left half → steer left (negative error)."""
        from lib.nav.line_follower import LineFollower, LineFollowerConfig

        follower = LineFollower(
            **LineFollowerConfig(
                config_path=str(PROJECT_ROOT_PATH / "config/models/line_follow.yaml")
            ).dict()
        )
        follower._line_color = "white"
        follower._min_area = 100

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[320:, 40:140] = [255, 255, 255]  # white stripe on left side

        centroid, _ = follower._find_line_centroid(frame)
        if centroid is not None:
            error = (centroid[0] - 320) / 320
            assert error < 0, f"Left stripe should give negative error, got {error:.3f}"

    def test_line_right_of_centre_gives_positive_steering(self):
        from lib.nav.line_follower import LineFollower, LineFollowerConfig

        follower = LineFollower(
            **LineFollowerConfig(
                config_path=str(PROJECT_ROOT_PATH / "config/models/line_follow.yaml")
            ).dict()
        )
        follower._line_color = "white"
        follower._min_area = 100

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[320:, 500:600] = [255, 255, 255]  # white stripe on right side

        centroid, _ = follower._find_line_centroid(frame)
        if centroid is not None:
            error = (centroid[0] - 320) / 320
            assert error > 0, f"Right stripe should give positive error, got {error:.3f}"

    def test_small_contour_returns_no_centroid(self):
        from lib.nav.line_follower import LineFollower, LineFollowerConfig

        follower = LineFollower(
            **LineFollowerConfig(
                config_path=str(PROJECT_ROOT_PATH / "config/models/line_follow.yaml")
            ).dict()
        )
        follower._line_color = "white"
        follower._min_area = 500

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Only a 5×5 white patch — well below min_area
        frame[400:405, 318:323] = [255, 255, 255]
        centroid, _ = follower._find_line_centroid(frame)
        assert centroid is None

    def test_annotate_cnn_frame(self):
        from lib.nav.line_follower import LineFollower

        frame = _dummy_frame()
        out = LineFollower._annotate_cnn_frame(frame.copy(), 0.5)
        assert out.shape == frame.shape
        assert not np.array_equal(out, frame)


# lib/nav/road_follower.py — config validation
class TestRoadFollowerConfig:
    def test_import(self):
        from lib.nav.road_follower import RoadFollower

        assert RoadFollower is not None

    def test_missing_model_raises(self):
        from pydantic import ValidationError

        from lib.nav.road_follower import RoadFollowerConfig

        with pytest.raises(ValidationError):
            RoadFollowerConfig(model_path="/nonexistent/road_follower.pth")

    def test_annotate_frame_draws_arrow(self):
        from lib.nav.road_follower import RoadFollower

        frame = _dummy_frame()
        out = RoadFollower._annotate_frame(frame.copy(), 0.3)
        assert out.shape == frame.shape
        assert not np.array_equal(out, frame)

    def test_annotate_negative_steering(self):
        from lib.nav.road_follower import RoadFollower

        frame = _dummy_frame()
        out_left = RoadFollower._annotate_frame(frame.copy(), -0.5)
        out_right = RoadFollower._annotate_frame(frame.copy(), 0.5)
        assert not np.array_equal(out_left, out_right)


# lib/nav/apriltag_nav.py — config and algorithm
class TestAprilTagNavigator:
    def test_import(self):
        from lib.nav.apriltag_nav import AprilTagNavigator

        assert AprilTagNavigator is not None

    def test_missing_config_raises(self, tmp_path):
        from pydantic import ValidationError

        from lib.nav.apriltag_nav import AprilTagNavConfig

        with pytest.raises(ValidationError):
            AprilTagNavConfig(config_path=str(tmp_path / "missing.yaml"))

    def test_annotate_no_detections(self):
        from lib.nav.apriltag_nav import AprilTagNavigator

        frame = _dummy_frame()
        out = AprilTagNavigator._annotate_frame(frame.copy(), [])
        assert out.shape == frame.shape

    def test_annotate_with_detection(self):
        from lib.nav.apriltag_nav import AprilTagNavigator

        frame = _dummy_frame()
        det = {
            "id": 0,
            "center": (320, 240),
            "corners": [(270, 190), (370, 190), (370, 290), (270, 290)],
        }
        out = AprilTagNavigator._annotate_frame(frame.copy(), [det])
        assert out.shape == frame.shape
        assert not np.array_equal(out, frame)

    def test_annotate_shows_searching_when_empty(self):
        from lib.nav.apriltag_nav import AprilTagNavigator

        frame = _dummy_frame()
        out = AprilTagNavigator._annotate_frame(frame.copy(), [])
        # Frame should be modified (SEARCHING text drawn)
        assert not np.array_equal(out, frame)
