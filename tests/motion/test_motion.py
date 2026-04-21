"""
Tests for commands/motion.py — argument parsing and dispatch.

Run with:
    pytest tests/test_motion.py -v
    pytest tests/test_motion.py -v -m "not hardware"

Hardware-dependent tests (marked with @pytest.mark.hardware) are skipped
automatically on CI via:
    pytest -m "not hardware"
"""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from lib.settings import NanoSettings


# Helpers
def _make_parser(settings: NanoSettings = None) -> argparse.ArgumentParser:
    """Build a top-level parser with the motion sub-commands registered."""
    from commands.motion import register

    if settings is None:
        settings = NanoSettings()

    parser = argparse.ArgumentParser(prog="nano-explorer")
    sub = parser.add_subparsers(dest="group")
    motion_parser = sub.add_parser("motion")
    register(motion_parser, settings)
    return parser


def _parse(args: list, settings: NanoSettings = None) -> argparse.Namespace:
    return _make_parser(settings).parse_args(["motion"] + args)


# register() — sub-command structure
class TestRegister:
    def test_teleop_subcommand_exists(self):
        ns = _parse(["teleop"])
        assert ns.command == "teleop"

    def test_stream_subcommand_exists(self):
        ns = _parse(["stream"])
        assert ns.command == "stream"

    def test_collision_subcommand_exists(self):
        ns = _parse(["collision", "--model", "assets/models/fake.pth"])
        assert ns.command == "collision"

    def test_missing_subcommand_exits(self):
        with pytest.raises(SystemExit):
            _make_parser().parse_args(["motion"])


# teleop sub-command defaults
class TestTeleopDefaults:
    def test_default_speed_from_settings(self):
        settings = NanoSettings()
        ns = _parse(["teleop"], settings)
        assert ns.speed == settings.default_speed

    def test_default_turn_gain_from_settings(self):
        settings = NanoSettings()
        ns = _parse(["teleop"], settings)
        assert ns.turn_gain == settings.default_turn_gain

    def test_default_mode_is_arrows(self):
        ns = _parse(["teleop"])
        assert ns.mode == "arrows"

    def test_default_stream_is_false(self):
        ns = _parse(["teleop"])
        assert ns.stream is False

    def test_default_stream_port_from_settings(self):
        settings = NanoSettings()
        ns = _parse(["teleop"], settings)
        assert ns.stream_port == settings.stream_port

    def test_func_is_set(self):
        from commands.motion import _run_teleop

        ns = _parse(["teleop"])
        assert ns.func is _run_teleop


class TestTeleopArgs:
    def test_custom_speed(self):
        ns = _parse(["teleop", "--speed", "0.7"])
        assert ns.speed == pytest.approx(0.7)

    def test_custom_turn_gain(self):
        ns = _parse(["teleop", "--turn-gain", "0.8"])
        assert ns.turn_gain == pytest.approx(0.8)

    def test_stream_flag(self):
        ns = _parse(["teleop", "--stream"])
        assert ns.stream is True

    def test_custom_stream_port(self):
        ns = _parse(["teleop", "--stream-port", "9090"])
        assert ns.stream_port == 9090

    @pytest.mark.parametrize("mode", ["auto", "arrows", "pynput", "stdin"])
    def test_valid_modes(self, mode):
        ns = _parse(["teleop", "--mode", mode])
        assert ns.mode == mode

    def test_invalid_mode_exits(self):
        with pytest.raises(SystemExit):
            _parse(["teleop", "--mode", "joystick"])


# stream sub-command defaults
class TestStreamDefaults:
    def test_default_mode_is_mjpeg(self):
        ns = _parse(["stream"])
        assert ns.mode == "mjpeg"

    def test_default_port_from_settings(self):
        settings = NanoSettings()
        ns = _parse(["stream"], settings)
        assert ns.port == settings.stream_port

    def test_default_width(self):
        ns = _parse(["stream"])
        assert ns.width == 640

    def test_default_height(self):
        ns = _parse(["stream"])
        assert ns.height == 480

    def test_default_fps(self):
        ns = _parse(["stream"])
        assert ns.fps == 30

    def test_func_is_set(self):
        from commands.motion import _run_stream

        ns = _parse(["stream"])
        assert ns.func is _run_stream


class TestStreamArgs:
    @pytest.mark.parametrize("mode", ["mjpeg", "opencv"])
    def test_valid_modes(self, mode):
        ns = _parse(["stream", "--mode", mode])
        assert ns.mode == mode

    def test_invalid_mode_exits(self):
        with pytest.raises(SystemExit):
            _parse(["stream", "--mode", "rtsp"])

    def test_custom_port(self):
        ns = _parse(["stream", "--port", "8888"])
        assert ns.port == 8888

    def test_custom_resolution(self):
        ns = _parse(["stream", "--width", "1280", "--height", "720"])
        assert ns.width == 1280
        assert ns.height == 720

    def test_custom_fps(self):
        ns = _parse(["stream", "--fps", "15"])
        assert ns.fps == 15


# collision sub-command defaults
class TestCollisionDefaults:
    def test_default_model_from_settings(self):
        settings = NanoSettings()
        _ = _parse(["collision", "--model", "assets/models/fake.pth"], settings)
        # explicit --model wins; test the settings default is wired up correctly
        assert settings.collision_model_path == "assets/models/collision_avoidance.pth"

    def test_default_threshold_from_settings(self):
        settings = NanoSettings()
        ns = _parse(["collision", "--model", "assets/models/fake.pth"], settings)
        assert ns.threshold == settings.collision_threshold

    def test_default_speed_from_settings(self):
        settings = NanoSettings()
        ns = _parse(["collision", "--model", "assets/models/fake.pth"], settings)
        assert ns.speed == settings.default_speed

    def test_default_stream_is_false(self):
        ns = _parse(["collision", "--model", "assets/models/fake.pth"])
        assert ns.stream is False

    def test_func_is_set(self):
        from commands.motion import _run_collision

        ns = _parse(["collision", "--model", "assets/models/fake.pth"])
        assert ns.func is _run_collision


class TestCollisionArgs:
    def test_custom_threshold(self):
        ns = _parse(["collision", "--model", "m.pth", "--threshold", "0.8"])
        assert ns.threshold == pytest.approx(0.8)

    def test_custom_speed(self):
        ns = _parse(["collision", "--model", "m.pth", "--speed", "0.6"])
        assert ns.speed == pytest.approx(0.6)

    def test_stream_flag(self):
        ns = _parse(["collision", "--model", "m.pth", "--stream"])
        assert ns.stream is True

    def test_custom_stream_port(self):
        ns = _parse(["collision", "--model", "m.pth", "--stream-port", "9999"])
        assert ns.stream_port == 9999


# Dispatch — _run_* functions (no hardware required)
class TestRunTeleop:
    @patch("lib.motion.teleoperation.TeleopController")
    @patch("lib.motion.teleoperation.TeleopConfig")
    def test_run_teleop_creates_config_and_runs(self, mock_config_cls, mock_controller_cls):
        from commands.motion import _run_teleop

        mock_config = MagicMock()
        mock_config.dict.return_value = {"speed": 0.3, "turn_gain": 0.5, "mode": "arrows"}
        mock_config_cls.return_value = mock_config

        mock_controller_cls.return_value = MagicMock()

        args = argparse.Namespace(
            speed=0.3,
            turn_gain=0.5,
            mode="arrows",
            stream=False,
            stream_port=8080,
            func=_run_teleop,
            command="teleop",
        )
        _run_teleop(args)

        mock_config_cls.assert_called_once()
        mock_controller_cls.assert_called_once()
        mock_controller_cls.return_value.run.assert_called_once()


class TestRunStream:
    @patch("lib.motion.stream.CameraStreamer")
    @patch("lib.motion.stream.CameraStreamerConfig")
    def test_run_stream_creates_config_and_runs(self, mock_config_cls, mock_streamer_cls):
        from commands.motion import _run_stream

        mock_config = MagicMock()
        mock_config.dict.return_value = {
            "mode": "mjpeg",
            "port": 8080,
            "width": 640,
            "height": 480,
            "fps": 30,
        }
        mock_config_cls.return_value = mock_config

        mock_streamer_cls.return_value = MagicMock()

        args = argparse.Namespace(
            mode="mjpeg",
            port=8080,
            width=640,
            height=480,
            fps=30,
            func=_run_stream,
            command="stream",
        )
        _run_stream(args)

        mock_config_cls.assert_called_once()
        mock_streamer_cls.assert_called_once()
        mock_streamer_cls.return_value.run.assert_called_once()


class TestRunCollision:
    @patch("lib.motion.collision.CollisionAvoider")
    @patch("lib.motion.collision.CollisionConfig")
    def test_run_collision_creates_config_and_runs(self, mock_config_cls, mock_avoider_cls):
        from commands.motion import _run_collision

        mock_config = MagicMock()
        mock_config.dict.return_value = {
            "model_path": "m.pth",
            "threshold": 0.5,
            "speed": 0.3,
            "stream": False,
            "stream_port": 8080,
        }
        mock_config_cls.return_value = mock_config

        mock_avoider_cls.return_value = MagicMock()

        args = argparse.Namespace(
            model="m.pth",
            threshold=0.5,
            speed=0.3,
            stream=False,
            stream_port=8080,
            func=_run_collision,
            command="collision",
        )
        _run_collision(args)

        mock_config_cls.assert_called_once()
        mock_avoider_cls.assert_called_once()
        mock_avoider_cls.return_value.run.assert_called_once()
