"""
Argparse registration and dispatch for the 'motion' command group.

All lib imports are deferred to the dispatch functions (_run_*) so that
torch/numpy are never imported during argument parsing — only when the
user actually runs a command.
"""

import argparse

from lib.settings import NanoSettings


def register(parser: argparse.ArgumentParser, settings: NanoSettings):
    """Attach sub-commands to the 'motion' argument parser."""
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # teleop
    p_teleop = sub.add_parser(
        "teleop",
        help="Drive the robot with keyboard arrow keys over Wi-Fi",
    )
    p_teleop.add_argument(
        "--speed",
        type=float,
        default=settings.default_speed,
        metavar="SPEED",
        help="Motor speed 0.0-1.0 (default: 0.3)",
    )
    p_teleop.add_argument(
        "--turn-gain",
        type=float,
        default=settings.default_turn_gain,
        dest="turn_gain",
        metavar="GAIN",
        help="Differential turn gain 0.0-1.0 (default: 0.5)",
    )
    p_teleop.add_argument(
        "--mode",
        choices=["auto", "arrows", "pynput", "stdin"],
        default="arrows",
        help=(
            "Input mode: "
            "arrows = raw terminal arrow keys, hold to move (default/recommended over SSH); "
            "pynput = needs X or uinput; "
            "stdin = type commands + ENTER; "
            "auto = try arrows -> pynput -> stdin"
        ),
    )
    p_teleop.add_argument(
        "--stream",
        action="store_true",
        help="Start a background MJPEG camera stream while driving "
        "(open http://<nano-ip>:8080/stream in your browser)",
    )
    p_teleop.add_argument(
        "--stream-port",
        type=int,
        default=settings.stream_port,
        dest="stream_port",
        metavar="PORT",
        help="MJPEG server port (default: 8080)",
    )
    p_teleop.set_defaults(func=_run_teleop)

    # stream
    p_stream = sub.add_parser("stream", help="Stream the camera over Wi-Fi")
    p_stream.add_argument(
        "--mode",
        choices=["mjpeg", "opencv"],
        default="mjpeg",
        help="Streaming mode (default: mjpeg)",
    )
    p_stream.add_argument(
        "--port",
        type=int,
        default=settings.stream_port,
        metavar="PORT",
        help="HTTP port (default: 8080)",
    )
    p_stream.add_argument("--width", type=int, default=640, metavar="W")
    p_stream.add_argument("--height", type=int, default=480, metavar="H")
    p_stream.add_argument("--fps", type=int, default=30, metavar="FPS")
    p_stream.set_defaults(func=_run_stream)

    # collision
    p_col = sub.add_parser(
        "collision",
        help="Run collision avoidance using a trained binary CNN",
    )
    p_col.add_argument(
        "--model",
        default=settings.collision_model_path,
        metavar="PATH",
        help="Path to collision avoidance model (.pth or .engine)",
    )
    p_col.add_argument(
        "--threshold",
        type=float,
        default=settings.collision_threshold,
        metavar="T",
        help="Blocked probability threshold (default: 0.5)",
    )
    p_col.add_argument(
        "--speed",
        type=float,
        default=settings.default_speed,
        metavar="SPEED",
        help="Forward motor speed (default: 0.3)",
    )
    p_col.add_argument(
        "--stream",
        action="store_true",
        help="Start a background MJPEG camera stream while running "
        "(open http://<nano-ip>:8080/ in your browser)",
    )
    p_col.add_argument(
        "--stream-port",
        type=int,
        default=settings.stream_port,
        dest="stream_port",
        metavar="PORT",
        help="MJPEG server port (default: 8080)",
    )
    p_col.set_defaults(func=_run_collision)


# Dispatch functions
# Imports happen HERE, not at module level, so torch/numpy are only loaded
# when the user explicitly runs one of these commands.


def _run_teleop(args):
    from lib.motion.teleoperation import TeleopConfig, TeleopController

    config = TeleopConfig(**vars(args))
    TeleopController(**config.dict()).run()


def _run_stream(args):
    from lib.motion.stream import CameraStreamer, CameraStreamerConfig

    config = CameraStreamerConfig(**vars(args))
    CameraStreamer(**config.dict()).run()


def _run_collision(args):
    from lib.motion.collision import CollisionAvoider, CollisionConfig

    config = CollisionConfig(**vars(args))
    CollisionAvoider(**config.dict()).run()
