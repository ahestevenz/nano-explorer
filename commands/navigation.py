"""
Argparse registration and dispatch for the 'nav' command group.

All lib imports are deferred to the dispatch functions (_run_*) so that
torch/numpy are never imported during argument parsing — only when the
user actually runs a command.
"""

import argparse

from lib.settings import NanoSettings


def register(parser: argparse.ArgumentParser, settings: NanoSettings):
    """Attach sub-commands to the 'nav' argument parser."""
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # line-follow
    p_line = sub.add_parser(
        "line-follow",
        help="Follow a coloured line on the floor (classical HSV or CNN)",
    )
    p_line.add_argument(
        "--config",
        default=str(settings.line_follow_config_path),
        dest="config_path",
        metavar="YAML",
        help="Path to line follower config (default: config/models/line_follow.yaml)",
    )
    p_line.add_argument(
        "--speed",
        type=float,
        default=settings.default_speed,
        metavar="SPEED",
        help="Forward motor speed 0.0–1.0 (default: 0.3)",
    )
    p_line.add_argument(
        "--turn-gain",
        type=float,
        default=settings.default_turn_gain,
        dest="turn_gain",
        metavar="GAIN",
        help="Differential turn gain 0.0–1.0 (default: 0.5)",
    )
    p_line.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        default=True,
        help="Disable the background MJPEG camera stream",
    )
    p_line.add_argument(
        "--stream-port",
        type=int,
        default=settings.stream_port,
        dest="stream_port",
        metavar="PORT",
        help="MJPEG server port (default: 8080)",
    )
    p_line.set_defaults(func=_run_line_follow)

    # road-follow
    p_road = sub.add_parser(
        "road-follow",
        help="Follow a road using a regression CNN (steering angle output)",
    )
    p_road.add_argument(
        "--model",
        default=str(settings.road_follow_model_path),
        dest="model_path",
        metavar="PATH",
        help="Path to trained .pth or .engine regression model",
    )
    p_road.add_argument(
        "--speed",
        type=float,
        default=settings.default_speed,
        metavar="SPEED",
        help="Constant forward speed 0.0–1.0 (default: 0.3)",
    )
    p_road.add_argument(
        "--turn-gain",
        type=float,
        default=settings.default_turn_gain,
        dest="turn_gain",
        metavar="GAIN",
        help="Steering gain scale 0.0–1.0 (default: 0.5)",
    )
    p_road.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        default=True,
        help="Disable the background MJPEG camera stream",
    )
    p_road.add_argument(
        "--stream-port",
        type=int,
        default=settings.stream_port,
        dest="stream_port",
        metavar="PORT",
        help="MJPEG server port (default: 8080)",
    )
    p_road.set_defaults(func=_run_road_follow)

    # apriltag
    p_april = sub.add_parser(
        "apriltag",
        help="Navigate by detecting AprilTag / ArUco fiducial markers",
    )
    p_april.add_argument(
        "--config",
        default=str(settings.apriltag_config_path),
        dest="config_path",
        metavar="YAML",
        help="Path to AprilTag config (default: config/models/apriltag.yaml)",
    )
    p_april.add_argument(
        "--speed",
        type=float,
        default=settings.default_speed,
        metavar="SPEED",
        help="Motor speed 0.0–1.0 (default: 0.3)",
    )
    p_april.add_argument(
        "--turn-gain",
        type=float,
        default=settings.default_turn_gain,
        dest="turn_gain",
        metavar="GAIN",
        help="Steering gain scale 0.0–1.0 (default: 0.5)",
    )
    p_april.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        default=True,
        help="Disable the background MJPEG camera stream",
    )
    p_april.add_argument(
        "--stream-port",
        type=int,
        default=settings.stream_port,
        dest="stream_port",
        metavar="PORT",
        help="MJPEG server port (default: 8080)",
    )
    p_april.set_defaults(func=_run_apriltag)


# Dispatch functions


def _run_line_follow(args):
    from lib.nav.line_follower import LineFollower, LineFollowerConfig

    config = LineFollowerConfig(**vars(args))
    LineFollower(**config.dict()).run()


def _run_road_follow(args):
    from lib.nav.road_follower import RoadFollower, RoadFollowerConfig

    config = RoadFollowerConfig(**vars(args))
    RoadFollower(**config.dict()).run()


def _run_apriltag(args):
    from lib.nav.apriltag_nav import AprilTagNavConfig, AprilTagNavigator

    config = AprilTagNavConfig(**vars(args))
    AprilTagNavigator(**config.dict()).run()
