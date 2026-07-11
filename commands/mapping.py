"""
Argparse registration and dispatch for the 'map' command group.

All lib imports are deferred to dispatch functions so that torch/numpy/orbslam2
are never imported during argument parsing.
"""

import argparse

from lib.settings import NanoSettings


def register(parser: argparse.ArgumentParser, settings: NanoSettings):
    """Attach sub-commands to the 'map' argument parser."""
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # odometry
    p_odom = sub.add_parser(
        "odometry",
        help="Monocular visual odometry using ORB/SIFT/AKAZE feature tracking",
    )
    p_odom.add_argument(
        "--config",
        default=str(settings.odometry_config_path),
        dest="config_path",
        metavar="YAML",
        help="Path to odometry config (default: config/models/odometry.yaml)",
    )
    p_odom.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        default=True,
        help="Disable the background MJPEG camera stream",
    )
    p_odom.add_argument(
        "--stream-port",
        type=int,
        default=settings.stream_port,
        dest="stream_port",
        metavar="PORT",
        help="MJPEG server port (default: 8080)",
    )
    p_odom.set_defaults(func=_run_odometry)

    # slam
    p_slam = sub.add_parser(
        "slam",
        help="[Experimental] Minimal monocular SLAM (ORB-SLAM2 or RTAB-Map)",
    )
    p_slam.add_argument(
        "--config",
        default=str(settings.slam_config_path),
        dest="config_path",
        metavar="YAML",
        help="Path to SLAM config (default: config/models/slam.yaml)",
    )
    p_slam.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        default=True,
        help="Disable the background MJPEG camera stream",
    )
    p_slam.add_argument(
        "--stream-port",
        type=int,
        default=settings.stream_port,
        dest="stream_port",
        metavar="PORT",
        help="MJPEG server port (default: 8080)",
    )
    p_slam.add_argument(
        "--teleop",
        action="store_true",
        default=False,
        help="Enable arrow-key motor control while mapping",
    )
    p_slam.add_argument(
        "--speed",
        type=float,
        default=0.3,
        metavar="SPEED",
        help="Motor speed 0.0-1.0 (default: 0.3)",
    )
    p_slam.add_argument(
        "--turn-gain",
        type=float,
        default=0.5,
        dest="turn_gain",
        metavar="GAIN",
        help="Turn gain 0.0-1.0 (default: 0.5)",
    )
    p_slam.set_defaults(func=_run_slam)


# Dispatch functions


def _run_odometry(args):
    from lib.map.odometry import OdometryConfig, VisualOdometry

    config = OdometryConfig(**vars(args))
    VisualOdometry(**config.dict()).run()


def _run_slam(args):
    from lib.map.slam import SlamConfig, SlamMapper

    config = SlamConfig(**vars(args))
    SlamMapper(**config.dict()).run()
