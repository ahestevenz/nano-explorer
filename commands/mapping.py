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
    p_odom.add_argument(
        "--speed",
        type=float,
        default=0.3,
        metavar="SPEED",
        help="Motor speed 0.0-1.0 (default: 0.3)",
    )
    p_odom.add_argument(
        "--turn-gain",
        type=float,
        default=0.5,
        dest="turn_gain",
        metavar="GAIN",
        help="Turn gain 0.0-1.0 (default: 0.5)",
    )
    p_odom.set_defaults(func=_run_odometry)

    # slam
    p_slam = sub.add_parser(
        "slam",
        help="[Experimental] Minimal monocular SLAM (ORB-SLAM2)",
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
        "--speed",
        type=float,
        default=0.2,
        metavar="SPEED",
        help="Motor speed 0.0-1.0 (default: 0.2)",
    )
    p_slam.add_argument(
        "--turn-gain",
        type=float,
        default=0.5,
        dest="turn_gain",
        metavar="GAIN",
        help="Turn gain 0.0-1.0 (default: 0.5)",
    )
    p_slam.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Show a debug view in the stream: trajectory map + detected ORB "
        "keypoints on top, frame-to-frame matches below (green=raw, "
        "blue=surviving RANSAC)",
    )
    p_slam.add_argument(
        "--no-auto-recover",
        action="store_false",
        dest="auto_recover",
        default=True,
        help="Don't reverse your last moves automatically when tracking is lost",
    )
    p_slam.add_argument(
        "--max-exposure-ms",
        type=float,
        default=None,
        dest="max_exposure_ms",
        metavar="MS",
        help="Cap the camera's auto-exposure time to reduce motion blur, e.g. 10 "
        "(default: camera's own range). Needs good light: frames get darker and noisier",
    )
    p_slam.set_defaults(func=_run_slam)

    # imu-check
    p_imu = sub.add_parser(
        "imu-check",
        help="Check the BNO055 IMU readings and the reader's sampling timing",
    )
    p_imu.add_argument(
        "-t",
        "--seconds",
        type=float,
        default=10.0,
        metavar="S",
        help="How long to run (default: 10)",
    )
    p_imu.add_argument(
        "--rate",
        type=float,
        default=200.0,
        dest="rate_hz",
        metavar="HZ",
        help="Polling rate in Hz (default: 200)",
    )
    p_imu.add_argument(
        "--bus",
        type=int,
        default=0,
        metavar="N",
        help="I2C bus the IMU is on (default: 0)",
    )
    p_imu.add_argument(
        "--load",
        action="store_true",
        default=False,
        help="Keep the other CPU cores busy, to measure the jitter under a SLAM-like load",
    )
    p_imu.set_defaults(func=_run_imu_check)


# Dispatch functions


def _run_odometry(args):
    from lib.map.odometry import OdometryConfig, VisualOdometry

    config = OdometryConfig(**vars(args))
    VisualOdometry(**config.dict()).run()


def _run_slam(args):
    from lib.map.slam import SlamConfig, SlamMapper

    config = SlamConfig(**vars(args))
    SlamMapper(**config.dict()).run()


def _run_imu_check(args):
    from lib.imu import ImuCheck, ImuCheckConfig

    config = ImuCheckConfig(**vars(args))
    ImuCheck(**config.dict()).run()
