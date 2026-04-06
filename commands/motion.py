"""
Argparse registration and dispatch for the 'motion' command group.

All lib imports are deferred to the dispatch functions (_run_*) so that
torch/numpy are never imported during argument parsing — only when the
user actually runs a command.
"""


def register(parser):
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
        default=0.3,
        metavar="SPEED",
        help="Motor speed 0.0-1.0 (default: 0.3)",
    )
    p_teleop.add_argument(
        "--turn-gain",
        type=float,
        default=0.5,
        dest="turn_gain",
        metavar="GAIN",
        help="Differential turn gain 0.0-1.0 (default: 0.5)",
    )
    p_teleop.add_argument(
        "--mode",
        choices=["auto", "arrows", "pynput", "stdin"],
        default="auto",
        help=(
            "Input mode: "
            "arrows = raw terminal arrow keys, hold to move (default/recommended over SSH); "
            "pynput = needs X or uinput; "
            "stdin = type commands + ENTER; "
            "auto = try arrows -> pynput -> stdin"
        ),
    )
    p_teleop.set_defaults(func=_run_teleop)

    # stream
    p_stream = sub.add_parser("stream", help="Stream the camera over Wi-Fi")
    p_stream.add_argument(
        "--mode",
        choices=["mjpeg", "opencv", "webrtc"],
        default="mjpeg",
        help="Streaming mode (default: mjpeg)",
    )
    p_stream.add_argument(
        "--port",
        type=int,
        default=8080,
        metavar="PORT",
        help="HTTP/WebRTC port (default: 8080)",
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
        default="assets/models/collision_avoidance.pth",
        metavar="PATH",
        help="Path to collision avoidance model (.pth or .engine)",
    )
    p_col.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        metavar="T",
        help="Blocked probability threshold (default: 0.5)",
    )
    p_col.add_argument(
        "--speed",
        type=float,
        default=0.3,
        metavar="SPEED",
        help="Forward motor speed (default: 0.3)",
    )
    p_col.set_defaults(func=_run_collision)


# Dispatch functions
# Imports happen HERE, not at module level, so torch/numpy are only loaded
# when the user explicitly runs one of these commands.


def _run_teleop(args):
    from lib.motion.teleoperation import TeleopController

    TeleopController(
        speed=args.speed,
        turn_gain=args.turn_gain,
        mode=args.mode,
    ).run()


def _run_stream(args):
    from lib.motion.stream import CameraStreamer

    CameraStreamer(
        mode=args.mode,
        port=args.port,
        width=args.width,
        height=args.height,
        fps=args.fps,
    ).run()


def _run_collision(args):
    from lib.motion.collision import CollisionAvoider

    CollisionAvoider(
        model_path=args.model,
        threshold=args.threshold,
        speed=args.speed,
    ).run()
