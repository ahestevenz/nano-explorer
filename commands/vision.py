"""
Argparse registration and dispatch for the 'vision' command group.

All lib imports are deferred to dispatch functions so that torch/numpy
are never imported during argument parsing.
"""

import argparse

from lib.settings import NanoSettings


def register(parser: argparse.ArgumentParser, settings: NanoSettings):
    """Attach vision sub-commands to the parser."""
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # detect
    p_det = sub.add_parser("detect", help="Real-time object detection")
    p_det.add_argument(
        "--config",
        default=str(settings.detection_config_path),
        dest="config_path",
        metavar="YAML",
        help="Path to detection model config (default: config/models/detection.yaml)",
    )
    p_det.add_argument(
        "--threshold", type=float, default=0.5, metavar="T",
        help="Confidence threshold override (default: 0.5)",
    )
    p_det.add_argument("--stream", action="store_true",
                       help="Serve annotated MJPEG stream")
    p_det.add_argument("--stream-port", type=int, default=settings.stream_port,
                       dest="stream_port", metavar="PORT")
    p_det.set_defaults(func=_run_detect)

    # faces
    p_face = sub.add_parser("faces", help="Face and people detection")
    p_face.add_argument(
        "--config",
        default=str(settings.face_config_path),
        dest="config_path",
        metavar="YAML",
    )
    p_face.add_argument("--stream", action="store_true",
                        help="Serve annotated MJPEG stream")
    p_face.add_argument("--stream-port", type=int, default=settings.stream_port,
                        dest="stream_port", metavar="PORT")
    p_face.set_defaults(func=_run_faces)

    # track
    p_track = sub.add_parser(
        "track", help="Track objects, colours or blobs and steer the robot"
    )
    p_track.add_argument(
        "--mode", choices=["color", "blob", "object"], default="color",
        help="Tracking mode (default: color)",
    )
    p_track.add_argument(
        "--color", default="red",
        choices=["red", "green", "blue", "yellow", "orange"],
        help="Target colour (mode=color only, default: red)",
    )
    p_track.add_argument(
        "--label", default="person", metavar="CLASS",
        help="COCO class label (mode=object only, default: person)",
    )
    p_track.add_argument(
        "--speed", type=float, default=settings.default_speed, metavar="SPEED",
    )
    p_track.add_argument("--stream", action="store_true")
    p_track.add_argument("--stream-port", type=int, default=settings.stream_port,
                         dest="stream_port", metavar="PORT")
    p_track.set_defaults(func=_run_track)

    # segment
    p_seg = sub.add_parser("segment", help="Semantic segmentation")
    p_seg.add_argument(
        "--config",
        default=str(settings.segmentation_config_path),
        dest="config_path",
        metavar="YAML",
    )
    p_seg.add_argument("--stream", action="store_true")
    p_seg.add_argument("--stream-port", type=int, default=settings.stream_port,
                       dest="stream_port", metavar="PORT")
    p_seg.set_defaults(func=_run_segment)

    # pose
    p_pose = sub.add_parser("pose", help="Human pose estimation (trt_pose)")
    p_pose.add_argument(
        "--config",
        default=str(settings.pose_config_path),
        dest="config_path",
        metavar="YAML",
    )
    p_pose.add_argument("--stream", action="store_true")
    p_pose.add_argument("--stream-port", type=int, default=settings.stream_port,
                        dest="stream_port", metavar="PORT")
    p_pose.set_defaults(func=_run_pose)

    # gesture 
    p_gest = sub.add_parser(
        "gesture", help="Control the robot with hand/body gestures (trt_pose)"
    )
    p_gest.add_argument(
        "--config",
        default=str(settings.pose_config_path),
        dest="config_path",
        metavar="YAML",
    )
    p_gest.add_argument(
        "--speed", type=float, default=settings.default_speed, metavar="SPEED",
    )
    p_gest.add_argument("--stream", action="store_true")
    p_gest.add_argument("--stream-port", type=int, default=settings.stream_port,
                        dest="stream_port", metavar="PORT")
    p_gest.set_defaults(func=_run_gesture)


# Dispatch functions

def _run_detect(args):
    from lib.vision.detector import DetectionConfig, ObjectDetector
    config = DetectionConfig(**vars(args))
    ObjectDetector(**config.dict()).run()


def _run_faces(args):
    from lib.vision.face_detector import FaceDetectionConfig, FaceDetector
    config = FaceDetectionConfig(**vars(args))
    FaceDetector(**config.dict()).run()


def _run_track(args):
    from lib.vision.tracker import ObjectTracker, TrackerConfig
    config = TrackerConfig(**vars(args))
    ObjectTracker(**config.dict()).run()


def _run_segment(args):
    from lib.vision.segmentation import SegmentationConfig, Segmenter
    config = SegmentationConfig(**vars(args))
    Segmenter(**config.dict()).run()


def _run_pose(args):
    from lib.vision.pose import PoseConfig, PoseEstimator
    config = PoseConfig(**vars(args))
    PoseEstimator(**config.dict()).run()


def _run_gesture(args):
    from lib.vision.gesture import GestureConfig, GestureController
    config = GestureConfig(**vars(args))
    GestureController(**config.dict()).run()
