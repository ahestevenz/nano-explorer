"""
Tests for commands/vision.py and lib/vision/* — argument parsing,
config validation, and algorithm logic.

Run with:
    pytest tests/vision/ -v
    pytest tests/vision/ -v -m "not hardware"

Hardware-dependent tests (marked @pytest.mark.hardware) are skipped on CI.
"""

import argparse
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

from lib.settings import NanoSettings, PROJECT_ROOT_PATH


def _make_parser(settings: NanoSettings = None) -> argparse.ArgumentParser:
    from commands.vision import register

    if settings is None:
        settings = NanoSettings()
    parser = argparse.ArgumentParser(prog="nano-explorer")
    sub = parser.add_subparsers(dest="group")
    vision_parser = sub.add_parser("vision")
    register(vision_parser, settings)
    return parser


def _parse(args: list, settings: NanoSettings = None) -> argparse.Namespace:
    return _make_parser(settings).parse_args(["vision"] + args)


def _dummy_frame(h=480, w=640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# commands/vision.py — sub-command structure 
class TestRegister:
    def test_detect_subcommand_exists(self):
        ns = _parse(["detect"])
        assert ns.command == "detect"

    def test_faces_subcommand_exists(self):
        ns = _parse(["faces"])
        assert ns.command == "faces"

    def test_track_subcommand_exists(self):
        ns = _parse(["track"])
        assert ns.command == "track"

    def test_segment_subcommand_exists(self):
        ns = _parse(["segment"])
        assert ns.command == "segment"

    def test_pose_subcommand_exists(self):
        ns = _parse(["pose"])
        assert ns.command == "pose"

    def test_gesture_subcommand_exists(self):
        ns = _parse(["gesture"])
        assert ns.command == "gesture"

    def test_missing_subcommand_exits(self):
        with pytest.raises(SystemExit):
            _make_parser().parse_args(["vision"])


#  detect defaults
class TestDetectDefaults:
    def test_default_threshold(self):
        ns = _parse(["detect"])
        assert ns.threshold == 0.5

    def test_default_stream_is_false(self):
        ns = _parse(["detect"])
        assert ns.stream is False

    def test_stream_port_from_settings(self):
        settings = NanoSettings()
        ns = _parse(["detect"], settings)
        assert ns.stream_port == settings.stream_port

    def test_threshold_override(self):
        ns = _parse(["detect", "--threshold", "0.75"])
        assert ns.threshold == 0.75


# track defaults
class TestTrackDefaults:
    def test_default_mode_is_color(self):
        ns = _parse(["track"])
        assert ns.mode == "color"

    def test_default_color_is_red(self):
        ns = _parse(["track"])
        assert ns.color == "red"

    def test_mode_blob(self):
        ns = _parse(["track", "--mode", "blob"])
        assert ns.mode == "blob"

    def test_invalid_mode_exits(self):
        with pytest.raises(SystemExit):
            _parse(["track", "--mode", "invalid"])

    def test_invalid_color_exits(self):
        with pytest.raises(SystemExit):
            _parse(["track", "--color", "purple"])


# lib/vision/tracker.py
class TestObjectTracker:
    def test_import(self):
        from lib.vision.tracker import ObjectTracker
        assert ObjectTracker is not None

    def test_red_centroid_in_red_frame(self):
        """A pure-red frame should yield a centroid near the horizontal centre."""
        from lib.vision.tracker import ObjectTracker, TrackerConfig
        tracker = ObjectTracker(**TrackerConfig(mode="color", color="red").dict())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 2] = 200   # high R in BGR → red in HSV
        centroid, _ = tracker._find_color_centroid(frame)
        assert centroid is not None
        cx, _ = centroid
        assert 200 < cx < 440

    def test_blank_frame_returns_none(self):
        from lib.vision.tracker import ObjectTracker, TrackerConfig
        tracker = ObjectTracker(**TrackerConfig(mode="color", color="red").dict())
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        centroid, _ = tracker._find_color_centroid(frame)
        assert centroid is None

    def test_annotate_runs_without_centroid(self):
        from lib.vision.tracker import ObjectTracker
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = ObjectTracker.annotate(frame.copy(), None)
        assert out.shape == frame.shape

    def test_annotate_runs_with_centroid(self):
        from lib.vision.tracker import ObjectTracker
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = ObjectTracker.annotate(frame.copy(), (320, 240))
        assert out.shape == frame.shape

    def test_left_stripe_gives_negative_error(self):
        """A target on the left should produce a negative steering error."""
        from lib.vision.tracker import ObjectTracker, TrackerConfig, _Kp
        tracker = ObjectTracker(**TrackerConfig(mode="color", color="yellow").dict())
        # Yellow stripe on the left side of lower half
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[240:, 10:80] = [0, 255, 255]   # BGR yellow
        centroid, _ = tracker._find_color_centroid(frame)
        if centroid is not None:
            cx, _ = centroid
            error = (cx - 320) / 320
            assert error < 0, f"Left stripe should give negative error, got {error:.3f}"

    def test_right_stripe_gives_positive_error(self):
        from lib.vision.tracker import ObjectTracker, TrackerConfig
        tracker = ObjectTracker(**TrackerConfig(mode="color", color="yellow").dict())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[240:, 560:630] = [0, 255, 255]
        centroid, _ = tracker._find_color_centroid(frame)
        if centroid is not None:
            cx, _ = centroid
            error = (cx - 320) / 320
            assert error > 0


# lib/vision/detector.py
class TestObjectDetector:
    def test_import(self):
        from lib.vision.detector import ObjectDetector
        assert ObjectDetector is not None

    def test_annotate_empty(self):
        from lib.vision.detector import ObjectDetector
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = ObjectDetector.annotate(frame.copy(), [])
        assert np.array_equal(out, frame)

    def test_annotate_draws_box(self):
        from lib.vision.detector import ObjectDetector
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [{"label": "person", "conf": 0.9, "bbox": (10, 10, 200, 300)}]
        out = ObjectDetector.annotate(frame.copy(), dets)
        assert not np.array_equal(out, frame)

    def test_missing_config_raises(self, tmp_path):
        from lib.vision.detector import DetectionConfig
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DetectionConfig(config_path=str(tmp_path / "missing.yaml"))

    def test_threshold_default(self):
        from lib.vision.detector import DetectionConfig
        cfg = DetectionConfig(
            config_path=str(PROJECT_ROOT_PATH / "config/models/detection.yaml")
        )
        assert cfg.threshold == 0.5


# lib/vision/face_detector.py
class TestFaceDetector:
    def test_import(self):
        from lib.vision.face_detector import FaceDetector
        assert FaceDetector is not None

    def test_annotate_no_detections(self):
        from lib.vision.face_detector import FaceDetector
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = FaceDetector.annotate(frame.copy(), [])
        assert np.array_equal(out, frame)

    def test_annotate_face(self):
        from lib.vision.face_detector import FaceDetector
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [{"label": "face", "bbox": (50, 50, 200, 200)}]
        out = FaceDetector.annotate(frame.copy(), dets)
        assert not np.array_equal(out, frame)

    def test_missing_config_raises(self, tmp_path):
        from lib.vision.face_detector import FaceDetectionConfig
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            FaceDetectionConfig(config_path=str(tmp_path / "missing.yaml"))


# lib/vision/segmentation.py
class TestSegmenter:
    def test_import(self):
        from lib.vision.segmentation import Segmenter
        assert Segmenter is not None

    def test_missing_config_raises(self, tmp_path):
        from lib.vision.segmentation import SegmentationConfig
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SegmentationConfig(config_path=str(tmp_path / "missing.yaml"))


# lib/vision/pose.py
class TestPoseEstimator:
    def test_import(self):
        from lib.vision.pose import PoseEstimator
        assert PoseEstimator is not None

    def test_missing_config_raises(self, tmp_path):
        from lib.vision.pose import PoseConfig
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PoseConfig(config_path=str(tmp_path / "missing.yaml"))


# lib/vision/gesture.py
class TestGestureController:
    def test_import(self):
        from lib.vision.gesture import GestureController
        assert GestureController is not None

    def test_missing_config_raises(self, tmp_path):
        from lib.vision.gesture import GestureConfig
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GestureConfig(config_path=str(tmp_path / "missing.yaml"))
