"""
Gesture-based robot control using trt_pose skeleton keypoints.

Recognised gestures → motor commands:

  Both wrists ABOVE shoulders  → forward
  Both wrists BELOW hips       → backward
  Left  wrist extended LEFT    → turn left
  Right wrist extended RIGHT   → turn right
  T-pose (both arms level)     → stop

Keypoint indices (COCO 17-point):
    5=left_shoulder  6=right_shoulder
    9=left_wrist    10=right_wrist
    11=left_hip     12=right_hip

All torch imports are deferred to run() to avoid SIGILL on startup.
"""

import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera import Camera
from lib.camera_motion_mixin import CameraMotionMixIn
from lib.motor import MotorController
from lib.settings import ensure_user_config, user_config_path
from lib.vision.pose import PoseConfig, PoseEstimator

_KP = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
}

_MARGIN = 0.05  # normalised coordinate deadband

_GESTURE_NAMES = ["forward", "backward", "left", "right", "stop"]

# BGR, matched to the accent colors used for these gestures in README.md / doc/images/gestures
_GESTURE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "forward": (55, 127, 26),
    "backward": (46, 34, 207),
    "left": (0, 103, 154),
    "right": (0, 103, 154),
    "stop": (30, 7, 130),
    "none": (140, 140, 140),
}


class GestureConfig(BaseModel):
    """
    Gesture control configuration.

    Args:
        config_path: Path to pose YAML (same as PoseConfig).
        speed:       Motor speed for gesture commands [0.0, 1.0].
        stream:      Serve annotated MJPEG stream.
        stream_port: MJPEG server port.
    """

    config_path: Path = user_config_path("models/pose.yaml")
    speed: float = Field(0.3, ge=0.0, le=1.0)
    stream: bool = False
    stream_port: int = Field(8080, gt=1024, lt=65535)

    @validator("config_path", always=True)
    def config_must_exist(cls, v):  # pylint: disable=no-self-argument
        v = ensure_user_config(v)
        if not v.exists():
            raise ValueError(f"Pose config not found: {v}")
        return v


class GestureController(CameraMotionMixIn):
    """
    Interpret body gestures and drive the JetBot accordingly.

    Construct via GestureController(**config.dict()).
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._config = GestureConfig(**kwargs)
        self._motors = MotorController()
        # Build PoseEstimator sharing the same config path
        self._estimator = PoseEstimator(
            **PoseConfig(
                config_path=self._config.config_path,
                stream=False,
            ).dict()
        )

    @staticmethod
    def _keypoints(peaks, objects) -> Dict[str, Optional[Tuple[float, float]]]:
        """Return {name: (x, y)} in normalised coords for the first detected person."""

        def kp(name: str) -> Optional[Tuple[float, float]]:
            idx = _KP[name]
            k = int(objects[0, 0, idx])
            if k < 0:
                return None
            # peaks are (y_norm, x_norm)
            return float(peaks[0, idx, k, 1]), float(peaks[0, idx, k, 0])  # (x, y)

        return {name: kp(name) for name in _KP}

    def _classify(self, peaks, objects) -> str:
        """
        Map the first detected person's pose to a command string.

        Returns one of: "forward" "backward" "left" "right" "stop" "none"
        """
        kps = self._keypoints(peaks, objects)
        ls, rs, lw, rw, lh, rh = (
            kps["left_shoulder"],
            kps["right_shoulder"],
            kps["left_wrist"],
            kps["right_wrist"],
            kps["left_hip"],
            kps["right_hip"],
        )

        if None in (ls, rs, lw, rw):
            return "none"

        ls_x, ls_y = ls
        rs_x, rs_y = rs
        lw_x, lw_y = lw
        rw_x, rw_y = rw

        # Both wrists above shoulders → forward  (lower y = higher in image)
        if lw_y < ls_y - _MARGIN and rw_y < rs_y - _MARGIN:
            return "forward"

        # Both wrists below hips → backward
        if lh and rh:
            _, lh_y = lh
            _, rh_y = rh
            if lw_y > lh_y + _MARGIN and rw_y > rh_y + _MARGIN:
                return "backward"

        # Left arm extended left → turn left
        if lw_x < ls_x - _MARGIN and abs(rw_y - rs_y) < _MARGIN:
            return "left"

        # Right arm extended right → turn right
        if rw_x > rs_x + _MARGIN and abs(lw_y - ls_y) < _MARGIN:
            return "right"

        # Both arms level (T-pose) → stop
        if abs(lw_y - ls_y) < _MARGIN and abs(rw_y - rs_y) < _MARGIN:
            return "stop"

        return "none"

    def _scores(self, peaks, objects) -> Dict[str, Optional[float]]:
        """
        Per-gesture match strength for the HUD, built from the same keypoints _classify
        uses (not consumed by _classify itself — display only). Each value is the signed
        margin, in normalised coordinate units, by which that gesture's primary condition
        clears _MARGIN: 0 sits exactly at the decision boundary, positive means "matching
        more confidently", None means a required keypoint wasn't detected. "left"/"right"
        only score the arm-extension check, not the secondary other-arm-level condition
        _classify also applies.
        """
        kps = self._keypoints(peaks, objects)
        ls, rs, lw, rw, lh, rh = (
            kps["left_shoulder"],
            kps["right_shoulder"],
            kps["left_wrist"],
            kps["right_wrist"],
            kps["left_hip"],
            kps["right_hip"],
        )

        if None in (ls, rs, lw, rw):
            return {name: None for name in _GESTURE_NAMES}

        ls_x, ls_y = ls
        rs_x, rs_y = rs
        lw_x, lw_y = lw
        rw_x, rw_y = rw

        scores: Dict[str, Optional[float]] = {
            "forward": min(ls_y - lw_y, rs_y - rw_y) - _MARGIN,
            "left": (ls_x - lw_x) - _MARGIN,
            "right": (rw_x - rs_x) - _MARGIN,
            "stop": _MARGIN - max(abs(lw_y - ls_y), abs(rw_y - rs_y)),
        }
        if lh and rh:
            _, lh_y = lh
            _, rh_y = rh
            scores["backward"] = min(lw_y - lh_y, rw_y - rh_y) - _MARGIN
        else:
            scores["backward"] = None
        return scores

    @staticmethod
    def _bar_fraction(score: Optional[float]) -> float:
        """Map a _scores() margin to a [0, 1] bar length; 0.5 sits at the decision boundary."""
        if score is None:
            return 0.0
        return max(0.0, min(1.0, 0.5 + score / (4 * _MARGIN)))

    def _annotate_frame(self, frame, counts, objects, peaks, gesture: str) -> np.ndarray:
        """Skeleton dots plus the identified movement and a live per-gesture score HUD."""
        out = self._estimator.annotate_frame(frame, counts, objects, peaks)
        scores = self._scores(peaks, objects)
        color = _GESTURE_COLORS.get(gesture, _GESTURE_COLORS["none"])

        # Identified movement, top-left corner
        label = gesture.upper()
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(out, (10, 10), (10 + tw + 16, 10 + th + baseline + 14), color, -1)
        cv2.putText(
            out,
            label,
            (18, 10 + th + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Per-gesture score bars, top-right corner
        panel_w, row_h, bar_w = 150, 20, 80
        panel_x = out.shape[1] - panel_w - 10
        panel_y = 10
        cv2.rectangle(
            out,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + row_h * len(_GESTURE_NAMES) + 8),
            (30, 30, 30),
            -1,
        )
        for i, name in enumerate(_GESTURE_NAMES):
            y = panel_y + 18 + i * row_h
            row_color = _GESTURE_COLORS[name] if name == gesture else (130, 130, 130)
            cv2.putText(
                out,
                name[:4],
                (panel_x + 6, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                row_color,
                1,
                cv2.LINE_AA,
            )
            bar_x = panel_x + 46
            cv2.rectangle(out, (bar_x, y - 10), (bar_x + bar_w, y + 2), (70, 70, 70), 1)
            frac = self._bar_fraction(scores.get(name))
            if frac > 0:
                cv2.rectangle(
                    out, (bar_x, y - 10), (bar_x + int(bar_w * frac), y + 2), row_color, -1
                )

        return out

    def run(self) -> None:
        self._estimator._load()  # pylint: disable=protected-access
        self._motors.open()
        _stop = threading.Event()

        if self._config.stream:
            self._start_server_stream(stream_port=self._config.stream_port)

        # Gestures drive the motors directly, so arrow-key teleop can't be used here
        # (it would fight the pose-driven commands) — just listen for q/Ctrl+C to quit.
        self._start_quit_listener(_stop)

        dispatch = {
            "forward": lambda: self._motors.forward(self._config.speed),
            "backward": lambda: self._motors.backward(self._config.speed),
            "left": lambda: self._motors.turn_left(self._config.speed * 0.6),
            "right": lambda: self._motors.turn_right(self._config.speed * 0.6),
            "stop": self._motors.stop,
            "none": self._motors.stop,
        }

        logger.info(
            "Gesture control active — body poses drive the robot (q or Ctrl+C to stop)\n"
            "  Both arms UP   → forward\n"
            "  Both arms DOWN → backward\n"
            "  Left arm OUT   → turn left\n"
            "  Right arm OUT  → turn right\n"
            "  T-pose         → stop"
        )

        with Camera() as cam:
            try:
                while not _stop.is_set():
                    frame = cam.read()
                    counts, objects, peaks = self._estimator.infer(frame)

                    if int(counts[0]) == 0:
                        self._motors.stop()
                        gesture = "none"
                    else:
                        gesture = self._classify(peaks, objects)
                        dispatch.get(gesture, self._motors.stop)()

                    logger.debug(f"Gesture: {gesture}")

                    if self._server is not None:
                        annotated = self._annotate_frame(frame, counts, objects, peaks, gesture)
                        self._push_frame(annotated)

            except KeyboardInterrupt:
                pass
            finally:
                _stop.set()
                self._motors.stop()
                self._motors.close()
                if self._server is not None:
                    self._server.stop()
                logger.info("Gesture controller stopped.")
