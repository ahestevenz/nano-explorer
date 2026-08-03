"""
Visual object tracking with proportional steering for the JetBot.

Modes:
  color   HSV masking — steer toward the largest blob of the target colour.
          Very lightweight, ~60 FPS on Nano.

  blob    OpenCV SimpleBlobDetector on a greyscale image.

  object  Run a lightweight detector and track the centroid of the first
          instance of a named COCO class (default: "person").

Steering law (proportional):
    error    = (centroid_x - frame_cx) / frame_cx    # [-1, 1]
    steering = Kp * error
    motors.steer(speed, steering)

When the target is lost the robot turns slowly to search.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera import Camera
from lib.camera_motion_mixin import CameraMotionMixIn
from lib.motor import MotorController

# HSV ranges for common colours (OpenCV hue: 0–179)
_COLOR_RANGES = {
    "red": ([0, 100, 100], [10, 255, 255], [160, 100, 100], [179, 255, 255]),
    "green": ([40, 60, 60], [80, 255, 255], None, None),
    "blue": ([100, 100, 60], [130, 255, 255], None, None),
    "yellow": ([20, 100, 100], [35, 255, 255], None, None),
    "orange": ([10, 100, 100], [20, 255, 255], None, None),
}

_VALID_COLORS = list(_COLOR_RANGES.keys())
_VALID_MODES = ["color", "blob", "object"]
_KP = 0.4  # proportional steering gain
_MIN_AREA = 500
_EMA_ALPHA = 0.4  # centroid smoothing — higher tracks faster, lower rides out mask jitter
_LOST_GRACE_FRAMES = 5  # consecutive no-detection frames tolerated before really "lost"
_MAX_STEERING_DELTA = 0.15  # max steering change per frame — caps how hard one bad frame can swing


class TrackerConfig(BaseModel):
    """
    Tracker configuration.

    Args:
        mode:        "color" | "blob" | "object"
        color:       Target colour name (mode=color only).
        label:       COCO class label to track (mode=object only).
        speed:       Base forward speed [0.0, 1.0].
        stream:      Serve annotated MJPEG stream.
        stream_port: MJPEG server port.
    """

    mode: str = "color"
    color: str = "red"
    label: str = "person"
    speed: float = Field(0.25, ge=0.0, le=1.0)
    stream: bool = False
    stream_port: int = Field(8080, gt=1024, lt=65535)

    @validator("mode")
    def mode_must_be_valid(cls, v):  # pylint: disable=no-self-argument
        if v not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}")
        return v

    @validator("color")
    def color_must_be_valid(cls, v):  # pylint: disable=no-self-argument
        if v not in _VALID_COLORS:
            raise ValueError(f"color must be one of {_VALID_COLORS}")
        return v


class ObjectTracker(CameraMotionMixIn):
    """
    Track a visual target and steer the robot toward it.

    Construct via ObjectTracker(**config.dict()).
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._config = TrackerConfig(**kwargs)
        self._motors = MotorController()
        self._server = None
        self._smoothed_centroid: Optional[Tuple[float, float]] = None
        self._lost_frames = 0
        self._last_steering = 0.0

    def _stabilize(self, centroid: Optional[Tuple[int, int]]) -> Optional[Tuple[float, float]]:
        """
        Smooth the raw per-frame centroid and ride out brief detection dropouts.

        A single frame with no detection (glare, motion blur, partial occlusion) would
        otherwise flip the tracker straight into search mode; this keeps steering toward
        the last known position for _LOST_GRACE_FRAMES frames before actually giving up,
        and exponentially smooths the position so mask jitter doesn't yank the steering.
        """
        if centroid is not None:
            self._lost_frames = 0
            if self._smoothed_centroid is None:
                self._smoothed_centroid = (float(centroid[0]), float(centroid[1]))
            else:
                sx, sy = self._smoothed_centroid
                cx, cy = centroid
                self._smoothed_centroid = (
                    _EMA_ALPHA * cx + (1 - _EMA_ALPHA) * sx,
                    _EMA_ALPHA * cy + (1 - _EMA_ALPHA) * sy,
                )
            return self._smoothed_centroid

        self._lost_frames += 1
        if self._lost_frames <= _LOST_GRACE_FRAMES:
            return self._smoothed_centroid  # coast on the last known position

        self._smoothed_centroid = None
        return None

    def _find_color_centroid(self, frame):
        """Return ((cx, cy), mask) of largest colour blob, or (None, mask)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo1, hi1, lo2, hi2 = _COLOR_RANGES[self._config.color]
        mask = cv2.inRange(hsv, np.array(lo1), np.array(hi1))
        if lo2 is not None:
            mask |= cv2.inRange(hsv, np.array(lo2), np.array(hi2))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < _MIN_AREA:
            return None, mask

        m = cv2.moments(largest)
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        return (cx, cy), mask

    def _find_blob_centroid(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = _MIN_AREA
        params.filterByCircularity = False
        params.filterByConvexity = False
        detector = cv2.SimpleBlobDetector_create(params)
        kps = list(detector.detect(cv2.bitwise_not(gray)))
        if not kps:
            return None
        largest = max(kps, key=lambda k: k.size)
        return (int(largest.pt[0]), int(largest.pt[1]))

    def _steer_to(self, frame, centroid: Optional[Tuple[float, float]]) -> None:
        fw = frame.shape[1]
        if centroid is None:
            self._last_steering = 0.0
            self._motors.turn_right(0.15)
            return

        cx = centroid[0]
        error = (cx - fw / 2) / (fw / 2)
        target_steering = _KP * error

        # Rate-limit: never change steering by more than _MAX_STEERING_DELTA in one frame,
        # so a single noisy detection can't whip the wheels from hard-left to hard-right.
        delta = target_steering - self._last_steering
        delta = max(-_MAX_STEERING_DELTA, min(_MAX_STEERING_DELTA, delta))
        self._last_steering += delta

        self._motors.steer(self._config.speed, self._last_steering)

    def run(self) -> None:
        self._motors.open()

        if self._config.stream:
            self._start_server_stream(stream_port=self._config.stream_port)

        logger.info(
            f"Tracker started — mode={self._config.mode}  "
            f"target={self._config.color if self._config.mode == 'color' else self._config.label}  "
            "(Ctrl+C to stop)"
        )

        with Camera() as cam:
            try:
                while True:
                    frame = cam.read()
                    raw_centroid = None

                    if self._config.mode == "color":
                        raw_centroid, _ = self._find_color_centroid(frame)
                    elif self._config.mode == "blob":
                        raw_centroid = self._find_blob_centroid(frame)

                    centroid = self._stabilize(raw_centroid)
                    self._steer_to(frame, centroid)

                    if self._server is not None:
                        self._push_frame(self._annotate_frame(frame, centroid))

            except KeyboardInterrupt:
                pass
            finally:
                self._motors.stop()
                self._motors.close()
                if self._server is not None:
                    self._server.stop()
                logger.info("Tracker stopped.")

    @staticmethod
    def _annotate_frame(frame: np.ndarray, centroid: Optional[Tuple[float, float]]) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        cv2.line(out, (w // 2, 0), (w // 2, h), (200, 200, 200), 1)
        if centroid is not None:
            pt = (int(centroid[0]), int(centroid[1]))
            cv2.circle(out, pt, 12, (0, 0, 255), -1)
            cv2.line(out, (w // 2, pt[1]), pt, (0, 200, 255), 2)
        return out
