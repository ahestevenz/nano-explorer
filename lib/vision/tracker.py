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

from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera import Camera, MjpegServer
from lib.motor import MotorController
from lib.settings import PROJECT_ROOT_PATH

# HSV ranges for common colours (OpenCV hue: 0–179)
_COLOR_RANGES = {
    "red":    ([0, 100, 100],  [10, 255, 255],  [160, 100, 100], [179, 255, 255]),
    "green":  ([40, 60, 60],   [80, 255, 255],  None,            None),
    "blue":   ([100, 100, 60], [130, 255, 255], None,            None),
    "yellow": ([20, 100, 100], [35, 255, 255],  None,            None),
    "orange": ([10, 100, 100], [20, 255, 255],  None,            None),
}

_VALID_COLORS = list(_COLOR_RANGES.keys())
_VALID_MODES  = ["color", "blob", "object"]
_Kp = 0.4    # proportional steering gain
_MIN_AREA = 500


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


class ObjectTracker:
    """
    Track a visual target and steer the robot toward it.

    Construct via ObjectTracker(**config.dict()).
    """

    def __init__(self, **kwargs):
        self._config = TrackerConfig(**kwargs)
        self._motors = MotorController()
        self._server = None

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

        M = cv2.moments(largest)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy), mask

    def _find_blob_centroid(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = _MIN_AREA
        params.filterByCircularity = False
        params.filterByConvexity = False
        detector = cv2.SimpleBlobDetector_create(params)
        kps = detector.detect(cv2.bitwise_not(gray))
        if not kps:
            return None
        largest = max(kps, key=lambda k: k.size)
        return (int(largest.pt[0]), int(largest.pt[1]))

    def _steer_to(self, frame, centroid) -> None:
        fw = frame.shape[1]
        if centroid is None:
            self._motors.turn_right(0.15)
            return
        cx = centroid[0]
        error = (cx - fw / 2) / (fw / 2)
        self._motors.steer(self._config.speed, _Kp * error)

    @staticmethod
    def annotate(frame: np.ndarray, centroid) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        cv2.line(out, (w // 2, 0), (w // 2, h), (200, 200, 200), 1)
        if centroid:
            cv2.circle(out, centroid, 12, (0, 0, 255), -1)
            cv2.line(out, (w // 2, centroid[1]), centroid, (0, 200, 255), 2)
        return out

    def run(self) -> None:
        self._motors.open()

        if self._config.stream:
            self._server = MjpegServer(port=self._config.stream_port)
            self._server.start()

        logger.info(
            f"Tracker started — mode={self._config.mode}  "
            f"target={self._config.color if self._config.mode == 'color' else self._config.label}  "
            "(Ctrl+C to stop)"
        )

        with Camera() as cam:
            try:
                while True:
                    frame = cam.read()
                    centroid = None

                    if self._config.mode == "color":
                        centroid, _ = self._find_color_centroid(frame)
                    elif self._config.mode == "blob":
                        centroid = self._find_blob_centroid(frame)

                    self._steer_to(frame, centroid)

                    if self._server is not None:
                        self._server.frame_buffer.put(self.annotate(frame, centroid))

            except KeyboardInterrupt:
                pass
            finally:
                self._motors.stop()
                self._motors.close()
                if self._server is not None:
                    self._server.stop()
                logger.info("Tracker stopped.")
