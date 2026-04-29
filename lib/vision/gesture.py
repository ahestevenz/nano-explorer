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

from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera import Camera, MjpegServer
from lib.motor import MotorController
from lib.settings import PROJECT_ROOT_PATH
from lib.vision.pose import PoseEstimator, PoseConfig

_KP = {
    "left_shoulder":  5,
    "right_shoulder": 6,
    "left_wrist":     9,
    "right_wrist":   10,
    "left_hip":      11,
    "right_hip":     12,
}

_MARGIN = 0.05  # normalised coordinate deadband


class GestureConfig(BaseModel):
    """
    Gesture control configuration.

    Args:
        config_path: Path to pose YAML (same as PoseConfig).
        speed:       Motor speed for gesture commands [0.0, 1.0].
        stream:      Serve annotated MJPEG stream.
        stream_port: MJPEG server port.
    """

    config_path: Path = PROJECT_ROOT_PATH / "config/models/pose.yaml"
    speed: float = Field(0.3, ge=0.0, le=1.0)
    stream: bool = False
    stream_port: int = Field(8080, gt=1024, lt=65535)

    @validator("config_path")
    def config_must_exist(cls, v):  # pylint: disable=no-self-argument
        if not Path(v).exists():
            raise ValueError(f"Pose config not found: {v}")
        return v


class GestureController:
    """
    Interpret body gestures and drive the JetBot accordingly.

    Construct via GestureController(**config.dict()).
    """

    def __init__(self, **kwargs):
        self._config = GestureConfig(**kwargs)
        self._motors = MotorController()
        self._server = None
        # Build PoseEstimator sharing the same config path
        self._estimator = PoseEstimator(
            **PoseConfig(
                config_path=self._config.config_path,
                stream=False,
            ).dict()
        )

    def _classify(self, peaks, objects) -> str:
        """
        Map the first detected person's pose to a command string.

        Returns one of: "forward" "backward" "left" "right" "stop" "none"
        """

        def kp(name):
            idx = _KP[name]
            k = int(objects[0, 0, idx])
            if k < 0:
                return None
            # peaks are (y_norm, x_norm)
            return float(peaks[0, idx, k, 1]), float(peaks[0, idx, k, 0])  # (x, y)

        ls = kp("left_shoulder");   rs = kp("right_shoulder")
        lw = kp("left_wrist");      rw = kp("right_wrist")
        lh = kp("left_hip");        rh = kp("right_hip")

        if None in (ls, rs, lw, rw):
            return "none"

        ls_x, ls_y = ls;  rs_x, rs_y = rs
        lw_x, lw_y = lw;  rw_x, rw_y = rw

        # Both wrists above shoulders → forward  (lower y = higher in image)
        if lw_y < ls_y - _MARGIN and rw_y < rs_y - _MARGIN:
            return "forward"

        # Both wrists below hips → backward
        if lh and rh:
            _, lh_y = lh;  _, rh_y = rh
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

    def run(self) -> None:
        self._estimator._load()  # pylint: disable=protected-access
        self._motors.open()

        if self._config.stream:
            self._server = MjpegServer(port=self._config.stream_port)
            self._server.start()

        dispatch = {
            "forward":  lambda: self._motors.forward(self._config.speed),
            "backward": lambda: self._motors.backward(self._config.speed),
            "left":     lambda: self._motors.turn_left(self._config.speed * 0.6),
            "right":    lambda: self._motors.turn_right(self._config.speed * 0.6),
            "stop":     self._motors.stop,
            "none":     self._motors.stop,
        }

        logger.info(
            "Gesture control active — body poses drive the robot (Ctrl+C to stop)\n"
            "  Both arms UP   → forward\n"
            "  Both arms DOWN → backward\n"
            "  Left arm OUT   → turn left\n"
            "  Right arm OUT  → turn right\n"
            "  T-pose         → stop"
        )

        with Camera() as cam:
            try:
                while True:
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
                        annotated = self._estimator.annotate(frame, counts, objects, peaks)
                        self._server.frame_buffer.put(annotated)

            except KeyboardInterrupt:
                pass
            finally:
                self._motors.stop()
                self._motors.close()
                if self._server is not None:
                    self._server.stop()
                logger.info("Gesture controller stopped.")
