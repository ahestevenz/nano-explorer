"""
Line follower for the JetBot.

Two modes selected via config YAML:

  classical  (default)
             HSV colour masking on the bottom third of the frame.
             Steers proportionally toward the centroid of the largest
             line-coloured contour.  No model required — ~60 FPS on Nano.

  cnn        Run a small PyTorch regression model (ResNet18 with a 1-output
             head) that directly predicts a steering angle in [-1, 1].
             Requires a trained .pth or .engine model.

YAML fields (config/models/line_follow.yaml):
    mode:       "classical" | "cnn"
    line_color: "white" | "black" | "yellow"   (classical mode only)
    min_area:   int  minimum contour area       (classical mode only)
    model:      path to .pth/.engine            (cnn mode only)

Steering law (classical mode):
    error    = (centroid_x − frame_cx) / frame_cx     # [-1, 1]
    steering = Kp × error
    motors.steer(speed, steering)

When the line is lost the robot searches by turning slowly.
"""

import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera_motion_mixin import CameraMotionMixIn
from lib.motor import MotorController
from lib.settings import PROJECT_ROOT_PATH, ensure_user_config, user_config_path

# HSV ranges for supported line colours (OpenCV hue: 0–179)
_LINE_COLOR_RANGES = {
    "white": ([0, 0, 180], [179, 60, 255], None, None),
    "black": ([0, 0, 0], [179, 255, 50], None, None),
    "yellow": ([20, 100, 100], [35, 255, 255], None, None),
}

_VALID_LINE_COLORS = list(_LINE_COLOR_RANGES.keys())
_VALID_MODES = ["classical", "cnn"]
_KP = 0.5  # proportional steering gain
_SEARCH_SPEED = 0.15


class LineFollowerConfig(BaseModel):
    """
    Line follower configuration.

    Args:
        config_path: Path to line follow YAML config.
        speed:       Forward motor speed [0.0, 1.0].
        turn_gain:   Differential turn gain [0.0, 1.0].
        stream:      Serve annotated MJPEG stream.
        stream_port: MJPEG server port.
    """

    config_path: Path = user_config_path("models/line_follow.yaml")
    speed: float = Field(0.3, ge=0.0, le=1.0)
    turn_gain: float = Field(0.5, ge=0.0, le=1.0)
    stream: bool = False
    stream_port: int = Field(8080, gt=1024, lt=65535)

    @validator("config_path", always=True)
    def config_must_exist(cls, v: Path) -> Path:  # pylint: disable=no-self-argument
        v = ensure_user_config(v)
        if not v.exists():
            raise ValueError(
                f"Line follow config not found: {v}\n" "Expected at: config/models/line_follow.yaml"
            )
        return v


class LineFollower(CameraMotionMixIn):
    """
    Follow a coloured line on the floor using the camera.

    Construct via LineFollower(**config.dict()).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._config = LineFollowerConfig(**kwargs)
        self._motors = MotorController()
        self._mode = "classical"
        self._line_color = "white"
        self._min_area = 500
        self._model = None
        self._device = None
        self._transform = None

    def _load(self) -> None:
        import yaml

        with open(self._config.config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self._mode = cfg.get("mode", "classical")
        if self._mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got '{self._mode}'")

        self._line_color = cfg.get("line_color", "white")
        if self._line_color not in _VALID_LINE_COLORS:
            raise ValueError(
                f"line_color must be one of {_VALID_LINE_COLORS}, got '{self._line_color}'"
            )

        self._min_area = int(cfg.get("min_area", 500))

        if self._mode == "cnn":
            model_path = cfg.get("model")
            if not model_path:
                raise ValueError("CNN mode requires 'model' in config YAML")
            model_path = Path(model_path)
            if not model_path.is_absolute():
                model_path = PROJECT_ROOT_PATH / model_path
            self._load_cnn(model_path)

    def _load_cnn(self, model_path: Path) -> None:
        import torch
        import torchvision.transforms as T
        from torchvision.models import resnet18

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        model = resnet18(pretrained=False)
        model.fc = __import__("torch").nn.Linear(model.fc.in_features, 1)
        state = __import__("torch").load(str(model_path), map_location=self._device)
        model.load_state_dict(state)
        self._model = model.to(self._device).eval()
        logger.success(f"Loaded CNN line follower model: {model_path}")

    def _find_line_centroid(
        self, frame: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int]], np.ndarray]:
        """Return ((cx, cy), mask) of the largest line blob in the bottom third, or (None, mask)."""
        h = frame.shape[0]
        roi = frame[h * 2 // 3 :, :]  # bottom third only

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lo1, hi1, lo2, hi2 = _LINE_COLOR_RANGES[self._line_color]
        mask = cv2.inRange(hsv, np.array(lo1), np.array(hi1))
        if lo2 is not None:
            mask |= cv2.inRange(hsv, np.array(lo2), np.array(hi2))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self._min_area:
            return None, mask

        m = cv2.moments(largest)
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"]) + h * 2 // 3
        return (cx, cy), mask

    def _infer_steering(self, frame: np.ndarray) -> float:
        """Run CNN inference and return steering angle in [-1, 1]."""
        import torch

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = self._transform(rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            return float(self._model(inp).squeeze())

    def _steer_to(self, frame: np.ndarray, centroid: Optional[Tuple[int, int]]) -> None:
        fw = frame.shape[1]
        if centroid is None:
            self._motors.turn_right(_SEARCH_SPEED)
            return
        error = (centroid[0] - fw / 2) / (fw / 2)
        self._motors.steer(self._config.speed, _KP * error * self._config.turn_gain)

    def run(self) -> None:
        self._load()

        self._motors.open()
        _stop = threading.Event()

        if self._config.stream:
            cam = self._open_camera()
            self._start_stream(cam=cam, stop_event=_stop, stream_port=self._config.stream_port)
        else:
            cam = self._open_camera()

        self._start_quit_listener(_stop)
        logger.info(
            f"Line follower running — mode={self._mode}  "
            f"color={self._line_color}  speed={self._config.speed}  (q or Ctrl+C to stop)"
        )

        try:
            while not _stop.is_set():
                frame = cam.read()

                if self._mode == "classical":
                    centroid, mask = self._find_line_centroid(frame)
                    self._steer_to(frame, centroid)
                    annotated = self._annotate_frame(frame, centroid, mask)
                else:
                    steering = self._infer_steering(frame)
                    self._motors.steer(self._config.speed, steering * self._config.turn_gain)
                    annotated = self._annotate_cnn_frame(frame, steering)

                if self._config.stream and self._server is not None:
                    self._push_frame(annotated)

        except KeyboardInterrupt:
            pass
        finally:
            _stop.set()
            self._motors.stop()
            self._motors.close()
            self._close_camera(cam)
            logger.info("Line follower stopped.")

    @staticmethod
    def _annotate_frame(
        frame: np.ndarray,
        centroid: Optional[Tuple[int, int]],
        _mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        cv2.line(out, (w // 2, 0), (w // 2, h), (200, 200, 200), 1)
        cv2.line(out, (0, h * 2 // 3), (w, h * 2 // 3), (100, 100, 100), 1)
        if centroid:
            cv2.circle(out, centroid, 10, (0, 255, 0), -1)
            cv2.line(out, (w // 2, centroid[1]), centroid, (0, 200, 255), 2)
        status = "LINE" if centroid else "SEARCHING"
        color = (0, 255, 0) if centroid else (0, 0, 255)
        cv2.putText(out, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        return out

    @staticmethod
    def _annotate_cnn_frame(frame: np.ndarray, steering: float) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        cx = w // 2
        arrow_x = int(cx + steering * cx * 0.8)
        cv2.arrowedLine(out, (cx, h - 20), (arrow_x, h - 20), (0, 255, 0), 3, tipLength=0.3)
        cv2.putText(
            out,
            f"steer={steering:+.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return out
