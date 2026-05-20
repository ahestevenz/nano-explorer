"""
Road following for the JetBot using a regression CNN.

A ResNet18 (or MobileNetV2) with its final fully-connected layer replaced by a
single output neuron is trained to predict a steering angle in [-1, 1] from a
224×224 camera frame.  The robot is driven at a constant forward speed while the
steering angle is proportionally applied.

Training workflow (offline, on a workstation or Colab):
    1. Collect a dataset by driving manually and recording (frame, angle) pairs.
    2. Fine-tune ResNet18 with MSELoss.
    3. Save with ``torch.save(model.state_dict(), "road_follower.pth")``.
    4. Place the file in assets/models/ and point --model at it.

All torch imports are deferred to run() to avoid the OpenBLAS SIGILL on startup.
"""

import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera_motion_mixin import CameraMotionMixIn
from lib.motor import MotorController
from lib.settings import PROJECT_ROOT_PATH


class RoadFollowerConfig(BaseModel):
    """
    Road follower configuration.

    Args:
        model_path:  Path to trained .pth or .engine regression model.
        speed:       Constant forward speed [0.0, 1.0].
        turn_gain:   Scale applied to the raw steering output [0.0, 1.0].
        stream:      Serve annotated MJPEG stream.
        stream_port: MJPEG server port.
    """

    model_path: Path = PROJECT_ROOT_PATH / "assets/models/road_follower.pth"
    speed: float = Field(0.3, ge=0.0, le=1.0)
    turn_gain: float = Field(0.5, ge=0.0, le=1.0)
    stream: bool = False
    stream_port: int = Field(8080, gt=1024, lt=65535)

    @validator("model_path")
    def model_must_exist(cls, v: Path) -> Path:  # pylint: disable=no-self-argument
        if not Path(v).exists():
            raise ValueError(
                f"Model not found: {v}\n"
                "Train a ResNet18 regression model and place it at assets/models/road_follower.pth"
            )
        return v


class RoadFollower(CameraMotionMixIn):
    """
    Drive along a road/path using a CNN that predicts the steering angle.

    Construct via RoadFollower(**config.dict()).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._config = RoadFollowerConfig(**kwargs)
        self._motors = MotorController()
        self._model = None
        self._device = None
        self._transform = None

    def _load_model(self) -> None:
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

        model_path = Path(self._config.model_path)
        if model_path.suffix.lower() == ".engine":
            try:
                from torch2trt import TRTModule

                self._model = TRTModule()
                self._model.load_state_dict(torch.load(str(model_path)))
                logger.success(f"Loaded TensorRT road follower: {model_path}")
            except ImportError as e:
                raise RuntimeError(
                    "torch2trt is required for .engine models. See doc/jetbot-setup.md"
                ) from e
        else:
            model = resnet18(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 1)
            state = torch.load(str(model_path), map_location=self._device)
            model.load_state_dict(state)
            self._model = model.to(self._device).eval()
            logger.success(f"Loaded PyTorch road follower: {model_path}")

    def _infer_steering(self, frame: np.ndarray) -> float:
        import torch

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = self._transform(rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            return float(self._model(inp).squeeze().clamp(-1.0, 1.0))

    def run(self) -> None:
        import signal

        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self._load_model()

        self._motors.open()
        _stop = threading.Event()

        if self._config.stream:
            cam = self._open_camera()
            self._start_stream(cam=cam, stop_event=_stop, stream_port=self._config.stream_port)
        else:
            cam = self._open_camera()

        logger.info(
            f"Road follower running — speed={self._config.speed}  "
            f"turn_gain={self._config.turn_gain}  (Ctrl+C to stop)"
        )

        try:
            while not _stop.is_set():
                frame = cam.read()
                steering = self._infer_steering(frame)
                self._motors.steer(self._config.speed, steering * self._config.turn_gain)

                if self._config.stream and self._server is not None:
                    self._push_frame(self._annotate_frame(frame, steering))

        except KeyboardInterrupt:
            pass
        finally:
            _stop.set()
            self._motors.stop()
            self._motors.close()
            self._close_camera(cam)
            logger.info("Road follower stopped.")

    @staticmethod
    def _annotate_frame(frame: np.ndarray, steering: float) -> np.ndarray:
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
