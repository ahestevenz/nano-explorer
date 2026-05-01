"""
Collision avoidance using a lightweight binary CNN (blocked / free).
torch and torchvision are imported lazily inside _load_model() so that
importing this module never triggers the OpenBLAS SIGILL on the Nano.
"""

import threading
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.stream_mixin import StreamMixin
from lib.camera import Camera
from lib.motor import MotorController
from lib.settings import PROJECT_ROOT_PATH


class CollisionConfig(BaseModel):
    """
    Collision avoidance config.

    Args:
        model_path: Path to .pth or .engine model file.
        threshold:  Probability of "blocked" above which the robot reacts.
        speed:      Forward motor speed [0.0, 1.0].
        stream:     Start a background MJPEG camera stream while running.
        stream_port: Port for the MJPEG server (default 8080).
    """

    model_path: Path = PROJECT_ROOT_PATH / "assets/models/collision_avoidance.pth"
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    speed: float = Field(0.3, ge=0.0, le=1.0)
    stream: bool = False
    stream_port: int = Field(8080, gt=1024, lt=65535)

    @validator("model_path")
    def model_path_must_exist(cls, v):  # pylint: disable=no-self-argument
        if not Path(v).exists():
            raise ValueError(f"Model not found: {v}\nTrain one with: nano-explorer ml train")
        return v


class CollisionAvoider(StreamMixin):
    """
    Runs the collision avoidance loop.

    All parameters are validated by CollisionConfig before reaching here.
    Construct via CollisionAvoider(**config.dict()) — do not pass raw
    argparse values directly.
    """

    def __init__(self, **kwargs):
        self._config = CollisionConfig(**kwargs)
        self._model = None
        self._device = None
        self._transform = None

    def run(self) -> None:
        import torch

        self._load_model()

        cam = self._open_camera()
        motors = MotorController()
        motors.open()

        _stop_capture = threading.Event()
        if self._config.stream:
            self._start_stream(stream_port=self._config.stream_port)
            self._start_capture_thread(cam, _stop_capture)

        logger.info(
            f"Collision avoidance running — "
            f"threshold={self._config.threshold}  speed={self._config.speed}  "
            f"(Ctrl+C to stop)"
        )

        try:
            while True:
                frame = cam.read()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                inp = self._transform(rgb).unsqueeze(0).to(self._device)

                with torch.no_grad():
                    out = self._model(inp)
                    prob = torch.softmax(out, dim=1)[0][1].item()

                annotated = self._annotated_frame(frame=frame, model_score=prob)

                # push annotated frame to MJPEG stream if active
                if self._config.stream and self._server is not None:
                    self._push_frame(annotated)

                if prob > self._config.threshold:
                    motors.stop()
                    motors.backward(self._config.speed * 0.5)
                    _sleep(0.3)
                    motors.turn_right(self._config.speed * 0.5)
                    _sleep(0.3)
                    motors.stop()
                    logger.debug(f"BLOCKED  p={prob:.2f}")
                else:
                    motors.forward(self._config.speed)
                    logger.debug(f"free     p={prob:.2f}")

        except KeyboardInterrupt:
            logger.info("Collision avoidance stopped.")
        finally:
            motors.stop()
            motors.close()
            _stop_capture.set()
            self._close_camera(cam)

    def _load_model(self) -> None:
        # Lazy imports — only when actually running the command
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
        suffix = model_path.suffix.lower()

        if suffix == ".engine":
            try:
                from torch2trt import TRTModule

                self._model = TRTModule()
                self._model.load_state_dict(torch.load(str(model_path)))
                logger.success(f"Loaded TensorRT engine: {model_path}")
            except ImportError as e:
                raise RuntimeError(
                    "torch2trt is required for .engine models. See doc/jetbot-setup.md"
                ) from e
        else:
            state_dict = torch.load(str(model_path), map_location=self._device)
            first_key = next(iter(state_dict))

            if first_key.startswith("features"):
                # AlexNet-trained model (JetBot default notebook)
                from torchvision.models import alexnet

                model = alexnet(pretrained=False)
                model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
            else:
                # ResNet-trained model
                model = resnet18(pretrained=False)
                model.fc = torch.nn.Linear(model.fc.in_features, 2)

            model.load_state_dict(state_dict)
            self._model = model.to(self._device).eval()
            logger.success(f"Loaded PyTorch model: {model_path}")


    def _annotated_frame(self, frame: np.ndarray, model_score: float) -> np.ndarray:
        annotated = frame.copy()
        bar_origin = (10, 10)
        bar_end_x = 310
        bar_height = 35
        bar_scale = bar_end_x - bar_origin[0]

        blocked = model_score > self._config.threshold
        color = (0, 0, 255) if blocked else (0, 255, 0)

        bar_x = bar_origin[0] + int(model_score * bar_scale)
        marker_x = bar_origin[0] + int(self._config.threshold * bar_scale)

        cv2.rectangle(annotated, bar_origin, (bar_end_x, bar_height), (50, 50, 50), -1)
        cv2.rectangle(annotated, bar_origin, (bar_x, bar_height), color, -1)
        cv2.line(annotated, (marker_x, 8), (marker_x, 37), (255, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{'BLOCKED' if blocked else 'FREE'}  p={model_score:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        return annotated


def _sleep(seconds: float) -> None:
    """Thin wrapper so tests can monkeypatch time.sleep."""
    import time

    time.sleep(seconds)
