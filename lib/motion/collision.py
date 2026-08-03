"""
Collision avoidance using a lightweight binary CNN (blocked / free).
torch and torchvision are imported lazily inside _load_model() so that
importing this module never triggers the OpenBLAS SIGILL on the Nano.
"""

import shutil
import threading
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera_motion_mixin import CameraMotionMixIn
from lib.motor import MotorController
from lib.settings import (
    DEFAULT_COLLISION_MODEL_HF_FILENAME,
    DEFAULT_COLLISION_MODEL_HF_REPO_ID,
    DEFAULT_COLLISION_MODEL_HF_REVISION,
    DEFAULT_COLLISION_MODEL_PATH,
)

_HF_RESOLVE_URL = "https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"


def _download_from_hub(dest: Path, repo_id: str, filename: str, revision: str) -> Path:
    """
    Best-effort fetch of a missing model from Hugging Face Hub.

    Prefers huggingface_hub (handles auth/private repos) but falls back to a
    plain HTTPS GET so this also works on the Jetson's Python 3.6 venv, where
    huggingface_hub isn't installable (it requires Python >= 3.7). Any failure
    here is non-fatal — callers just see `dest` still missing and raise their
    own error.
    """
    logger.info(f"Model not found at {dest}; attempting download of {repo_id}@{revision}")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(f"Cannot create {dest.parent}: {exc}")
        return dest

    try:
        from huggingface_hub import hf_hub_download

        cached_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        shutil.copy(cached_path, dest)
        logger.info(f"Downloaded model via huggingface_hub to {dest}")
        return dest
    except ImportError:
        logger.debug("huggingface_hub not installed; falling back to direct HTTPS download")
    except Exception as exc:  # network/auth/404 errors surfaced by huggingface_hub
        logger.warning(f"huggingface_hub download failed: {exc}")
        return dest

    url = _HF_RESOLVE_URL.format(repo_id=repo_id, revision=revision, filename=filename)
    tmp_dest = dest.with_name(dest.name + ".part")
    try:
        urllib.request.urlretrieve(url, tmp_dest)  # nosec B310 - fixed https:// HF URL
        tmp_dest.rename(dest)
        logger.info(f"Downloaded model via direct HTTPS to {dest}")
    except (urllib.error.URLError, OSError) as exc:
        logger.warning(f"Direct download from {url} failed: {exc}")
        if tmp_dest.exists():
            tmp_dest.unlink()
    return dest


class CollisionConfig(BaseModel):
    """
    Collision avoidance config.

    Args:
        hf_repo_id:   Hugging Face repo to download model_path from if missing.
        hf_filename:  Filename within hf_repo_id.
        hf_revision:  Tag/branch/commit of hf_repo_id to download.
        model_path:   Path to .pth or .engine model file.
        threshold:    Probability of "blocked" above which the robot reacts.
        speed:        Forward motor speed [0.0, 1.0].
        stream:       Start a background MJPEG camera stream while running.
        stream_port:  Port for the MJPEG server (default 8080).
    """

    hf_repo_id: str = DEFAULT_COLLISION_MODEL_HF_REPO_ID
    hf_filename: str = DEFAULT_COLLISION_MODEL_HF_FILENAME
    hf_revision: str = DEFAULT_COLLISION_MODEL_HF_REVISION
    model_path: Path = DEFAULT_COLLISION_MODEL_PATH
    threshold: float = Field(0.6, ge=0.0, le=1.0)
    speed: float = Field(0.3, ge=0.0, le=1.0)
    stream: bool = False
    stream_port: int = Field(8080, gt=1024, lt=65535)

    @validator("model_path")
    def model_path_must_exist(cls, v, values):  # pylint: disable=no-self-argument
        path = Path(v).expanduser()
        is_default = path == Path(DEFAULT_COLLISION_MODEL_PATH)

        # Only auto-download for the untouched default path. If the caller set
        # model_path explicitly (--model, env var, config file), a typo'd or
        # wrong path should raise, not silently fetch an unrelated file there.
        if not path.exists() and is_default:
            path = _download_from_hub(
                dest=path,
                repo_id=values.get("hf_repo_id", DEFAULT_COLLISION_MODEL_HF_REPO_ID),
                filename=values.get("hf_filename", DEFAULT_COLLISION_MODEL_HF_FILENAME),
                revision=values.get("hf_revision", DEFAULT_COLLISION_MODEL_HF_REVISION),
            )
        if not path.exists():
            if is_default:
                source = (
                    "This is the default (NanoSettings.collision_model_path, see lib/settings.py). "
                    "Also tried downloading "
                    f"{values.get('hf_repo_id', DEFAULT_COLLISION_MODEL_HF_REPO_ID)}"
                    f"/{values.get('hf_filename', DEFAULT_COLLISION_MODEL_HF_FILENAME)}"
                    f"@{values.get('hf_revision', DEFAULT_COLLISION_MODEL_HF_REVISION)} "
                    "from Hugging Face — see the warning above for why that failed."
                )
            else:
                source = (
                    "This path was set explicitly "
                    "(via --model, an env var, or ~/.nano-explorer.env)."
                )
            raise ValueError(
                f"Collision model not found: {path.resolve()}\n"
                f"{source}\n"
                "\n"
                "Fix it one of these ways:\n"
                "  1. Train a model:  python tools/train_collision_avoidance.py "
                "--dataset <dataset_dir>\n"
                "  2. Copy an existing .pth to that path\n"
                "  3. Point at a different model:\n"
                "       --model /path/to/model.pth              (this run only)\n"
                "       NANO_COLLISION_MODEL_PATH=/path/to/model.pth  (env var)\n"
                "       ~/.nano-explorer.env                    (persistent — add "
                "NANO_COLLISION_MODEL_PATH=... on its own line)"
            )
        return path


class CollisionAvoider(CameraMotionMixIn):
    """
    Runs the collision avoidance loop.

    All parameters are validated by CollisionConfig before reaching here.
    Construct via CollisionAvoider(**config.dict()) — do not pass raw
    argparse values directly.
    """

    def __init__(self, **kwargs):
        super().__init__()
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
            self._start_stream(
                cam=cam, stop_event=_stop_capture, stream_port=self._config.stream_port
            )

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

                annotated = self._annotate_frame(frame=frame, model_score=prob)

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

    def _annotate_frame(self, frame: np.ndarray, model_score: float) -> np.ndarray:
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
