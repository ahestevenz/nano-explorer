"""
Semantic segmentation for the JetBot using jetson-inference segNet (TensorRT).

Available models (auto-downloaded on first use):
    fcn-resnet18-voc          Pascal VOC  — 21 classes  (person, car, chair …)
    fcn-resnet18-cityscapes   Cityscapes  — 19 classes  (road, building, car …)
    fcn-resnet18-deepscene    DeepScene   — outdoor/forest
    fcn-resnet18-mhp          Multi-Human Parsing — 15 body-part classes
    fcn-resnet18-sun          SUN RGB-D   — 37 indoor classes

YAML fields (config/models/segmentation.yaml):
    model:     jetson-inference model name
    visualize: "overlay" | "mask"  (default: overlay)
    threshold: float  (default: 0.0)
"""

import threading
from pathlib import Path
from typing import Any

import cv2
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera import Camera
from lib.camera_motion_mixin import CameraMotionMixIn
from lib.settings import ensure_user_config, user_config_path


class SegmentationConfig(BaseModel):
    """
    Segmentation configuration.

    Args:
        config_path: Path to segmentation YAML config.
        stream:      Serve annotated MJPEG stream.
        stream_port: MJPEG server port.
        speed:       Motor speed [0.0, 1.0].
        turn_gain:   Differential turn gain [0.0, 1.0].
    """

    config_path: Path = user_config_path("models/segmentation.yaml")
    stream: bool = False
    stream_port: int = Field(8080, gt=1024, lt=65535)
    speed: float = Field(0.3, ge=0.0, le=1.0)
    turn_gain: float = Field(0.5, ge=0.0, le=1.0)

    @validator("config_path", always=True)
    def config_must_exist(cls, v: Path) -> Path:  # pylint: disable=no-self-argument
        v = ensure_user_config(v)
        if not v.exists():
            raise ValueError(f"Segmentation config not found: {v}")
        return v


class Segmenter(CameraMotionMixIn):
    """
    Semantic segmentation runner using jetson-inference segNet.

    Construct via Segmenter(**config.dict()).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._config = SegmentationConfig(**kwargs)
        self._net = None
        self._visualize = "overlay"
        self._load()

    def _load(self) -> None:
        import yaml

        try:
            import jetson.inference as ji  # pylint: disable=import-error
            import jetson.utils as ju  # pylint: disable=import-error

            self._ji = ji
            self._ju = ju
        except ImportError as e:
            raise RuntimeError(
                "jetson.inference is required for segmentation. It ships with JetPack 4.6.1."
            ) from e

        with open(self._config.config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        model = cfg.get("model", "fcn-resnet18-voc")
        self._visualize = cfg.get("visualize", "overlay")

        self._net = self._ji.segNet(model)
        logger.success(f"Loaded segNet model: {model}")

    def run(self) -> None:
        _stop = threading.Event()

        if self._config.stream:
            self._start_server_stream(stream_port=self._config.stream_port)

        self._start_teleop_thread(_stop, self._config.speed, self._config.turn_gain)

        with Camera() as cam:
            logger.info("Segmentation running — Ctrl+C to stop")
            output = None
            try:
                while not _stop.is_set():
                    frame = cam.read()
                    cuda_img = self._ju.cudaFromNumpy(frame)

                    if output is None:
                        h, w = frame.shape[:2]
                        output = self._ju.cudaAllocMapped(width=w, height=h, format="rgb8")

                    self._net.Process(cuda_img)
                    if self._visualize == "mask":
                        self._net.Mask(output)
                    else:
                        self._net.Overlay(output)

                    result = self._ju.cudaToNumpy(output)
                    bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

                    if self._server is not None:
                        self._push_frame(bgr)

            except KeyboardInterrupt:
                pass
            finally:
                _stop.set()
                if self._server is not None:
                    self._server.stop()
                logger.info("Segmenter stopped.")
