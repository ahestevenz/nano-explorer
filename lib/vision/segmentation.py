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

from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera import Camera, MjpegServer
from lib.settings import PROJECT_ROOT_PATH


class SegmentationConfig(BaseModel):
    """
    Segmentation configuration.

    Args:
        config_path: Path to segmentation YAML config.
        stream:      Serve annotated MJPEG stream.
        stream_port: MJPEG server port.
    """

    config_path: Path = PROJECT_ROOT_PATH / "config/models/segmentation.yaml"
    stream: bool = False
    stream_port: int = Field(8080, gt=1024, lt=65535)

    @validator("config_path")
    def config_must_exist(cls, v):  # pylint: disable=no-self-argument
        if not Path(v).exists():
            raise ValueError(f"Segmentation config not found: {v}")
        return v


class Segmenter:
    """
    Semantic segmentation runner using jetson-inference segNet.

    Construct via Segmenter(**config.dict()).
    """

    def __init__(self, **kwargs):
        self._config = SegmentationConfig(**kwargs)
        self._net = None
        self._visualize = "overlay"
        self._server = None

    def _load(self) -> None:
        import yaml

        try:
            import jetson.inference as ji  # pylint: disable=import-error
            import jetson.utils as ju      # pylint: disable=import-error
            self._ji = ji
            self._ju = ju
        except ImportError as e:
            raise RuntimeError(
                "jetson.inference is required for segmentation. "
                "It ships with JetPack 4.6.1."
            ) from e

        with open(self._config.config_path) as f:
            cfg = yaml.safe_load(f)

        model = cfg.get("model", "fcn-resnet18-voc")
        self._visualize = cfg.get("visualize", "overlay")
        threshold = cfg.get("threshold", 0.0)

        self._net = self._ji.segNet(model, threshold=threshold)
        logger.success(f"Loaded segNet model: {model}")

    def run(self) -> None:
        self._load()

        if self._config.stream:
            self._server = MjpegServer(port=self._config.stream_port)
            self._server.start()

        with Camera() as cam:
            logger.info("Segmentation running — Ctrl+C to stop")
            output = None
            try:
                while True:
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
                        self._server.frame_buffer.put(bgr)

            except KeyboardInterrupt:
                pass
            finally:
                if self._server is not None:
                    self._server.stop()
                logger.info("Segmenter stopped.")
