"""
Object detection for the JetBot.

Supports two backends selected via config YAML:

  jetson-inference  TensorRT-accelerated detectNet.  Models are auto-downloaded
                    on first use.  Recommended — ~40 FPS on the Nano at 300×300.

  opencv-dnn        OpenCV DNN with CUDA backend.  Use when jetson-inference is
                    not available.  Requires model weights + config files on disk.

YAML fields (config/models/detection.yaml):
    backend:     "jetson-inference" | "opencv-dnn"
    model:       model name (jetson-inference) or path to .weights/.onnx
    config:      path to .cfg file (opencv-dnn only)
    labels:      path to labels .txt file (opencv-dnn only)
    threshold:   float confidence threshold
    input_width: int (opencv-dnn only, default 300)
    input_height: int (opencv-dnn only, default 300)

All torch / cv2 imports are deferred to run() so that importing this
module never triggers the OpenBLAS SIGILL on the Nano at startup.
"""

from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera import Camera, MjpegServer
from lib.settings import PROJECT_ROOT_PATH


class ObjectDetectorBackend(str, Enum):
    JETSON_INFERENCE = "jetson-inference"
    OPENCV_DNN = "opencv-dnn"


class DetectionConfig(BaseModel):
    """
    Object detection configuration.

    Args:
        config_path: Path to detection YAML config file.
        threshold:   Confidence threshold override (0.0–1.0).
        stream:      Serve annotated MJPEG stream while running.
        stream_port: MJPEG server port.
    """

    config_path: Path = PROJECT_ROOT_PATH / "config/models/detection.yaml"
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    stream: bool = False
    stream_port: int = Field(8080, gt=1024, lt=65535)

    @validator("config_path")
    def config_must_exist(cls, v):  # pylint: disable=no-self-argument
        if not Path(v).exists():
            raise ValueError(
                f"Detection config not found: {v}\nExpected at: config/models/detection.yaml"
            )
        return v


class ObjectDetector:
    """
    Model-agnostic object detector driven by a YAML config.

    Construct via ObjectDetector(**config.dict()).
    """

    def __init__(self, **kwargs):
        self._config = DetectionConfig(**kwargs)
        self._net = None
        self._labels = []
        self._backend = None
        self._server = None
        self._threshold = self._config.threshold

    def _load(self) -> None:
        import yaml

        with open(self._config.config_path) as f:
            cfg = yaml.safe_load(f)

        backend = cfg.get("backend", "jetson-inference")
        self._backend = backend
        threshold = self._config.threshold or cfg.get("threshold", 0.5)
        self._threshold = threshold

        if backend == "jetson-inference":
            try:
                import jetson.inference as ji  # pylint: disable=import-error
            except ImportError as e:
                raise RuntimeError(
                    "jetson.inference not found. It ships with JetPack — check your installation."
                ) from e
            self._net = ji.detectNet(cfg["model"], threshold=threshold)
            logger.success(f"Loaded jetson-inference model: {cfg['model']}")

        elif backend == "opencv-dnn":
            for key in ("model", "config", "labels"):
                if key not in cfg:
                    raise ValueError(f"opencv-dnn backend requires '{key}' in config YAML")
            self._net = cv2.dnn.readNet(cfg["model"], cfg["config"])
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            with open(cfg["labels"]) as f:
                self._labels = [ln.strip() for ln in f]
            self._inp_w = cfg.get("input_width", 300)
            self._inp_h = cfg.get("input_height", 300)
            logger.success(f"Loaded OpenCV DNN model: {cfg['model']}")

        else:
            raise ValueError(
                f"Unknown backend '{backend}'. Choose 'jetson-inference' or 'opencv-dnn'."
            )

    def _detect_jetsoni(self, frame):
        """Run jetson-inference detectNet on a BGR frame."""
        import jetson.utils as ju  # pylint: disable=import-error

        cuda_img = ju.cudaFromNumpy(frame)
        detections = self._net.Detect(cuda_img)
        return [
            {
                "label": self._net.GetClassDesc(d.ClassID),
                "conf": d.Confidence,
                "bbox": (int(d.Left), int(d.Top), int(d.Right), int(d.Bottom)),
            }
            for d in detections
        ]

    def _detect_opencv(self, frame):
        """Run OpenCV DNN detection on a BGR frame."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (self._inp_w, self._inp_h), swapRB=True, crop=False
        )
        self._net.setInput(blob)
        outs = self._net.forward(self._net.getUnconnectedOutLayersNames())
        results = []
        for out in outs:
            for det in out:
                scores = det[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence > self._threshold:
                    cx, cy = int(det[0] * w), int(det[1] * h)
                    bw, bh = int(det[2] * w), int(det[3] * h)
                    x1, y1 = cx - bw // 2, cy - bh // 2
                    label = (
                        self._labels[class_id] if class_id < len(self._labels) else str(class_id)
                    )
                    results.append(
                        {"label": label, "conf": confidence, "bbox": (x1, y1, x1 + bw, y1 + bh)}
                    )
        return results

    @staticmethod
    def annotate(frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw bounding boxes and labels on a copy of frame."""
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            label = f"{d['label']}  {d['conf']:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                out,
                label,
                (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return out

    def run(self) -> None:
        self._load()

        if self._config.stream:
            self._server = MjpegServer(port=self._config.stream_port)
            self._server.start()

        with Camera() as cam:
            logger.info("Object detection running — Ctrl+C to stop")
            try:
                while True:
                    frame = cam.read()

                    if self._backend == "jetson-inference":
                        detections = self._detect_jetsoni(frame)
                    else:
                        detections = self._detect_opencv(frame)

                    for d in detections:
                        logger.info(f"  {d['label']:<22} conf={d['conf']:.2f}  bbox={d['bbox']}")

                    if self._server is not None:
                        self._server.frame_buffer.put(self.annotate(frame, detections))

            except KeyboardInterrupt:
                pass
            finally:
                if self._server is not None:
                    self._server.stop()
                logger.info("Detector stopped.")
