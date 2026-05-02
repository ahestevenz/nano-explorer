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

import threading
from enum import Enum
from pathlib import Path
from typing import List

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.settings import PROJECT_ROOT_PATH
from lib.stream_mixin import StreamMixin

_SCORE_THRESHOLD: float = 0.5


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


class ObjectDetector(StreamMixin):
    """
    Model-agnostic object detector driven by a YAML config.

    Construct via ObjectDetector(**config.dict()).
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._config = DetectionConfig(**kwargs)
        self._net = None
        self._labels = []
        self._backend = None

    def _load(self) -> None:
        import os
        import signal

        import yaml

        # Suppress TensorRT / jetson_inference verbose logging
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ.setdefault("GLOG_minloglevel", "3")  # suppress glog
        os.environ.setdefault("TRT_LOGGER_VERBOSITY", "0")  # suppress TRT

        # Restore default SIGINT so Ctrl+C works even during TRT model loading
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        with open(self._config.config_path) as f:
            cfg = yaml.safe_load(f)

        self._backend = ObjectDetectorBackend(cfg.get("backend"))
        self._threshold = self._config.threshold or cfg.get("threshold", _SCORE_THRESHOLD)

        if self._backend == ObjectDetectorBackend.JETSON_INFERENCE:
            try:
                import jetson.inference as ji  # pylint: disable=import-error
            except ImportError as e:
                raise RuntimeError(
                    "jetson.inference not found. It ships with JetPack — check your installation."
                ) from e
            self._net = ji.detectNet(cfg["model"], threshold=self._threshold)
            self._detect_fn = self._detect_jetsoni
            logger.success(f"Loaded {ObjectDetectorBackend.JETSON_INFERENCE} model: {cfg['model']}")

        elif self._backend == ObjectDetectorBackend.OPENCV_DNN:
            for key in ("model", "config", "labels"):
                if key not in cfg:
                    raise ValueError(
                        f"{ObjectDetectorBackend.OPENCV_DNN} backend requires '{key}' in config YAML"
                    )
            self._net = cv2.dnn.readNet(cfg["model"], cfg["config"])
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            with open(cfg["labels"]) as f:
                self._labels = [ln.strip() for ln in f]
            self._inp_w = cfg.get("input_width", 300)
            self._inp_h = cfg.get("input_height", 300)
            self._detect_fn = self._detect_opencv
            logger.success(f"Loaded OpenCV DNN model: {cfg['model']}")

        else:
            raise ValueError(
                f"Unknown backend '{self._backend}'. Choose: {[b.value for b in ObjectDetectorBackend]}."
            )

    def _detect_jetsoni(self, frame: np.ndarray) -> List[dict]:
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

    def _detect_opencv(self, frame: np.ndarray) -> List[dict]:
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
    def _annotated_frame(frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw bounding boxes and labels onto a copy of frame."""
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            label = "{:<20} {:.2f}".format(d["label"], d["conf"])
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
        # Detection count overlay
        cv2.putText(
            out,
            "{} object(s)".format(len(detections)),
            (10, out.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return out

    def run(self) -> None:
        import signal

        # Ensure Ctrl+C is always catchable
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        self._load()

        cam = self._open_camera()
        _stop_capture = threading.Event()

        if self._config.stream:
            self._start_stream(
                cam=cam, stop_event=_stop_capture, stream_port=self._config.stream_port
            )

        logger.info("Object detection running — Ctrl+C to stop")

        try:
            while True:
                frame = cam.read()
                detections = self._detect_fn(frame)

                for d in detections:
                    logger.info(
                        "  {:<22} conf={:.2f}  bbox={}".format(d["label"], d["conf"], d["bbox"])
                    )

                if self._config.stream and self._server is not None:
                    self._push_frame(self._annotated_frame(frame, detections))

        except KeyboardInterrupt:
            pass
        finally:
            _stop_capture.set()
            self._close_camera(cam)
            logger.info("Detector stopped.")
