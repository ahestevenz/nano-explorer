"""
Face and people detection for the JetBot.

Backends selectable via config YAML:

  haar   OpenCV Haar cascades (CPU, lightweight, ~30 FPS on Nano).
         No extra downloads needed — cascades ship with OpenCV / JetPack.

  dnn    OpenCV DNN with Caffe ResNet SSD face detector (CUDA-accelerated,
         more accurate, ~15 FPS on Nano).

YAML fields (config/models/face.yaml):
    backend:       "haar" | "dnn"
    # haar
    cascade:       path to haarcascade_frontalface_default.xml
    body_cascade:  path to haarcascade_fullbody.xml (optional)
    scale_factor:  float (default 1.1)
    min_neighbors: int   (default 5)
    # dnn
    model:         path to .caffemodel
    config:        path to .prototxt
    threshold:     float (default 0.5)
"""

from enum import Enum
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera import Camera
from lib.camera_motion_mixin import CameraMotionMixIn
from lib.settings import PROJECT_ROOT_PATH

_DEFAULT_CASCADE = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"


class FaceDetectorBackend(str, Enum):
    HAAR = "haar"
    DNN = "dnn"


class FaceDetectionConfig(BaseModel):
    """
    Face detection configuration.

    Args:
        config_path: Path to face detection YAML config.
        stream:      Serve annotated MJPEG stream while running.
        stream_port: MJPEG server port.
        speed:       Motor speed [0.0, 1.0].
        turn_gain:   Differential turn gain [0.0, 1.0].
    """

    config_path: Path = PROJECT_ROOT_PATH / "config/models/face.yaml"
    stream: bool = False
    stream_port: int = Field(8080, gt=1024, lt=65535)
    speed: float = Field(0.3, ge=0.0, le=1.0)
    turn_gain: float = Field(0.5, ge=0.0, le=1.0)

    @validator("config_path")
    def config_must_exist(cls, v: Path) -> Path:  # pylint: disable=no-self-argument
        if not Path(v).exists():
            raise ValueError(f"Face config not found: {v}")
        return v


class FaceDetector(CameraMotionMixIn):
    """
    Face / people detector backed by Haar cascades or OpenCV DNN.

    Construct via FaceDetector(**config.dict()).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._config = FaceDetectionConfig(**kwargs)
        self._backend = None
        self._face_cascade = None
        self._body_cascade = None
        self._net = None
        self._threshold = 0.5
        self._load()

    def _load(self) -> None:
        import yaml

        with open(self._config.config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self._backend = FaceDetectorBackend(cfg.get("backend"))
        if self._backend == FaceDetectorBackend.HAAR:
            cascade_path = cfg.get("cascade", _DEFAULT_CASCADE)
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            if self._face_cascade.empty():
                raise RuntimeError(
                    f"Could not load Haar cascade: {cascade_path}\n"
                    "On JetPack 4.6.1 the path is:\n"
                    "  /usr/share/opencv4/haarcascades/"
                    "haarcascade_frontalface_default.xml"
                )
            body_path = cfg.get("body_cascade", "")
            if body_path:
                body_path = Path(body_path)
                if not body_path.is_absolute():
                    body_path = PROJECT_ROOT_PATH / body_path
                if body_path.exists():
                    self._body_cascade = cv2.CascadeClassifier(str(body_path))
            self._scale = cfg.get("scale_factor", 1.1)
            self._neigh = cfg.get("min_neighbors", 5)
            logger.success("Loaded Haar cascade face detector.")

        elif self._backend == FaceDetectorBackend.DNN:
            for key in ("model", "config"):
                if key not in cfg:
                    raise ValueError(f"dnn backend requires '{key}' in config YAML")

            def _resolve(p: str) -> Path:
                path = Path(p)
                return path if path.is_absolute() else PROJECT_ROOT_PATH / path

            self._net = cv2.dnn.readNetFromCaffe(
                str(_resolve(cfg["config"])), str(_resolve(cfg["model"]))
            )
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            self._threshold = cfg.get("threshold", 0.5)
            logger.success(f"Loaded DNN face detector: {cfg['model']}")

        else:
            raise ValueError(
                f"Unknown backend '{self._backend}'."
                f" Choose: {[b.value for b in FaceDetectorBackend]}."
            )

    def _detect_haar(self, frame: np.ndarray) -> List[dict]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, self._scale, self._neigh)
        results = [{"label": "face", "bbox": (x, y, x + w, y + h)} for x, y, w, h in faces]
        if self._body_cascade is not None:
            bodies = self._body_cascade.detectMultiScale(gray, self._scale, self._neigh)
            results += [{"label": "person", "bbox": (x, y, x + w, y + h)} for x, y, w, h in bodies]
        return results

    def _detect_dnn(self, frame: np.ndarray) -> List[dict]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self._net.setInput(blob)
        dets = self._net.forward()
        results = []
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf > self._threshold:
                box = (dets[0, 0, i, 3:7] * [w, h, w, h]).astype(int)
                results.append({"label": "face", "conf": conf, "bbox": tuple(box.tolist())})
        return results

    def run(self) -> None:
        import threading

        detect_fn = self._detect_haar if self._backend == "haar" else self._detect_dnn

        _stop = threading.Event()

        if self._config.stream:
            self._start_server_stream(stream_port=self._config.stream_port)

        self._start_teleop_thread(_stop, self._config.speed, self._config.turn_gain)

        with Camera() as cam:
            logger.info("Face detection running — Ctrl+C to stop")
            try:
                while not _stop.is_set():
                    frame = cam.read()
                    results = detect_fn(frame)
                    for r in results:
                        logger.info(r)
                    if self._server is not None:
                        self._push_frame(self._annotate_frame(frame, results))
            except KeyboardInterrupt:
                pass
            finally:
                _stop.set()
                if self._server is not None:
                    self._server.stop()
                logger.info("Face detector stopped.")

    @staticmethod
    def _annotate_frame(frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            conf = d.get("conf")
            text = f"{d['label']}  {conf:.2f}" if conf else d["label"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 80, 0), 2)
            cv2.putText(
                out,
                text,
                (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 80, 0),
                1,
                cv2.LINE_AA,
            )
        return out
