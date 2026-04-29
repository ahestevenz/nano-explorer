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

from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera import Camera, MjpegServer
from lib.settings import PROJECT_ROOT_PATH

_DEFAULT_CASCADE = (
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
)


class FaceDetectionConfig(BaseModel):
    """
    Face detection configuration.

    Args:
        config_path: Path to face detection YAML config.
        stream:      Serve annotated MJPEG stream while running.
        stream_port: MJPEG server port.
    """

    config_path: Path = PROJECT_ROOT_PATH / "config/models/face.yaml"
    stream: bool = False
    stream_port: int = Field(8080, gt=1024, lt=65535)

    @validator("config_path")
    def config_must_exist(cls, v):  # pylint: disable=no-self-argument
        if not Path(v).exists():
            raise ValueError(f"Face config not found: {v}")
        return v


class FaceDetector:
    """
    Face / people detector backed by Haar cascades or OpenCV DNN.

    Construct via FaceDetector(**config.dict()).
    """

    def __init__(self, **kwargs):
        self._config = FaceDetectionConfig(**kwargs)
        self._backend = None
        self._face_cascade = None
        self._body_cascade = None
        self._net = None
        self._threshold = 0.5
        self._server = None


    def _load(self) -> None:
        import yaml

        with open(self._config.config_path) as f:
            cfg = yaml.safe_load(f)

        self._backend = cfg.get("backend", "haar")

        if self._backend == "haar":
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
            if body_path and Path(body_path).exists():
                self._body_cascade = cv2.CascadeClassifier(body_path)
            self._scale = cfg.get("scale_factor", 1.1)
            self._neigh = cfg.get("min_neighbors", 5)
            logger.success("Loaded Haar cascade face detector.")

        elif self._backend == "dnn":
            for key in ("model", "config"):
                if key not in cfg:
                    raise ValueError(f"dnn backend requires '{key}' in config YAML")
            self._net = cv2.dnn.readNetFromCaffe(cfg["config"], cfg["model"])
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            self._threshold = cfg.get("threshold", 0.5)
            logger.success(f"Loaded DNN face detector: {cfg['model']}")

        else:
            raise ValueError(f"Unknown backend '{self._backend}'. Choose 'haar' or 'dnn'.")


    def _detect_haar(self, frame) -> list:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, self._scale, self._neigh)
        results = [{"label": "face", "bbox": (x, y, x + w, y + h)} for x, y, w, h in faces]
        if self._body_cascade is not None:
            bodies = self._body_cascade.detectMultiScale(gray, self._scale, self._neigh)
            results += [{"label": "person", "bbox": (x, y, x + w, y + h)} for x, y, w, h in bodies]
        return results

    def _detect_dnn(self, frame) -> list:
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
                results.append({"label": "face", "conf": conf,
                                 "bbox": tuple(box.tolist())})
        return results


    @staticmethod
    def annotate(frame: np.ndarray, detections: list) -> np.ndarray:
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            conf = d.get("conf")
            text = f"{d['label']}  {conf:.2f}" if conf else d["label"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 80, 0), 2)
            cv2.putText(
                out, text, (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 0), 1, cv2.LINE_AA,
            )
        return out

    def run(self) -> None:
        self._load()
        detect_fn = self._detect_haar if self._backend == "haar" else self._detect_dnn

        if self._config.stream:
            self._server = MjpegServer(port=self._config.stream_port)
            self._server.start()

        with Camera() as cam:
            logger.info("Face detection running — Ctrl+C to stop")
            try:
                while True:
                    frame = cam.read()
                    results = detect_fn(frame)
                    for r in results:
                        logger.info(r)
                    if self._server is not None:
                        self._server.frame_buffer.put(self.annotate(frame, results))
            except KeyboardInterrupt:
                pass
            finally:
                if self._server is not None:
                    self._server.stop()
                logger.info("Face detector stopped.")
