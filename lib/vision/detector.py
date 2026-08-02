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
import zlib
from enum import Enum
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera_motion_mixin import CameraMotionMixIn
from lib.settings import PROJECT_ROOT_PATH

_SCORE_THRESHOLD: float = 0.5

# Distinct BGR colors cycled per class label so each object type stays visually
# consistent across frames without needing a static label->color config.
_PALETTE: List[Tuple[int, int, int]] = [
    (68, 122, 244),  # blue
    (76, 175, 80),  # green
    (0, 202, 255),  # gold
    (60, 76, 231),  # red
    (191, 82, 155),  # purple
    (172, 178, 26),  # teal
    (35, 133, 235),  # orange
    (139, 148, 241),  # salmon
    (127, 191, 63),  # lime
    (219, 152, 52),  # light blue
    (18, 156, 243),  # amber
    (166, 165, 149),  # gray
]


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
        speed:       Motor speed [0.0, 1.0].
        turn_gain:   Differential turn gain [0.0, 1.0].
    """

    config_path: Path = PROJECT_ROOT_PATH / "config/models/detection.yaml"
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    stream: bool = False
    stream_port: int = Field(8080, gt=1024, lt=65535)
    speed: float = Field(0.3, ge=0.0, le=1.0)
    turn_gain: float = Field(0.5, ge=0.0, le=1.0)

    @validator("config_path")
    def config_must_exist(cls, v: Path) -> Path:  # pylint: disable=no-self-argument
        if not Path(v).exists():
            raise ValueError(
                f"Detection config not found: {v}\nExpected at: config/models/detection.yaml"
            )
        return v


class ObjectDetector(CameraMotionMixIn):
    """
    Model-agnostic object detector driven by a YAML config.

    Construct via ObjectDetector(**config.dict()).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._config = DetectionConfig(**kwargs)
        self._net = None
        self._labels = []
        self._backend = None
        self._load()

    def _load(self) -> None:
        import os
        import signal

        import yaml

        # Suppress TensorRT / jetson_inference verbose logging
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ.setdefault("GLOG_minloglevel", "3")  # suppress glog
        os.environ.setdefault("TRT_LOGGER_VERBOSITY", "0")  # suppress TRT

        with open(self._config.config_path, encoding="utf-8") as f:
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
            # SIG_DFL during TRT load only — lets Ctrl+C kill a hung model init
            # immediately. Restored to Python's handler once loading completes.
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            self._net = ji.detectNet(cfg["model"], threshold=self._threshold)
            signal.signal(signal.SIGINT, signal.default_int_handler)
            self._detect_fn = self._detect_jetsoni
            logger.success(f"Loaded {ObjectDetectorBackend.JETSON_INFERENCE} model: {cfg['model']}")

        elif self._backend == ObjectDetectorBackend.OPENCV_DNN:
            for key in ("model", "config", "labels"):
                if key not in cfg:
                    raise ValueError(
                        f"{ObjectDetectorBackend.OPENCV_DNN} backend"
                        f" requires '{key}' in config YAML"
                    )
            self._net = cv2.dnn.readNet(cfg["model"], cfg["config"])
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            with open(cfg["labels"], encoding="utf-8") as f:
                self._labels = [ln.strip() for ln in f]
            self._inp_w = cfg.get("input_width", 300)
            self._inp_h = cfg.get("input_height", 300)
            self._detect_fn = self._detect_opencv
            logger.success(f"Loaded OpenCV DNN model: {cfg['model']}")

        else:
            raise ValueError(
                f"Unknown backend '{self._backend}'."
                f" Choose: {[b.value for b in ObjectDetectorBackend]}."
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

    def run(self) -> None:
        cam = self._open_camera()
        _stop = threading.Event()

        if self._config.stream:
            self._start_stream(cam=cam, stop_event=_stop, stream_port=self._config.stream_port)

        self._start_teleop_thread(_stop, self._config.speed, self._config.turn_gain)

        logger.info("Object detection running — Ctrl+C to stop")

        try:
            while not _stop.is_set():
                frame = cam.read()
                detections = self._detect_fn(frame)

                for d in detections:
                    logger.info(f"{d['label']:<22} conf={d['conf']:.2f}  bbox={d['bbox']}")

                if self._config.stream and self._server is not None:
                    self._push_frame(self._annotate_frame(frame, detections))

        except KeyboardInterrupt:
            pass
        finally:
            _stop.set()
            self._close_camera(cam)
            if self._server is not None:
                self._server.stop()
            logger.info("Detector stopped.")

    @staticmethod
    def _color_for_label(label: str) -> Tuple[int, int, int]:
        """Deterministic per-class BGR color, stable across frames and runs."""
        return _PALETTE[zlib.crc32(label.encode("utf-8")) % len(_PALETTE)]

    @staticmethod
    def _readable_text_color(bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Black or white, whichever contrasts better against a BGR fill color."""
        b, g, r = bgr
        luminance = 0.114 * b + 0.587 * g + 0.299 * r
        return (0, 0, 0) if luminance > 140 else (255, 255, 255)

    @classmethod
    def _annotate_frame(cls, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Draw per-class colored ROIs, each tagged with object_type/score, plus a count."""
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            color = cls._color_for_label(d["label"])
            text_color = cls._readable_text_color(color)
            tag = f"{d['label']} {d['conf']:.2f}"

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            tag_y0 = max(y1 - th - baseline - 4, 0)
            cv2.rectangle(out, (x1, tag_y0), (x1 + tw + 6, tag_y0 + th + baseline + 4), color, -1)
            cv2.putText(
                out,
                tag,
                (x1 + 3, tag_y0 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                text_color,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            out,
            f"{len(detections)} object(s)",
            (10, out.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return out
