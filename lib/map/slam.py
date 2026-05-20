"""
Minimal monocular SLAM for the JetBot.

Two backends are supported, selected via config YAML:

  orbslam2   ORB-SLAM2 monocular mode.  Requires the Python bindings
             from github.com/muskie82/MonoSLAM or the pybind11 wrapper.
             Needs an ORB vocabulary file (ORBvoc.txt, ~1 MB download).
             Expect drift without an IMU but loop closure works in small rooms.

  rtabmap    RTAB-Map RGB-only monocular graph-SLAM.  Install via:
                 sudo apt install ros-noetic-rtabmap  (or python3-rtabmap).
             RTAB-Map manages its own database; the path is written next
             to the config file as ``slam.db``.

YAML fields (config/models/slam.yaml):
    backend:    "orbslam2" | "rtabmap"
    vocabulary: path to ORBvoc.txt          (orbslam2 only)

All heavy imports are deferred to run() to avoid SIGILL on startup.
"""

import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera_motion_mixin import CameraMotionMixIn
from lib.settings import PROJECT_ROOT_PATH

_VALID_BACKENDS = ["orbslam2", "rtabmap"]

# Tracking state labels used by ORB-SLAM2
_ORBSLAM2_STATES = {0: "NO_IMAGES", 1: "NOT_INIT", 2: "OK", 3: "LOST"}


class SlamConfig(BaseModel):
    """
    Minimal SLAM configuration.

    Args:
        config_path:  Path to SLAM YAML config.
        stream:       Serve annotated MJPEG stream.
        stream_port:  MJPEG server port.
    """

    config_path: Path = PROJECT_ROOT_PATH / "config/models/slam.yaml"
    stream: bool = False
    stream_port: int = Field(8080, gt=1024, lt=65535)

    @validator("config_path")
    def config_must_exist(cls, v: Path) -> Path:  # pylint: disable=no-self-argument
        if not Path(v).exists():
            raise ValueError(
                f"SLAM config not found: {v}\n"
                "Expected at: config/models/slam.yaml"
            )
        return v


class SlamMapper(CameraMotionMixIn):
    """
    Run monocular SLAM using ORB-SLAM2 or RTAB-Map.

    Construct via SlamMapper(**config.dict()).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._config = SlamConfig(**kwargs)
        self._backend = "orbslam2"
        self._slam = None

    def _load(self) -> None:
        import yaml

        with open(self._config.config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self._backend = cfg.get("backend", "orbslam2")
        if self._backend not in _VALID_BACKENDS:
            raise ValueError(f"backend must be one of {_VALID_BACKENDS}")

        if self._backend == "orbslam2":
            self._load_orbslam2(cfg)
        else:
            self._load_rtabmap(cfg)

    def _load_orbslam2(self, cfg: dict) -> None:
        try:
            import orbslam2  # pylint: disable=import-error
        except ImportError as e:
            raise RuntimeError(
                "orbslam2 Python bindings not found.\n"
                "Build from: github.com/muskie82/MonoSLAM or use backend: rtabmap"
            ) from e

        vocab = cfg.get("vocabulary", "assets/models/ORBvoc.txt")
        if not Path(vocab).exists():
            raise FileNotFoundError(
                f"ORB vocabulary not found: {vocab}\n"
                "Download ORBvoc.txt from github.com/raulmur/ORB_SLAM2/tree/master/Vocabulary"
            )
        self._slam = orbslam2.System(vocab, orbslam2.Sensor.MONOCULAR)
        self._slam.set_use_viewer(False)
        logger.success(f"ORB-SLAM2 initialised — vocab={vocab}")

    def _load_rtabmap(self, cfg: dict) -> None:  # pylint: disable=unused-argument
        try:
            from rtabmap.util import rtabmap as rtab  # pylint: disable=import-error
        except ImportError as e:
            raise RuntimeError(
                "RTAB-Map Python bindings not found.\n"
                "Install with: sudo apt install python3-rtabmap"
            ) from e
        self._slam = rtab.Rtabmap()
        self._slam.init()
        logger.success("RTAB-Map initialised")

    def _process_frame_orbslam2(self, frame: np.ndarray, timestamp: float) -> int:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        state = self._slam.process_image_mono(gray, timestamp)
        return int(state)

    def _process_frame_rtabmap(self, frame: np.ndarray, timestamp: float) -> bool:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return bool(self._slam.process(rgb, timestamp))

    def run(self) -> None:
        import signal
        import time

        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self._load()

        _stop = threading.Event()
        cam = self._open_camera()

        if self._config.stream:
            self._start_stream(cam=cam, stop_event=_stop, stream_port=self._config.stream_port)

        logger.info(f"SLAM running — backend={self._backend}  (Ctrl+C to stop)")

        try:
            while not _stop.is_set():
                frame = cam.read()
                ts = time.time()

                if self._backend == "orbslam2":
                    state = self._process_frame_orbslam2(frame, ts)
                    label = _ORBSLAM2_STATES.get(state, "UNKNOWN")
                    logger.debug(f"ORB-SLAM2 state={label}")
                else:
                    ok = self._process_frame_rtabmap(frame, ts)
                    label = "TRACKING" if ok else "LOST"
                    logger.debug(f"RTAB-Map tracking={ok}")

                if self._config.stream and self._server is not None:
                    self._push_frame(self._annotate_frame(frame, label, self._backend))

        except KeyboardInterrupt:
            pass
        finally:
            _stop.set()
            if self._slam is not None:
                try:
                    self._slam.shutdown()
                except Exception:  # pylint: disable=broad-except
                    pass
            self._close_camera(cam)
            logger.info("SLAM stopped.")

    @staticmethod
    def _annotate_frame(frame: np.ndarray, state_label: str, backend: str) -> np.ndarray:
        out = frame.copy()
        color = (0, 255, 0) if state_label in ("OK", "TRACKING") else (0, 0, 255)
        cv2.putText(
            out,
            f"[{backend}] {state_label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        return out
