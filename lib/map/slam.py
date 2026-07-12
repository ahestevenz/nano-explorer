"""
Minimal monocular SLAM for the JetBot.

Backend: ORB-SLAM2 monocular mode.  Requires the Python bindings from
github.com/muskie82/MonoSLAM.  Needs an ORB vocabulary file (ORBvoc.txt).
Expect drift without an IMU; loop closure works in small rooms.

YAML fields (config/models/slam.yaml):
    backend:    "orbslam2"
    vocabulary: path to ORBvoc.txt

All heavy imports are deferred to run() to avoid SIGILL on startup.
"""

import contextlib
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera_motion_mixin import CameraMotionMixIn
from lib.settings import PROJECT_ROOT_PATH

_VALID_BACKENDS = ["orbslam2"]

# Tracking state labels used by ORB-SLAM2
_ORBSLAM2_STATES = {0: "NO_IMAGES", 1: "NOT_INIT", 2: "OK", 3: "LOST"}


class SlamConfig(BaseModel):
    """
    Minimal SLAM configuration.

    Args:
        config_path:  Path to SLAM YAML config.
        stream:       Serve annotated MJPEG stream.
        stream_port:  MJPEG server port.
        speed:        Motor speed [0.0, 1.0].
        turn_gain:    Turn gain [0.0, 1.0].
    """

    config_path: Path = PROJECT_ROOT_PATH / "config/models/slam.yaml"
    stream: bool = False
    stream_port: int = Field(8080, gt=1024, lt=65535)
    speed: float = Field(0.3, ge=0.0, le=1.0)
    turn_gain: float = Field(0.5, ge=0.0, le=1.0)

    @validator("config_path")
    def config_must_exist(cls, v: Path) -> Path:  # pylint: disable=no-self-argument
        if not Path(v).exists():
            raise ValueError(f"SLAM config not found: {v}\n" "Expected at: config/models/slam.yaml")
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

        self._load_orbslam2(cfg)

    def _load_orbslam2(self, cfg: dict) -> None:
        try:
            import orbslam2  # pylint: disable=import-error
        except ImportError as e:
            raise RuntimeError(
                "orbslam2 Python bindings not found.\n"
                "Build from: https://github.com/raulmur/ORB_SLAM2"
            ) from e

        vocab = cfg.get("vocabulary", "assets/models/ORBvoc.txt")
        if not Path(vocab).exists():
            raise FileNotFoundError(
                f"ORB vocabulary not found: {vocab}\n"
                "Download ORBvoc.txt from github.com/raulmur/ORB_SLAM2/tree/master/Vocabulary"
            )
        settings = cfg.get("settings", "config/models/orbslam2_mono.yaml")
        if not Path(settings).exists():
            raise FileNotFoundError(
                f"ORB-SLAM2 settings not found: {settings}\n"
                "Create camera calibration YAML at config/models/orbslam2_mono.yaml"
            )
        self._slam = orbslam2.System(vocab, settings, orbslam2.Sensor.MONOCULAR)
        self._slam.set_use_viewer(False)
        logger.success(f"ORB-SLAM2 initialised — vocab={vocab}  settings={settings}")

    def _process_frame_orbslam2(self, frame: np.ndarray, timestamp: float) -> int:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        state = self._slam.process_image_mono(gray, timestamp)
        return int(state)

    def _get_trajectory(self) -> list:
        try:
            return self._slam.get_trajectory_points()
        except Exception:  # pylint: disable=broad-except
            return []

    def run(self) -> None:
        import signal
        import time

        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self._load()

        _stop = threading.Event()
        cam = self._open_camera()

        if self._config.stream:
            self._start_server_stream(stream_port=self._config.stream_port)

        self._start_teleop_thread(
            stop_event=_stop,
            speed=self._config.speed,
            turn_gain=self._config.turn_gain,
        )
        logger.info(f"SLAM running — backend={self._backend}  " "(arrow keys to drive, q to stop)")

        try:
            while not _stop.is_set():
                frame = cam.read()
                ts = time.time()

                state = self._process_frame_orbslam2(frame, ts)
                label = _ORBSLAM2_STATES.get(state, "UNKNOWN")
                logger.debug(f"ORB-SLAM2 state={label}")

                if self._config.stream and self._server is not None:
                    traj = self._get_trajectory()
                    h, w = frame.shape[:2]
                    scale = 2
                    self._push_frame(
                        self._render_map_view(
                            traj, frame, label, self._backend, w * scale, h * scale
                        )
                    )

        except KeyboardInterrupt:
            pass
        finally:
            _stop.set()
            if self._slam is not None:
                with contextlib.suppress(Exception):
                    self._slam.shutdown()
            self._close_camera(cam)
            logger.info("SLAM stopped.")

    @staticmethod
    def _annotate_frame(
        cam_frame: np.ndarray,
        state_label: str,
        backend: str,
    ) -> np.ndarray:
        h, w = cam_frame.shape[:2]
        return SlamMapper._render_map_view([], cam_frame, state_label, backend, w, h)

    @staticmethod
    def _render_map_view(  # pylint: disable=too-many-positional-arguments
        traj: list,
        cam_frame: np.ndarray,
        state_label: str,
        backend: str,
        canvas_w: int,
        canvas_h: int,
    ) -> np.ndarray:
        """
        Top-down trajectory map (main) with live camera PiP in the top-right corner.

        Canvas matches the camera frame dimensions so the stream size is consistent.
        traj: list of 4x4 SE3 numpy arrays from get_trajectory_points().
        """
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        if traj:
            try:
                positions = np.array([[T[0, 3], T[2, 3]] for T in traj], dtype=np.float32)
                pad = 40
                x_range = float(np.ptp(positions[:, 0])) or 1.0
                z_range = float(np.ptp(positions[:, 1])) or 1.0
                scale = min(
                    (canvas_w - 2 * pad) / x_range,
                    (canvas_h - 2 * pad) / z_range,
                )
                x_min, z_min = positions[:, 0].min(), positions[:, 1].min()
                pts = [
                    (int((x - x_min) * scale + pad), int((z - z_min) * scale + pad))
                    for x, z in positions
                ]
                for i in range(1, len(pts)):
                    cv2.line(canvas, pts[i - 1], pts[i], (0, 180, 0), 1)
                cv2.circle(canvas, pts[-1], 5, (0, 255, 0), -1)
            except Exception:  # pylint: disable=broad-except
                pass

        color = (0, 255, 0) if state_label == "OK" else (0, 0, 255)
        cv2.putText(
            canvas,
            f"[{backend}] {state_label}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "TOP-DOWN MAP",
            (10, canvas_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (100, 100, 100),
            1,
            cv2.LINE_AA,
        )

        # Camera PiP — top-right corner, 1/5 of canvas height
        pip_h = canvas_h // 5
        pip_w = int(cam_frame.shape[1] * pip_h / cam_frame.shape[0])
        pip = cv2.resize(cam_frame, (pip_w, pip_h))
        margin = 8
        x1, y1 = canvas_w - pip_w - margin, margin
        canvas[y1 - 2 : y1 + pip_h + 2, x1 - 2 : x1 + pip_w + 2] = (80, 80, 80)
        canvas[y1 : y1 + pip_h, x1 : x1 + pip_w] = pip

        return canvas
