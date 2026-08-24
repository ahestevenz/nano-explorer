"""
AprilTag / ArUco fiducial navigation for the JetBot.

Place AprilTag markers around the space and let the robot navigate by
recognising them.  Each detected tag is steered toward using proportional
control on the tag's image-plane centre.  Very reliable without extra sensors.

YAML fields (config/models/apriltag.yaml):
    family:        Tag family string (default: tag36h11).
    nthreads:      Detection threads (default: 2).
    quad_decimate: Speed/accuracy trade-off (default: 2.0).
    refine_edges:  bool (default: true).

Requires the ``pupil-apriltags`` Python package:
    pip install pupil-apriltags

Steering law:
    error    = (tag_cx − frame_cx) / frame_cx     # [-1, 1]
    steering = Kp × error
    motors.steer(speed, steering)

When no tag is visible the robot turns slowly to search.
"""

import threading
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera_motion_mixin import CameraMotionMixIn
from lib.motor import MotorController
from lib.settings import ensure_user_config, user_config_path

_KP = 0.6
_SEARCH_SPEED = 0.15


class AprilTagNavConfig(BaseModel):
    """
    AprilTag navigation configuration.

    Args:
        config_path: Path to AprilTag YAML config.
        speed:       Motor speed [0.0, 1.0].
        turn_gain:   Steering gain scale [0.0, 1.0].
        stream:      Serve annotated MJPEG stream.
        stream_port: MJPEG server port.
    """

    config_path: Path = user_config_path("models/apriltag.yaml")
    speed: float = Field(0.3, ge=0.0, le=1.0)
    turn_gain: float = Field(0.5, ge=0.0, le=1.0)
    stream: bool = False
    stream_port: int = Field(8080, gt=1024, lt=65535)

    @validator("config_path", always=True)
    def config_must_exist(cls, v: Path) -> Path:  # pylint: disable=no-self-argument
        v = ensure_user_config(v)
        if not v.exists():
            raise ValueError(
                f"AprilTag config not found: {v}\n" "Expected at: config/models/apriltag.yaml"
            )
        return v


class AprilTagNavigator(CameraMotionMixIn):
    """
    Navigate toward AprilTag fiducial markers.

    Construct via AprilTagNavigator(**config.dict()).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._config = AprilTagNavConfig(**kwargs)
        self._motors = MotorController()
        self._detector = None

    def _load(self) -> None:
        import pupil_apriltags  # pylint: disable=import-error
        import yaml

        with open(self._config.config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self._detector = pupil_apriltags.Detector(
            families=cfg.get("family", "tag36h11"),
            nthreads=int(cfg.get("nthreads", 2)),
            quad_decimate=float(cfg.get("quad_decimate", 2.0)),
            refine_edges=int(cfg.get("refine_edges", True)),
        )
        logger.success(f"AprilTag detector ready — family={cfg.get('family', 'tag36h11')}")

    def _detect(self, frame: np.ndarray) -> List[dict]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self._detector.detect(gray)
        detections = []
        for r in results:
            cx = int(r.center[0])
            cy = int(r.center[1])
            corners = [(int(p[0]), int(p[1])) for p in r.corners]
            detections.append({"id": r.tag_id, "center": (cx, cy), "corners": corners})
        return detections

    def _steer_to_nearest(
        self, frame: np.ndarray, detections: List[dict]
    ) -> Optional[Tuple[int, int]]:
        fw = frame.shape[1]
        if not detections:
            self._motors.turn_right(_SEARCH_SPEED)
            return None
        # Pick the tag with the largest apparent area (closest)
        nearest = max(
            detections,
            key=lambda d: cv2.contourArea(np.array(d["corners"])),
        )
        cx = nearest["center"][0]
        error = (cx - fw / 2) / (fw / 2)
        self._motors.steer(self._config.speed, _KP * error * self._config.turn_gain)
        return nearest["center"]

    def run(self) -> None:
        self._load()

        self._motors.open()
        _stop = threading.Event()

        if self._config.stream:
            cam = self._open_camera()
            self._start_stream(cam=cam, stop_event=_stop, stream_port=self._config.stream_port)
        else:
            cam = self._open_camera()

        self._start_quit_listener(_stop)
        logger.info("AprilTag navigator running — q or Ctrl+C to stop")

        try:
            while not _stop.is_set():
                frame = cam.read()
                detections = self._detect(frame)
                target = self._steer_to_nearest(frame, detections)

                if detections:
                    logger.debug(
                        f"{len(detections)} tag(s) — nearest id={detections[0]['id']}  "
                        f"center={target}"
                    )

                if self._config.stream and self._server is not None:
                    self._push_frame(self._annotate_frame(frame, detections))

        except KeyboardInterrupt:
            pass
        finally:
            _stop.set()
            self._motors.stop()
            self._motors.close()
            self._close_camera(cam)
            logger.info("AprilTag navigator stopped.")

    @staticmethod
    def _annotate_frame(frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        cv2.line(out, (w // 2, 0), (w // 2, h), (200, 200, 200), 1)
        for d in detections:
            corners = np.array(d["corners"], dtype=np.int32)
            cv2.polylines(out, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
            cx, cy = d["center"]
            cv2.circle(out, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(
                out,
                f"id={d['id']}",
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        status = f"{len(detections)} tag(s)" if detections else "SEARCHING"
        color = (0, 255, 0) if detections else (0, 0, 255)
        cv2.putText(out, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return out
