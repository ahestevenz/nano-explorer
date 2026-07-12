"""
Monocular visual odometry for the JetBot.

Estimates camera motion frame-to-frame using 2D–2D feature matching and
essential-matrix decomposition.  No IMU or depth sensor required.

Important limitations (monocular only):
  - Scale is ambiguous — translation is up to an unknown scale factor.
  - Drift accumulates over time; the pose estimate degrades in featureless
    environments or after rapid motion.
  - Best suited for slow indoor traversal of a small room.

YAML fields (config/models/odometry.yaml):
    feature_detector: "orb" | "sift" | "akaze"  (default: orb)
    max_features:     int  (default: 500)
    match_ratio:      float Lowe's ratio threshold (default: 0.75)

Algorithm overview:
    1. Detect keypoints and compute descriptors (ORB / SIFT / AKAZE).
    2. Match descriptors between consecutive frames using BFMatcher + ratio test.
    3. Recover the Essential matrix (assuming a fixed calibration-free model).
    4. Decompose into R, t and accumulate the relative pose.
    5. Annotate the live frame with matched tracks and the translation vector.
"""

import threading
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera_motion_mixin import CameraMotionMixIn
from lib.settings import PROJECT_ROOT_PATH

_VALID_DETECTORS = ["orb", "sift", "akaze"]


class OdometryConfig(BaseModel):
    """
    Visual odometry configuration.

    Args:
        config_path:  Path to odometry YAML config.
        stream:       Serve annotated MJPEG stream.
        stream_port:  MJPEG server port.
        speed:        Motor speed [0.0, 1.0].
        turn_gain:    Turn gain [0.0, 1.0].
    """

    config_path: Path = PROJECT_ROOT_PATH / "config/models/odometry.yaml"
    stream: bool = False
    stream_port: int = Field(8080, gt=1024, lt=65535)
    speed: float = Field(0.3, ge=0.0, le=1.0)
    turn_gain: float = Field(0.5, ge=0.0, le=1.0)

    @validator("config_path")
    def config_must_exist(cls, v: Path) -> Path:  # pylint: disable=no-self-argument
        if not Path(v).exists():
            raise ValueError(
                f"Odometry config not found: {v}\n" "Expected at: config/models/odometry.yaml"
            )
        return v


class VisualOdometry(CameraMotionMixIn):
    """
    Frame-to-frame monocular visual odometry.

    Construct via VisualOdometry(**config.dict()).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._config = OdometryConfig(**kwargs)
        self._detector = None
        self._matcher = None
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_kps = None
        self._prev_desc = None
        self._pose = np.eye(4, dtype=np.float64)
        self._max_features = 500
        self._match_ratio = 0.75

    def _init_detector(self, detector_type: str, max_features: int) -> None:
        dt = detector_type.lower()
        if dt == "orb":
            self._detector = cv2.ORB_create(nfeatures=max_features)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif dt == "sift":
            self._detector = cv2.SIFT_create(nfeatures=max_features)
            self._matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        elif dt == "akaze":
            self._detector = cv2.AKAZE_create()
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError(f"feature_detector must be one of {_VALID_DETECTORS}")

    def _load(self) -> None:
        import yaml

        with open(self._config.config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        detector_type = cfg.get("feature_detector", "orb")
        self._max_features = int(cfg.get("max_features", 500))
        self._match_ratio = float(cfg.get("match_ratio", 0.75))
        self._init_detector(detector_type, self._max_features)
        logger.success(f"Visual odometry ready — detector={detector_type}")

    def _find_keypoints(self, gray: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        kps, desc = self._detector.detectAndCompute(gray, None)
        return list(kps), desc

    def _match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None:
            return []
        raw = self._matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self._match_ratio * n.distance:
                    good.append(m)
        return good

    def _estimate_motion(
        self,
        kps1: List[cv2.KeyPoint],
        kps2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        frame_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Return a 4×4 relative transform, or None if estimation fails."""
        if len(matches) < 8:
            return None

        h, w = frame_shape
        # Approximate focal length from image width (no calibration)
        f = w
        cx, cy = w / 2.0, h / 2.0
        k_mat = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

        pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

        e_mat, mask = cv2.findEssentialMat(
            pts1, pts2, k_mat, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if e_mat is None:
            return None

        _, r_mat, t, _ = cv2.recoverPose(e_mat, pts1, pts2, k_mat, mask=mask)

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = r_mat
        transform[:3, 3] = t.ravel()
        return transform

    def run(self) -> None:
        import signal

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
        logger.info("Visual odometry running — (arrow keys to drive, q to stop)")

        try:
            while not _stop.is_set():
                frame = cam.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                kps, desc = self._find_keypoints(gray)
                matches: List[cv2.DMatch] = []
                delta_t: Optional[np.ndarray] = None

                if self._prev_gray is not None and self._prev_desc is not None:
                    matches = self._match_features(self._prev_desc, desc)
                    delta_t = self._estimate_motion(self._prev_kps, kps, matches, gray.shape)
                    if delta_t is not None:
                        self._pose = self._pose @ delta_t
                        t = self._pose[:3, 3]
                        logger.debug(
                            f"pos=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})  " f"matches={len(matches)}"
                        )

                self._prev_gray = gray
                self._prev_kps = kps
                self._prev_desc = desc

                if self._config.stream and self._server is not None:
                    self._push_frame(self._annotate_frame(frame, kps, matches, self._pose))

        except KeyboardInterrupt:
            pass
        finally:
            _stop.set()
            self._close_camera(cam)
            logger.info("Visual odometry stopped.")

    @staticmethod
    def _annotate_frame(
        frame: np.ndarray,
        kps: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        pose: np.ndarray,
    ) -> np.ndarray:
        out = frame.copy()
        for kp in kps:
            cv2.circle(out, (int(kp.pt[0]), int(kp.pt[1])), 2, (0, 200, 0), -1)

        t = pose[:3, 3]
        cv2.putText(
            out,
            f"pos: ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"matches: {len(matches)}  kps: {len(kps)}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return out
