"""
Minimal monocular SLAM for the JetBot.

Backend: ORB-SLAM2 or ORB-SLAM3 monocular mode, selected via config/models/slam.yaml.
ORB-SLAM2 needs the Python bindings from github.com/ahestevenz/ORB_SLAM2-PythonBindings;
ORB-SLAM3 needs github.com/ahestevenz/pyorbslam. Both need an ORB vocabulary file
(ORBvoc.txt) — the same file works for either. Expect drift without an IMU; loop
closure works in small rooms.

YAML fields (config/models/slam.yaml):
    backend:    "orbslam2" | "orbslam3"

The backend's camera-calibration settings live in a fixed, non-configurable
path (config/models/orbslam2_mono.yaml or orbslam3_mono.yaml). The ORB
vocabulary path lives inside that same file, under the "Vocabulary" key.

All heavy imports are deferred to run() to avoid SIGILL on startup.
"""

import bisect
import contextlib
import threading
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera_motion_mixin import CameraMotionMixIn
from lib.settings import PROJECT_ROOT_PATH, ensure_user_config, user_config_path

_VALID_BACKENDS = ["orbslam2", "orbslam3"]

# Display names for log messages — self._backend is the raw config string.
_BACKEND_LABELS = {"orbslam2": "ORB-SLAM2", "orbslam3": "ORB-SLAM3"}

# Tracking state labels shared verbatim by both backends' eTrackingState enums
# (identical values through OK) — kept separate below because the two diverge
# at 3: ORB-SLAM2 has no RECENTLY_LOST grace period, so its 3 means LOST
# outright, while ORB-SLAM3's 3 is RECENTLY_LOST and 4 is LOST. Merging past
# OK would silently mislabel whichever backend didn't get to define state 3.
_STATES_COMMON = {-1: "NOT_READY", 0: "NO_IMAGES", 1: "NOT_INIT", 2: "OK"}

# ORB_SLAM2::Tracking::eTrackingState
_ORBSLAM2_STATES = {**_STATES_COMMON, 3: "LOST"}

# ORB_SLAM3::Tracking::eTrackingState — RECENTLY_LOST is a grace period before
# LOST, and OK_KLT is a KLT-tracking fallback state ORB-SLAM2 doesn't have.
_ORBSLAM3_STATES = {**_STATES_COMMON, 3: "RECENTLY_LOST", 4: "LOST", 5: "OK_KLT"}

# How often (in frames) to log the extra frame-quality diagnostics below —
# frequent enough to catch a stuck init within a couple seconds, cheap enough
# (one extra ORB detection pass) not to disturb the frame rate.
_DIAG_INTERVAL = 30
# If tracking hasn't left NOT_INIT/NO_IMAGES by this many frames, log one
# warning pointing at the diagnostics instead of silently looping forever.
_STALL_WARN_FRAMES = 150

# --visualize overlay: independent of ORB-SLAM2's own extractor (which exposes
# no accessor for its features/matches — see _frame_diagnostics above), so this
# runs a second, throwaway ORB detect+match pass purely for display.
_VIZ_MAX_FEATURES = 500
_VIZ_MATCH_RATIO = 0.75

# States in which the backend is still working inside a map (tracking it, or
# trying to relocalize in it), and states that mean the map was abandoned and a
# new one has to be initialized. A move from the first set to the second is a
# map restart: ORB-SLAM3 either stored the map and started a new one, or reset
# it, and in both cases the new map has its own origin, heading and scale.
_IN_MAP_LABELS = {"OK", "OK_KLT", "RECENTLY_LOST", "LOST"}
_RESTART_LABELS = {"NOT_INIT", "NO_IMAGES"}

# The state in which each backend is still relocalizing against the current map —
# the window auto-recovery has to work in. ORB-SLAM2 relocalizes in LOST
# indefinitely; ORB-SLAM3 monocular gives up after 3 s (Tracking.cc).
_RELOCALIZING_LABEL = {"orbslam2": "LOST", "orbslam3": "RECENTLY_LOST"}
# How far before the loss to start undoing moves: the motion that took the
# camera off the mapped view happened just before tracking reported it.
_RECOVERY_LOOKBACK_S = 1.0

# Map view: heading arrow length in pixels.
_HEADING_ARROW_PX = 30


class SlamConfig(BaseModel):
    """
    Minimal SLAM configuration.

    Args:
        config_path:  Path to SLAM YAML config.
        stream:       Serve annotated MJPEG stream.
        stream_port:  MJPEG server port.
        speed:        Motor speed [0.0, 1.0].
        turn_gain:    Turn gain [0.0, 1.0].
        visualize:    Replace the stream's map view with a debug view: trajectory
                       map + ORB keypoints on top, frame-to-frame matches below
                       (green=raw, blue=surviving RANSAC).
        auto_recover: When tracking is lost, reverse the operator's last moves
                       (after they release the keys) to get back to the mapped
                       view before the backend gives up on the map.
    """

    config_path: Path = user_config_path("models/slam.yaml")
    stream: bool = False
    stream_port: int = Field(8080, gt=1024, lt=65535)
    # Slower than the other commands' 0.3: at the Nano's ~6 fps SLAM rate, 0.3
    # moves the camera too far between frames to match features reliably.
    speed: float = Field(0.2, ge=0.0, le=1.0)
    turn_gain: float = Field(0.5, ge=0.0, le=1.0)
    visualize: bool = False
    auto_recover: bool = True

    @validator("config_path", always=True)
    def config_must_exist(cls, v: Path) -> Path:  # pylint: disable=no-self-argument
        v = ensure_user_config(v)
        if not v.exists():
            raise ValueError(f"SLAM config not found: {v}\n" "Expected at: config/models/slam.yaml")
        return v


class SlamMapper(CameraMotionMixIn):
    """
    Run monocular SLAM using ORB-SLAM2 or ORB-SLAM3.

    Construct via SlamMapper(**config.dict()).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._config = SlamConfig(**kwargs)
        self._backend = "orbslam2"
        self._slam = None
        self._frame_idx = 0
        self._run_start_ts = None
        self._last_frame_ts = None
        self._last_state_label = None
        self._diag_orb = None
        self._stall_warned = False
        self._viz_detector = None
        self._viz_matcher = None
        self._viz_prev_frame = None
        self._viz_prev_kps = None
        self._viz_prev_desc = None
        # Timestamps at which the backend abandoned a map and had to start a new
        # one (see _IN_MAP_LABELS / _RESTART_LABELS). Poses tracked after the
        # latest entry belong to the current map.
        self._map_starts: List[float] = []

    def _load(self) -> None:
        import yaml

        with open(self._config.config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self._backend = cfg.get("backend", "orbslam2")
        if self._backend not in _VALID_BACKENDS:
            raise ValueError(f"backend must be one of {_VALID_BACKENDS}")

        if self._backend == "orbslam3":
            self._load_orbslam3()
        else:
            self._load_orbslam2()

    @staticmethod
    def _resolve_settings(default_settings_rel: str) -> Path:
        """
        Backend settings path — fixed per backend, not user-configurable in slam.yaml.
        Routed through the same user-config seeding as every other config_path, so a
        hand-tuned calibration survives a package upgrade.
        """
        return ensure_user_config(user_config_path(default_settings_rel[len("config/") :]))

    @staticmethod
    def _resolve_vocab(settings: Path) -> Path:
        """
        ORB vocabulary path — read from the "Vocabulary" key inside the backend's own
        OpenCV FileStorage settings file, falling back to the default ORBvoc.txt
        location if that key is absent.
        """
        vocab_rel = "assets/models/ORBvoc.txt"
        fs = cv2.FileStorage(str(settings), cv2.FILE_STORAGE_READ)
        node = fs.getNode("Vocabulary")
        if not node.empty():
            vocab_rel = node.string()
        fs.release()

        vocab = Path(vocab_rel)
        if not vocab.is_absolute():
            vocab = PROJECT_ROOT_PATH / vocab
        if not vocab.exists():
            raise FileNotFoundError(
                f"ORB vocabulary not found: {vocab}\n"
                "Download ORBvoc.txt from github.com/raulmur/ORB_SLAM2/tree/master/Vocabulary"
            )
        return vocab

    def _load_orbslam2(self) -> None:
        try:
            import orbslam2  # pylint: disable=import-error
        except ImportError as e:
            raise RuntimeError(
                "orbslam2 Python bindings not found.\n"
                "Build from: https://github.com/raulmur/ORB_SLAM2"
            ) from e

        settings = self._resolve_settings("config/models/orbslam2_mono.yaml")
        if not settings.exists():
            raise FileNotFoundError(
                f"ORB-SLAM2 settings not found: {settings}\n"
                "Create camera calibration YAML at config/models/orbslam2_mono.yaml"
            )
        vocab = self._resolve_vocab(settings)
        self._slam = orbslam2.System(str(vocab), str(settings), orbslam2.Sensor.MONOCULAR)
        self._slam.set_use_viewer(False)
        # System() only records the vocab/settings paths — initialize() is what actually
        # loads the ORB vocabulary and starts tracking/mapping/loop-closing. Without this
        # call process_image_mono() runs against a system that was never started, so the
        # tracking state stays NO_IMAGES_YET forever regardless of what frames come in.
        self._slam.initialize()
        logger.success(f"ORB-SLAM2 initialised — vocab={vocab}  settings={settings}")

    def _load_orbslam3(self) -> None:
        try:
            from pyorbslam import orbslam3  # pylint: disable=import-error
        except ImportError as e:
            raise RuntimeError(
                "pyorbslam Python bindings not found.\n"
                "Build from: https://github.com/ahestevenz/pyorbslam"
            ) from e

        settings = self._resolve_settings("config/models/orbslam3_mono.yaml")
        if not settings.exists():
            raise FileNotFoundError(
                f"ORB-SLAM3 settings not found: {settings}\n"
                "Create camera calibration YAML at config/models/orbslam3_mono.yaml"
            )
        vocab = self._resolve_vocab(settings)
        self._slam = orbslam3.System(str(vocab), str(settings), orbslam3.Sensor.MONOCULAR)
        self._slam.set_use_viewer(False)
        self._slam.initialize()
        logger.success(f"ORB-SLAM3 initialised — vocab={vocab}  settings={settings}")

    def _process_frame_orbslam2(self, gray: np.ndarray, timestamp: float) -> int:
        # process_image_mono() returns a bool — whether *this frame* tracked
        # successfully — not the tracking-state enum. Casting that bool through
        # _ORBSLAM2_STATES only ever produced NO_IMAGES(0)/NOT_INIT(1); OK/LOST
        # could never appear. get_tracking_state() is the real state accessor.
        self._slam.process_image_mono(gray, timestamp)
        return int(self._slam.get_tracking_state())

    def _process_frame_orbslam3(self, gray: np.ndarray, timestamp: float) -> int:
        # process_image_mono() returns the estimated pose as an ndarray (all
        # zeros on failure) — not the tracking-state enum, same caveat as
        # orbslam2's binding above. get_tracking_state() is the real accessor.
        # GrabImageMonocular handles single-channel input directly (no internal
        # cvtColor when the image is already gray), so passing gray here — same
        # as the orbslam2 path — avoids a redundant color conversion.
        self._slam.process_image_mono(gray, timestamp, "")
        return int(self._slam.get_tracking_state())

    def _frame_diagnostics(self, gray: np.ndarray) -> str:
        """
        Independent-of-ORBSLAM2 frame health check: brightness/contrast and a plain
        cv2 ORB feature count. The orbslam2 Python bindings expose no accessor for
        how many features/matches *it* found on a given frame, so this runs a
        second, throwaway ORB pass purely to tell "camera delivering unusable frames"
        (black/blown-out/blurry/low-texture — nfeatures near 0) apart from "frames
        look fine but motion is degenerate for triangulation" (nfeatures healthy,
        tracking still stuck in NOT_INIT) without needing the C++ side instrumented.
        """
        if self._diag_orb is None:
            self._diag_orb = cv2.ORB_create(nfeatures=500)
        mean, std = cv2.meanStdDev(gray)
        keypoints = self._diag_orb.detect(gray, None)
        return (
            f"  shape={gray.shape[::-1]}  brightness={mean[0, 0]:.0f}±{std[0, 0]:.0f}"
            f"  cv2_orb_kpts={len(keypoints)}"
        )

    def _init_diagnostics(self) -> str:
        """
        ORB-SLAM3 monocular-init counters from the instrumented pyorbslam fork
        (Tracking::MonocularInitialization), for telling apart which gate a stuck
        NOT_INIT is failing:
          init_kpts  <= 100  -> not enough features to even set a reference frame
          init_raw   <  100  -> reference frame doesn't match the current one
                                (blur, too much motion between frames, low texture)
          init_inl   low/-1  -> matches found, but two-view reconstruction rejects
                                them (too little translation / parallax)
        -1 means "not computed on the last call" (e.g. a new reference frame was
        just set). '' for ORB-SLAM2 or a binding without these accessors.
        """
        if self._backend != "orbslam3" or not hasattr(self._slam, "get_last_init_detections"):
            return ""
        try:
            return (
                f"  init_kpts={self._slam.get_last_init_detections()}"
                f"  init_raw={self._slam.get_last_init_raw_matches()}"
                f"  init_inl={self._slam.get_last_init_inlier_matches()}"
            )
        except Exception:  # pylint: disable=broad-except
            return ""

    def _update_visualization(self, frame: np.ndarray, gray: np.ndarray) -> tuple:
        """
        Detect ORB keypoints in the current frame, ratio-test match them against
        the previous frame, then RANSAC-filter those matches via the fundamental
        matrix. Returns (keypoints, matches, inlier_mask, prev_frame, prev_kps) —
        prev_frame/prev_kps/inlier_mask are None on the first call (or whenever
        there aren't enough matches to fit a fundamental matrix).

        Independent of ORB-SLAM2's own extractor (see _frame_diagnostics) —
        only feeds the --visualize overlay, never the tracker itself.
        """
        if self._viz_detector is None:
            self._viz_detector = cv2.ORB_create(nfeatures=_VIZ_MAX_FEATURES)
            self._viz_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        keypoints, desc = self._viz_detector.detectAndCompute(gray, None)
        matches = []
        inlier_mask = None
        if self._viz_prev_desc is not None and desc is not None:
            raw = self._viz_matcher.knnMatch(self._viz_prev_desc, desc, k=2)
            for pair in raw:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < _VIZ_MATCH_RATIO * n.distance:
                        matches.append(m)
            inlier_mask = self._ransac_inlier_mask(self._viz_prev_kps, keypoints, matches)

        prev_frame, prev_kps = self._viz_prev_frame, self._viz_prev_kps
        self._viz_prev_frame, self._viz_prev_kps, self._viz_prev_desc = (
            frame.copy(),
            keypoints,
            desc,
        )
        return keypoints, matches, inlier_mask, prev_frame, prev_kps

    @staticmethod
    def _ransac_inlier_mask(prev_kps: list, keypoints: list, matches: list) -> Optional[np.ndarray]:
        """
        RANSAC-filter ratio-tested matches by fitting a fundamental matrix.
        None if there are too few matches (<8) to fit one — cv2.findFundamentalMat's
        minimum for the RANSAC method.
        """
        if len(matches) < 8:
            return None
        pts1 = np.float32([prev_kps[m.queryIdx].pt for m in matches])
        pts2 = np.float32([keypoints[m.trainIdx].pt for m in matches])
        _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
        return mask

    def _get_trajectory(self) -> list:
        if self._backend == "orbslam3":
            return self._get_trajectory_orbslam3()
        return self._get_trajectory_orbslam2()

    def _get_stamped_trajectory(self) -> List[Tuple[float, np.ndarray]]:
        """
        (timestamp, 4x4 camera-to-world pose) per tracked frame, across every map
        the backend has built. The timestamps are the ones run() passed in, so
        _current_map_poses() can tell which map each pose belongs to.
        """
        try:
            raw = self._slam.get_trajectory_points()
            if self._backend == "orbslam3":
                return [
                    (float(t), np.linalg.inv(np.asarray(tcw, dtype=np.float64))) for t, tcw in raw
                ]
            stamped = []
            for r in raw:
                pose = np.eye(4)
                pose[:3, :4] = np.asarray(r[1:], dtype=np.float64).reshape(3, 4)
                stamped.append((float(r[0]), pose))
            return stamped
        except Exception:  # pylint: disable=broad-except
            return []

    def _current_map_poses(self, stamped: List[Tuple[float, np.ndarray]]) -> list:
        """
        Poses of the current map only. Each map has its own origin, heading and
        scale, so poses from earlier maps can't be drawn in the same axes —
        doing that made every map look like a straight line from (0, 0).
        """
        if not self._map_starts:
            return [pose for _, pose in stamped]
        start = self._map_starts[-1]
        times = [t for t, _ in stamped]
        return [pose for _, pose in stamped[bisect.bisect_right(times, start) :]]

    def _map_label(self) -> str:
        """Current map number for the stream overlay and logs."""
        n = len(self._map_starts)
        if n == 0:
            return "map #1"
        hidden = "1 earlier map" if n == 1 else f"{n} earlier maps"
        return f"map #{n + 1} · {hidden} not shown"

    def _backend_map_count(self) -> Optional[int]:
        try:
            return int(self._slam.get_map_count()) if hasattr(self._slam, "get_map_count") else None
        except Exception:  # pylint: disable=broad-except
            return None

    def _handle_state_change(self, prev: Optional[str], label: str, ts: float) -> None:
        """
        React to a tracking-state transition: record map restarts and start or
        stop auto-recovery.
        """
        backend_label = _BACKEND_LABELS[self._backend]

        if prev in _IN_MAP_LABELS and label in _RESTART_LABELS:
            self._map_starts.append(ts)
            count = self._backend_map_count()
            atlas = f"  (backend holds {count} map(s))" if count is not None else ""
            logger.warning(
                f"{backend_label} abandoned the map — starting map #{len(self._map_starts) + 1}. "
                f"Its coordinates restart at (0, 0) facing forward; the old map is not "
                f"connected to it unless the robot revisits a mapped place.{atlas}"
            )

        teleop = getattr(self, "_teleop", None)
        relocalizing = _RELOCALIZING_LABEL.get(self._backend)
        if teleop is None or not self._config.auto_recover:
            return
        if label == relocalizing and prev not in (relocalizing, None):
            teleop.request_recovery(since=ts - _RECOVERY_LOOKBACK_S)
            logger.warning(
                f"{backend_label} lost tracking — release the keys and auto-recovery will "
                "reverse your last moves to find the mapped view again (any arrow key takes "
                "over). Tip: turn in short taps and drive forward between them."
            )
        elif prev == relocalizing and label != relocalizing:
            teleop.cancel_recovery()
            if label in ("OK", "OK_KLT"):
                logger.success(f"{backend_label} relocalized in map #{len(self._map_starts) + 1}")

    def _get_trajectory_orbslam2(self) -> list:
        # get_trajectory_points() (jskinn/ORB_SLAM2-PythonBindings, src/ORBSlamPython.cpp)
        # returns one 13-element tuple per processed frame:
        #   (timestamp, R00,R01,R02,t0, R10,R11,R12,t1, R20,R21,R22,t2)
        # — a flattened 3x4 [R|t] pose with the timestamp prepended, NOT a 4x4 SE3 matrix
        # or a flat 16-element sequence. Rebuild the 4x4 here so every consumer can rely
        # on pose[0, 3] / pose[2, 3] indexing (_last_pose_xz, _render_map_view) instead of
        # unpacking the raw tuple itself.
        try:
            poses = []
            for raw in self._slam.get_trajectory_points():
                pose = np.eye(4)
                pose[:3, :4] = np.asarray(raw[1:], dtype=np.float64).reshape(3, 4)
                poses.append(pose)
            return poses
        except Exception:  # pylint: disable=broad-except
            return []

    def _get_trajectory_orbslam3(self) -> list:
        # pyorbslam's get_trajectory_points() (ORBSlamPython.cpp) returns one
        # (timestamp, Tcw) tuple per processed frame, where Tcw is already a full
        # 4x4 ndarray — but it's the WORLD-TO-CAMERA transform, not camera-to-world.
        # ORB-SLAM3's own reference trajectory export (System::SaveTrajectoryTUM,
        # System.cc) explicitly computes `Twc = Tcw.inverse()` before treating the
        # translation as the camera's world position — this does the same, so
        # pose[0, 3] / pose[2, 3] below mean the same thing they do for the
        # orbslam2 backend above (_last_pose_xz, _render_map_view). Skipping this
        # inversion would silently plot a mirrored/wrong trajectory.
        try:
            poses = []
            for _timestamp, tcw in self._slam.get_trajectory_points():
                poses.append(np.linalg.inv(np.asarray(tcw, dtype=np.float64)))
            return poses
        except Exception:  # pylint: disable=broad-except
            return []

    @staticmethod
    def _last_pose_xz(traj: list) -> str:
        """Last (x, z) from the trajectory, formatted for logging, or '' if unavailable."""
        if not traj:
            return ""
        # Defensive on top of _get_trajectory()'s own normalization: this is a debug-log
        # convenience, not core tracking — an unexpected pose shape must never crash a
        # working SLAM run over a nice-to-have log field.
        try:
            pose = traj[-1]
            return f"  pos=({pose[0, 3]:+.2f},{pose[2, 3]:+.2f})"
        except (IndexError, TypeError):
            return ""

    def run(self) -> None:
        import time

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
        if self._config.visualize and not self._config.stream:
            logger.warning("--visualize has no effect with --no-stream (nothing to display it in)")

        self._run_start_ts = time.time()

        try:
            while not _stop.is_set():
                frame = cam.read()
                ts = time.time()
                dt = ts - self._last_frame_ts if self._last_frame_ts is not None else 0.0
                self._last_frame_ts = ts
                self._frame_idx += 1

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if self._backend == "orbslam3":
                    state = self._process_frame_orbslam3(gray, ts)
                    label = _ORBSLAM3_STATES.get(state, "UNKNOWN")
                else:
                    state = self._process_frame_orbslam2(gray, ts)
                    label = _ORBSLAM2_STATES.get(state, "UNKNOWN")
                backend_label = _BACKEND_LABELS[self._backend]

                # get_trajectory_points() walks the whole map and copies it into Python —
                # not free, and it grows as the map does. Only pay for it on frames where
                # it's actually used (a state change, or an active stream), not every frame.
                state_changed = label != self._last_state_label
                if state_changed:
                    logger.info(
                        f"{backend_label} state changed: {self._last_state_label} -> {label}  "
                        f"(frame={self._frame_idx}  t={ts - self._run_start_ts:.1f}s  "
                        f"map #{len(self._map_starts) + 1})"
                    )
                    self._handle_state_change(self._last_state_label, label, ts)
                    self._last_state_label = label

                need_traj = state_changed or (self._config.stream and self._server is not None)
                traj = self._current_map_poses(self._get_stamped_trajectory()) if need_traj else []

                traj_info = (
                    f"  map=#{len(self._map_starts) + 1}  poses={len(traj)}"
                    f"{self._last_pose_xz(traj)}"
                    if need_traj
                    else ""
                )
                need_diag = state_changed or self._frame_idx % _DIAG_INTERVAL == 0

                # opt(lazy=True) defers evaluating every arg (incl. calling
                # _frame_diagnostics, which does a real ORB pass over the frame)
                # until loguru confirms a DEBUG-level sink is actually active —
                # so this costs nothing when only INFO/WARNING are enabled.
                logger.opt(lazy=True).debug(
                    "{} frame={}  dt={:.0f}ms  state={}{}{}{}",
                    lambda backend_label=backend_label: backend_label,
                    lambda: self._frame_idx,
                    lambda dt=dt: dt * 1000,
                    lambda label=label: label,
                    lambda traj_info=traj_info: traj_info,
                    lambda label=label: self._init_diagnostics() if label == "NOT_INIT" else "",
                    lambda gray=gray, need_diag=need_diag: (
                        self._frame_diagnostics(gray) if need_diag else ""
                    ),
                )

                if (
                    label in ("NOT_INIT", "NO_IMAGES")
                    and self._frame_idx == _STALL_WARN_FRAMES
                    and not self._stall_warned
                ):
                    self._stall_warned = True
                    logger.warning(
                        f"{backend_label} still {label} after {_STALL_WARN_FRAMES} frames. "
                        "Re-run with DEBUG logging enabled to see the per-frame diagnostics: "
                        "cv2_orb_kpts near 0 means the camera feed itself is unusable "
                        "(dark/blurry/low-texture); a healthy keypoint count with no init "
                        "means the motion so far is degenerate for triangulation — drive "
                        "with forward/backward translation, not pure in-place turns, "
                        "until state changes to OK."
                    )

                if self._config.stream and self._server is not None:
                    h, w = frame.shape[:2]
                    scale = 2
                    if self._config.visualize:
                        (
                            keypoints,
                            matches,
                            inlier_mask,
                            prev_frame,
                            prev_kps,
                        ) = self._update_visualization(frame, gray)
                        view = self._render_visualization_view(
                            traj,
                            frame,
                            keypoints,
                            matches,
                            inlier_mask,
                            prev_frame,
                            prev_kps,
                            label,
                            self._backend,
                            w * scale,
                            h * scale,
                            self._map_label(),
                        )
                    else:
                        view = self._render_map_view(
                            traj,
                            frame,
                            label,
                            self._backend,
                            w * scale,
                            h * scale,
                            self._map_label(),
                        )
                    self._push_frame(view)

        except KeyboardInterrupt:
            pass
        finally:
            _stop.set()
            if getattr(self, "_teleop", None) is not None:
                self._teleop.cancel_recovery()
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
        map_label: str = "",
    ) -> np.ndarray:
        """
        Top-down trajectory map (main) with live camera PiP in the top-right corner.

        Canvas matches the camera frame dimensions so the stream size is consistent.
        traj: 4x4 camera-to-world poses of the current map, oldest first.

        Seen from above with the map's initial forward direction (+z) pointing up
        and +x to the right. A hollow circle marks where the map started, the
        filled circle is the current position, and the arrow is the camera's
        current heading, so a turn in place is visible even though the
        position doesn't change.
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
                    (int((x - x_min) * scale + pad), int(canvas_h - pad - (z - z_min) * scale))
                    for x, z in positions
                ]
                for i in range(1, len(pts)):
                    cv2.line(canvas, pts[i - 1], pts[i], (0, 180, 0), 1)
                cv2.circle(canvas, pts[0], 5, (0, 180, 0), 1)
                cv2.circle(canvas, pts[-1], 5, (0, 255, 0), -1)

                # Heading: the camera's optical axis (+z in the camera frame) is the
                # third column of the camera-to-world rotation; drop its height.
                fwd = np.asarray(traj[-1], dtype=np.float64)[:3, 2]
                norm = float(np.hypot(fwd[0], fwd[2]))
                if norm > 1e-6:
                    tip = (
                        int(pts[-1][0] + _HEADING_ARROW_PX * fwd[0] / norm),
                        int(pts[-1][1] - _HEADING_ARROW_PX * fwd[2] / norm),
                    )
                    cv2.arrowedLine(canvas, pts[-1], tip, (0, 255, 255), 2, cv2.LINE_AA, 0, 0.3)
            except Exception as e:  # pylint: disable=broad-except
                # Plotting is best-effort and must never take the stream down, but a
                # silent `pass` here is exactly what hid the pose-shape bug for two
                # rounds — log it so a bad frame is visible instead of just blank.
                logger.debug(f"Map view render skipped this frame: {e}")

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
        if map_label:
            cv2.putText(
                canvas,
                map_label,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
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

    @staticmethod
    def _render_features_panel(frame: np.ndarray, keypoints: list) -> np.ndarray:
        """Camera-frame-sized panel: current frame with detected ORB keypoints circled."""
        panel = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))
        cv2.putText(
            panel,
            f"ORB keypoints: {len(keypoints)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return panel

    @staticmethod
    def _render_matches_panel(  # pylint: disable=too-many-positional-arguments
        prev_frame: Optional[np.ndarray],
        prev_kps: Optional[list],
        frame: np.ndarray,
        keypoints: list,
        matches: list,
        inlier_mask: Optional[np.ndarray],
        w: int,
        h: int,
    ) -> np.ndarray:
        """
        Full-row-width panel: previous | current frame side by side. Every
        ratio-test match is drawn in green (raw); the subset that also survives
        RANSAC (fundamental-matrix fit) is drawn over it in blue — so a green-only
        line is a match RANSAC rejected as an outlier. Blank (with a status
        message) until a previous frame exists.
        """
        if prev_frame is None:
            panel = np.zeros((h, 2 * w, 3), dtype=np.uint8)
            cv2.putText(
                panel,
                "warming up...",
                (10, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (100, 100, 100),
                2,
                cv2.LINE_AA,
            )
            return panel

        combined = cv2.drawMatches(
            prev_frame,
            prev_kps,
            frame,
            keypoints,
            matches,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        n_inliers = 0
        if inlier_mask is not None:
            n_inliers = int(inlier_mask.sum())
            combined = cv2.drawMatches(
                prev_frame,
                prev_kps,
                frame,
                keypoints,
                matches,
                combined,
                matchColor=(255, 0, 0),
                singlePointColor=None,
                matchesMask=inlier_mask.ravel().tolist(),
                flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG
                | cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
        cv2.putText(
            combined,
            f"matches: {len(matches)} raw / {n_inliers} after RANSAC",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return combined

    @staticmethod
    def _render_visualization_view(  # pylint: disable=too-many-positional-arguments
        traj: list,
        frame: np.ndarray,
        keypoints: list,
        matches: list,
        inlier_mask: Optional[np.ndarray],
        prev_frame: Optional[np.ndarray],
        prev_kps: Optional[list],
        state_label: str,
        backend: str,
        canvas_w: int,
        canvas_h: int,
        map_label: str = "",
    ) -> np.ndarray:
        """
        Debug view for --visualize:
            top:    trajectory map | detected ORB keypoints
            bottom: frame-to-frame matches, full width (green=raw, blue=RANSAC inliers)
        """
        h, w = frame.shape[:2]

        map_panel = SlamMapper._render_map_view(traj, frame, state_label, backend, w, h, map_label)
        kp_panel = SlamMapper._render_features_panel(frame, keypoints)
        match_panel = SlamMapper._render_matches_panel(
            prev_frame, prev_kps, frame, keypoints, matches, inlier_mask, w, h
        )

        canvas = np.vstack(
            [
                np.hstack([map_panel, kp_panel]),
                match_panel,
            ]
        )
        if canvas.shape[:2] != (canvas_h, canvas_w):
            canvas = cv2.resize(canvas, (canvas_w, canvas_h))
        return canvas
