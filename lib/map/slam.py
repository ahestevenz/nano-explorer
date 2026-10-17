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
import collections
import contextlib
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera_motion_mixin import CameraMotionMixIn
from lib.map import slam_view
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

# ORB-SLAM3 monocular's built-in RECENTLY_LOST window. The nano-explorer pyorbslam
# fork makes it configurable ("System.RecentlyLostSeconds" in orbslam3_mono.yaml)
# and reports the value it uses; older builds always use this.
_DEFAULT_RECENTLY_LOST_S = 3.0
# Auto-recovery replays at most (window - margin) seconds, so it finishes with
# time left to relocalize, and never more than the cap: undoing much more than
# that rarely gets back to the mapped view.
_RECOVERY_MARGIN_S = 1.0
_RECOVERY_REPLAY_CAP_S = 8.0


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
        max_exposure_ms: Cap the CSI camera's auto-exposure time (ms) to reduce
                       motion blur; None keeps the camera's default range.
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
    max_exposure_ms: Optional[float] = Field(None, gt=0)

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
        self._viz = slam_view.MatchVisualizer()
        # Timestamps at which the backend abandoned a map and had to start a new
        # one (see _IN_MAP_LABELS / _RESTART_LABELS). Poses tracked after the
        # latest entry belong to the current map.
        self._map_starts: List[float] = []
        self._last_map_count: Optional[int] = None
        self._recently_lost_s = _DEFAULT_RECENTLY_LOST_S

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
        self._recently_lost_s = self._resolve_recently_lost(settings)

    def _resolve_recently_lost(self, settings: Path) -> float:
        """
        The RECENTLY_LOST window ORB-SLAM3 actually uses. The fork reports it;
        an older build always uses 3 s, so a longer value in the settings file
        would be silently ignored — say so instead.
        """
        if hasattr(self._slam, "get_recently_lost_seconds"):
            try:
                effective = float(self._slam.get_recently_lost_seconds())
            except Exception:  # pylint: disable=broad-except
                effective = -1.0
            if effective > 0:
                logger.info(f"ORB-SLAM3 relocalization window: {effective:.1f} s")
                return effective

        fs = cv2.FileStorage(str(settings), cv2.FILE_STORAGE_READ)
        node = fs.getNode("System.RecentlyLostSeconds")
        requested = None if node.empty() else node.real()
        fs.release()
        if requested is not None and abs(requested - _DEFAULT_RECENTLY_LOST_S) > 1e-6:
            logger.warning(
                f"System.RecentlyLostSeconds={requested:g} in {settings} is ignored by this "
                f"pyorbslam build, which always uses {_DEFAULT_RECENTLY_LOST_S:g} s. Rebuild "
                "pyorbslam from the nano-explorer fork to use it."
            )
        return _DEFAULT_RECENTLY_LOST_S

    def _recovery_replay_s(self) -> float:
        """Longest auto-recovery replay that still leaves time to relocalize."""
        return min(
            max(self._recently_lost_s - _RECOVERY_MARGIN_S, 1.0),
            _RECOVERY_REPLAY_CAP_S,
        )

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

    # Rendering lives in lib/map/slam_view.py. These aliases keep the names that
    # _annotate_frame, run() and the tests use.
    _ransac_inlier_mask = staticmethod(slam_view.ransac_inlier_mask)
    _render_map_view = staticmethod(slam_view.render_map_view)
    _render_features_panel = staticmethod(slam_view.render_features_panel)
    _render_matches_panel = staticmethod(slam_view.render_matches_panel)
    _render_visualization_view = staticmethod(slam_view.render_visualization_view)

    def _update_visualization(self, frame: np.ndarray, gray: np.ndarray) -> tuple:
        """--visualize overlay data; see slam_view.MatchVisualizer.update."""
        return self._viz.update(frame, gray)

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

    def _get_map_trajectory(self) -> List[Tuple[float, np.ndarray, int]]:
        """
        (timestamp, camera-to-world pose, map key) per tracked frame.

        With the nano-explorer pyorbslam fork, the map key is ORB-SLAM3's own map
        id for each pose, so poses of a map that got merged into the current one
        report the current map's id and are drawn with it again. Older builds
        don't report map ids; then each pose is assigned to the map that was
        active when it was tracked, using the restart times in self._map_starts.
        """
        if self._backend == "orbslam3" and hasattr(self._slam, "get_trajectory_points_with_map"):
            try:
                return [
                    (float(t), np.linalg.inv(np.asarray(tcw, dtype=np.float64)), int(map_id))
                    for t, tcw, map_id in self._slam.get_trajectory_points_with_map()
                ]
            except Exception:  # pylint: disable=broad-except
                return []
        return [
            (t, pose, bisect.bisect_right(self._map_starts, t))
            for t, pose in self._get_stamped_trajectory()
        ]

    def _current_map_key(self) -> int:
        """Key of the map the backend is working in now (see _get_map_trajectory)."""
        if hasattr(self._slam, "get_current_map_id"):
            try:
                map_id = int(self._slam.get_current_map_id())
                if map_id >= 0:
                    return map_id
            except Exception:  # pylint: disable=broad-except
                pass
        return len(self._map_starts)

    def _split_maps(
        self, traj: List[Tuple[float, np.ndarray, int]]
    ) -> Tuple[list, List[Tuple[int, list]]]:
        """
        (current map's poses, [(key, poses) of each earlier map, oldest first]).

        Each map has its own origin, heading and scale, so only the current map
        is drawn in the main view — overlaying them made every map look like a
        straight line from (0, 0). Earlier maps are drawn as thumbnails instead.
        """
        groups: Dict[int, list] = collections.OrderedDict()
        for _t, pose, key in traj:
            groups.setdefault(key, []).append(pose)
        current = groups.pop(self._current_map_key(), [])
        return current, list(groups.items())

    def _map_number(self) -> int:
        return self._current_map_key() + 1

    def _map_label(self, n_earlier: Optional[int] = None) -> str:
        """Current map number for the stream overlay."""
        n = len(self._map_starts) if n_earlier is None else n_earlier
        if n == 0:
            return f"map #{self._map_number()}"
        earlier = "1 earlier map" if n == 1 else f"{n} earlier maps"
        return f"map #{self._map_number()} · {earlier} in thumbnails"

    def _check_map_merge(self) -> None:
        """Log when the backend merges maps (the atlas shrinks)."""
        count = self._backend_map_count()
        if count is None:
            return
        if self._last_map_count is not None and count < self._last_map_count:
            logger.success(
                f"{_BACKEND_LABELS[self._backend]} merged an earlier map into the current one "
                f"after recognising a mapped place ({count} map(s) left)"
            )
        self._last_map_count = count

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
                f"{backend_label} abandoned the map — starting map #{self._map_number()}. "
                f"Its coordinates restart at (0, 0) facing forward; the old map is not "
                f"connected to it unless the robot revisits a mapped place.{atlas}"
            )

        teleop = getattr(self, "_teleop", None)
        relocalizing = _RELOCALIZING_LABEL.get(self._backend)
        if teleop is None or not self._config.auto_recover:
            return
        if label == relocalizing and prev not in (relocalizing, None):
            teleop.request_recovery(
                since=ts - _RECOVERY_LOOKBACK_S, max_total=self._recovery_replay_s()
            )
            logger.warning(
                f"{backend_label} lost tracking — release the keys and auto-recovery will "
                "reverse your last moves to find the mapped view again (any arrow key takes "
                "over). Tip: turn in short taps and drive forward between them."
            )
        elif prev == relocalizing and label != relocalizing:
            teleop.cancel_recovery()
            if label in ("OK", "OK_KLT"):
                logger.success(f"{backend_label} relocalized in map #{self._map_number()}")

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
        self._load()

        _stop = threading.Event()
        camera_kwargs = {}
        if self._config.max_exposure_ms is not None:
            camera_kwargs["exposure_max_ms"] = self._config.max_exposure_ms
            logger.info(f"Camera auto-exposure capped at {self._config.max_exposure_ms:g} ms")
        cam = self._open_camera(**camera_kwargs)

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
                t0 = time.perf_counter()
                if self._backend == "orbslam3":
                    state = self._process_frame_orbslam3(gray, ts)
                    label = _ORBSLAM3_STATES.get(state, "UNKNOWN")
                else:
                    state = self._process_frame_orbslam2(gray, ts)
                    label = _ORBSLAM2_STATES.get(state, "UNKNOWN")
                t_slam = time.perf_counter() - t0
                backend_label = _BACKEND_LABELS[self._backend]

                state_changed = label != self._last_state_label
                if state_changed:
                    logger.info(
                        f"{backend_label} state changed: {self._last_state_label} -> {label}  "
                        f"(frame={self._frame_idx}  t={ts - self._run_start_ts:.1f}s  "
                        f"map #{self._map_number()})"
                    )
                    self._handle_state_change(self._last_state_label, label, ts)
                    self._last_state_label = label
                self._check_map_merge()

                # get_trajectory_points() walks the whole map and copies it into Python —
                # not free, and it grows as the map does. Only pay for it on frames where
                # it's actually used (a state change, or an active stream), not every frame.
                t0 = time.perf_counter()
                need_traj = state_changed or (self._config.stream and self._server is not None)
                traj, earlier = (
                    self._split_maps(self._get_map_trajectory()) if need_traj else ([], [])
                )
                t_traj = time.perf_counter() - t0

                t0 = time.perf_counter()
                self._push_view(frame, gray, traj, earlier, label)
                t_view = time.perf_counter() - t0

                self._log_frame(
                    gray,
                    label,
                    dt,
                    (t_slam, t_traj, t_view),
                    traj if need_traj else None,
                    state_changed,
                )
                self._maybe_warn_stall(label)

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

    def _push_view(  # pylint: disable=too-many-positional-arguments
        self, frame: np.ndarray, gray: np.ndarray, traj: list, earlier: list, label: str
    ) -> None:
        """Render the map view (or --visualize debug view) into the MJPEG stream."""
        if not (self._config.stream and self._server is not None):
            return
        h, w = frame.shape[:2]
        scale = 2
        map_label = self._map_label(len(earlier))
        if self._config.visualize:
            keypoints, matches, inlier_mask, prev_frame, prev_kps = self._update_visualization(
                frame, gray
            )
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
                map_label,
                earlier,
            )
        else:
            view = self._render_map_view(
                traj, frame, label, self._backend, w * scale, h * scale, map_label, earlier
            )
        self._push_frame(view)

    def _log_frame(  # pylint: disable=too-many-positional-arguments
        self,
        gray: np.ndarray,
        label: str,
        dt: float,
        timings: Tuple[float, float, float],
        traj: Optional[list],
        state_changed: bool,
    ) -> None:
        """
        Per-frame DEBUG line. timings = (t_slam, t_traj, t_view) in seconds: time
        in the backend's process_image_mono, in fetching and splitting the
        trajectory, and in rendering + pushing the stream frame. traj is None on
        frames where the trajectory wasn't fetched.
        """
        backend_label = _BACKEND_LABELS[self._backend]
        t_slam, t_traj, t_view = timings
        timing = (
            f"  t_slam={t_slam * 1000:.0f}ms  t_traj={t_traj * 1000:.0f}ms"
            f"  t_view={t_view * 1000:.0f}ms"
        )
        traj_info = (
            ""
            if traj is None
            else f"  map=#{self._map_number()}  poses={len(traj)}{self._last_pose_xz(traj)}"
        )
        need_diag = state_changed or self._frame_idx % _DIAG_INTERVAL == 0

        # opt(lazy=True) defers evaluating every arg (incl. calling
        # _frame_diagnostics, which does a real ORB pass over the frame)
        # until loguru confirms a DEBUG-level sink is actually active —
        # so this costs nothing when only INFO/WARNING are enabled.
        logger.opt(lazy=True).debug(
            "{} frame={}  dt={:.0f}ms{}  state={}{}{}{}",
            lambda: backend_label,
            lambda: self._frame_idx,
            lambda: dt * 1000,
            lambda: timing,
            lambda: label,
            lambda: traj_info,
            lambda: self._init_diagnostics() if label == "NOT_INIT" else "",
            lambda: self._frame_diagnostics(gray) if need_diag else "",
        )

    def _maybe_warn_stall(self, label: str) -> None:
        """One warning if the map still hasn't initialized after _STALL_WARN_FRAMES frames."""
        if (
            label in ("NOT_INIT", "NO_IMAGES")
            and self._frame_idx == _STALL_WARN_FRAMES
            and not self._stall_warned
        ):
            self._stall_warned = True
            logger.warning(
                f"{_BACKEND_LABELS[self._backend]} still {label} after {_STALL_WARN_FRAMES} "
                "frames. Re-run with DEBUG logging enabled to see the per-frame diagnostics: "
                "cv2_orb_kpts near 0 means the camera feed itself is unusable "
                "(dark/blurry/low-texture); a healthy keypoint count with no init "
                "means the motion so far is degenerate for triangulation — drive "
                "with forward/backward translation, not pure in-place turns, "
                "until state changes to OK."
            )

    @staticmethod
    def _annotate_frame(
        cam_frame: np.ndarray,
        state_label: str,
        backend: str,
    ) -> np.ndarray:
        h, w = cam_frame.shape[:2]
        return SlamMapper._render_map_view([], cam_frame, state_label, backend, w, h)
