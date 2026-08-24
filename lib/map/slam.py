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
from typing import Any, Optional

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera_motion_mixin import CameraMotionMixIn
from lib.settings import PROJECT_ROOT_PATH, ensure_user_config, user_config_path

_VALID_BACKENDS = ["orbslam2"]

# Tracking state labels used by ORB-SLAM2 (ORB_SLAM2::Tracking::eTrackingState)
_ORBSLAM2_STATES = {-1: "NOT_READY", 0: "NO_IMAGES", 1: "NOT_INIT", 2: "OK", 3: "LOST"}

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
    """

    config_path: Path = user_config_path("models/slam.yaml")
    stream: bool = False
    stream_port: int = Field(8080, gt=1024, lt=65535)
    speed: float = Field(0.3, ge=0.0, le=1.0)
    turn_gain: float = Field(0.5, ge=0.0, le=1.0)
    visualize: bool = False

    @validator("config_path", always=True)
    def config_must_exist(cls, v: Path) -> Path:  # pylint: disable=no-self-argument
        v = ensure_user_config(v)
        if not v.exists():
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

        vocab = Path(cfg.get("vocabulary", "assets/models/ORBvoc.txt"))
        if not vocab.is_absolute():
            vocab = PROJECT_ROOT_PATH / vocab
        if not vocab.exists():
            raise FileNotFoundError(
                f"ORB vocabulary not found: {vocab}\n"
                "Download ORBvoc.txt from github.com/raulmur/ORB_SLAM2/tree/master/Vocabulary"
            )
        settings_rel = cfg.get("settings", "config/models/orbslam2_mono.yaml")
        settings = Path(settings_rel)
        if not settings.is_absolute():
            # This one's a config/*.yaml file (camera calibration), not an
            # assets/ blob like vocab above — route it through the same
            # user-config seeding as every other config_path, so a hand-tuned
            # calibration survives a package upgrade. Anything relative but
            # NOT rooted at "config/" (a custom path someone set explicitly)
            # falls back to plain PROJECT_ROOT_PATH anchoring.
            settings = (
                ensure_user_config(user_config_path(settings_rel[len("config/") :]))
                if settings_rel.startswith("config/")
                else PROJECT_ROOT_PATH / settings
            )
        if not settings.exists():
            raise FileNotFoundError(
                f"ORB-SLAM2 settings not found: {settings}\n"
                "Create camera calibration YAML at config/models/orbslam2_mono.yaml"
            )
        self._slam = orbslam2.System(str(vocab), str(settings), orbslam2.Sensor.MONOCULAR)
        self._slam.set_use_viewer(False)
        # System() only records the vocab/settings paths — initialize() is what actually
        # loads the ORB vocabulary and starts tracking/mapping/loop-closing. Without this
        # call process_image_mono() runs against a system that was never started, so the
        # tracking state stays NO_IMAGES_YET forever regardless of what frames come in.
        self._slam.initialize()
        logger.success(f"ORB-SLAM2 initialised — vocab={vocab}  settings={settings}")

    def _process_frame_orbslam2(self, gray: np.ndarray, timestamp: float) -> int:
        # process_image_mono() returns a bool — whether *this frame* tracked
        # successfully — not the tracking-state enum. Casting that bool through
        # _ORBSLAM2_STATES only ever produced NO_IMAGES(0)/NOT_INIT(1); OK/LOST
        # could never appear. get_tracking_state() is the real state accessor.
        self._slam.process_image_mono(gray, timestamp)
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
                state = self._process_frame_orbslam2(gray, ts)
                label = _ORBSLAM2_STATES.get(state, "UNKNOWN")

                # get_trajectory_points() walks the whole map and copies it into Python —
                # not free, and it grows as the map does. Only pay for it on frames where
                # it's actually used (a state change, or an active stream), not every frame.
                state_changed = label != self._last_state_label
                need_traj = state_changed or (self._config.stream and self._server is not None)
                traj = self._get_trajectory() if need_traj else []

                if state_changed:
                    logger.info(
                        f"ORB-SLAM2 state changed: {self._last_state_label} -> {label}  "
                        f"(frame={self._frame_idx}  t={ts - self._run_start_ts:.1f}s)"
                    )
                    self._last_state_label = label

                traj_info = f"  poses={len(traj)}{self._last_pose_xz(traj)}" if need_traj else ""
                need_diag = state_changed or self._frame_idx % _DIAG_INTERVAL == 0

                # opt(lazy=True) defers evaluating every arg (incl. calling
                # _frame_diagnostics, which does a real ORB pass over the frame)
                # until loguru confirms a DEBUG-level sink is actually active —
                # so this costs nothing when only INFO/WARNING are enabled.
                logger.opt(lazy=True).debug(
                    "ORB-SLAM2 frame={}  dt={:.0f}ms  state={}{}{}",
                    lambda: self._frame_idx,
                    lambda dt=dt: dt * 1000,
                    lambda label=label: label,
                    lambda traj_info=traj_info: traj_info,
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
                        f"ORB-SLAM2 still {label} after {_STALL_WARN_FRAMES} frames. "
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
                        )
                    else:
                        view = self._render_map_view(
                            traj, frame, label, self._backend, w * scale, h * scale
                        )
                    self._push_frame(view)

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
    ) -> np.ndarray:
        """
        Debug view for --visualize:
            top:    trajectory map | detected ORB keypoints
            bottom: frame-to-frame matches, full width (green=raw, blue=RANSAC inliers)
        """
        h, w = frame.shape[:2]

        map_panel = SlamMapper._render_map_view(traj, frame, state_label, backend, w, h)
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
