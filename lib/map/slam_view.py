"""
Stream rendering for SlamMapper (lib/map/slam.py): the top-down map view with
earlier-map thumbnails, and the --visualize debug view.

The --visualize overlay runs its own throwaway ORB detect + match pass
(MatchVisualizer), because neither SLAM backend's bindings expose the features
or matches the backend itself used.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

# --visualize overlay: features per frame and Lowe ratio-test threshold.
_VIZ_MAX_FEATURES = 500
_VIZ_MATCH_RATIO = 0.75
# Map view: heading arrow length in pixels.
_HEADING_ARROW_PX = 30
# Map view: how many earlier maps get a thumbnail (most recent ones).
_MAX_THUMBNAILS = 3


class MatchVisualizer:
    """Frame-to-frame ORB matches for the --visualize overlay (display only)."""

    def __init__(self) -> None:
        self._detector = None
        self._matcher = None
        self._prev_frame = None
        self._prev_kps = None
        self._prev_desc = None

    def update(self, frame: np.ndarray, gray: np.ndarray) -> tuple:
        """
        Detect ORB keypoints in the current frame, ratio-test match them against
        the previous frame, then RANSAC-filter those matches via the fundamental
        matrix. Returns (keypoints, matches, inlier_mask, prev_frame, prev_kps) —
        prev_frame/prev_kps/inlier_mask are None on the first call (or whenever
        there aren't enough matches to fit a fundamental matrix).
        """
        if self._detector is None:
            self._detector = cv2.ORB_create(nfeatures=_VIZ_MAX_FEATURES)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        keypoints, desc = self._detector.detectAndCompute(gray, None)
        matches = []
        inlier_mask = None
        if self._prev_desc is not None and desc is not None:
            raw = self._matcher.knnMatch(self._prev_desc, desc, k=2)
            for pair in raw:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < _VIZ_MATCH_RATIO * n.distance:
                        matches.append(m)
            inlier_mask = ransac_inlier_mask(self._prev_kps, keypoints, matches)

        prev_frame, prev_kps = self._prev_frame, self._prev_kps
        self._prev_frame, self._prev_kps, self._prev_desc = frame.copy(), keypoints, desc
        return keypoints, matches, inlier_mask, prev_frame, prev_kps


def ransac_inlier_mask(prev_kps: list, keypoints: list, matches: list) -> Optional[np.ndarray]:
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


def render_map_view(  # pylint: disable=too-many-positional-arguments
    traj: list,
    cam_frame: np.ndarray,
    state_label: str,
    backend: str,
    canvas_w: int,
    canvas_h: int,
    map_label: str = "",
    earlier: Optional[List[Tuple[int, list]]] = None,
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

    earlier: (map key, poses) of earlier maps, oldest first. The most recent
    _MAX_THUMBNAILS are drawn as thumbnails along the bottom right, each in
    its own axes, since their coordinates aren't related to the current map.
    """
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    try:
        draw_path(canvas, traj, (0, 0, canvas_w, canvas_h), 40, _HEADING_ARROW_PX, bright=True)
        draw_thumbnails(canvas, (earlier or [])[-_MAX_THUMBNAILS:])
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


def draw_path(  # pylint: disable=too-many-positional-arguments
    canvas: np.ndarray,
    poses: list,
    box: Tuple[int, int, int, int],
    pad: int,
    arrow_px: int,
    bright: bool,
) -> None:
    """
    Draw a top-down path (+z up, +x right) scaled to fit box = (x, y, w, h):
    line, hollow start circle, filled current-position dot and heading arrow.
    """
    if not poses:
        return
    bx, by, bw, bh = box
    positions = np.array([[T[0, 3], T[2, 3]] for T in poses], dtype=np.float32)
    x_range = float(np.ptp(positions[:, 0])) or 1.0
    z_range = float(np.ptp(positions[:, 1])) or 1.0
    scale = min((bw - 2 * pad) / x_range, (bh - 2 * pad) / z_range)
    x_min, z_min = positions[:, 0].min(), positions[:, 1].min()
    pts = [
        (int(bx + pad + (x - x_min) * scale), int(by + bh - pad - (z - z_min) * scale))
        for x, z in positions
    ]
    line = (0, 180, 0) if bright else (0, 120, 0)
    dot = (0, 255, 0) if bright else (0, 170, 0)
    radius = 5 if bright else 3
    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i - 1], pts[i], line, 1)
    cv2.circle(canvas, pts[0], radius, line, 1)
    cv2.circle(canvas, pts[-1], radius, dot, -1)

    # Heading: the camera's optical axis (+z in the camera frame) is the
    # third column of the camera-to-world rotation; drop its height.
    fwd = np.asarray(poses[-1], dtype=np.float64)[:3, 2]
    norm = float(np.hypot(fwd[0], fwd[2]))
    if norm > 1e-6:
        tip = (
            int(pts[-1][0] + arrow_px * fwd[0] / norm),
            int(pts[-1][1] - arrow_px * fwd[2] / norm),
        )
        color = (0, 255, 255) if bright else (0, 170, 170)
        cv2.arrowedLine(canvas, pts[-1], tip, color, 2 if bright else 1, cv2.LINE_AA, 0, 0.3)


def draw_thumbnails(canvas: np.ndarray, earlier: List[Tuple[int, list]]) -> None:
    """Earlier maps, oldest left, along the bottom right of the canvas."""
    if not earlier:
        return
    canvas_h, canvas_w = canvas.shape[:2]
    tw, th, margin = canvas_w // 5, canvas_h // 5, 8
    y = canvas_h - th - margin
    for i, (key, poses) in enumerate(earlier):
        x = canvas_w - margin - (len(earlier) - i) * (tw + margin)
        canvas[y : y + th, x : x + tw] = (25, 25, 25)
        cv2.rectangle(canvas, (x, y), (x + tw - 1, y + th - 1), (80, 80, 80), 1)
        draw_path(canvas, poses, (x, y, tw, th), 10, 12, bright=False)
        cv2.putText(
            canvas,
            f"#{key + 1}",
            (x + 5, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )


def render_features_panel(frame: np.ndarray, keypoints: list) -> np.ndarray:
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


def render_matches_panel(  # pylint: disable=too-many-positional-arguments
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


def render_visualization_view(  # pylint: disable=too-many-positional-arguments
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
    earlier: Optional[List[Tuple[int, list]]] = None,
) -> np.ndarray:
    """
    Debug view for --visualize:
        top:    trajectory map | detected ORB keypoints
        bottom: frame-to-frame matches, full width (green=raw, blue=RANSAC inliers)
    """
    h, w = frame.shape[:2]

    map_panel = render_map_view(traj, frame, state_label, backend, w, h, map_label, earlier)
    kp_panel = render_features_panel(frame, keypoints)
    match_panel = render_matches_panel(
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
