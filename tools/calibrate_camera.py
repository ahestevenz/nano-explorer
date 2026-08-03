#!/usr/bin/env python3
"""
Calibrate the JetBot's camera intrinsics with a checkerboard, for ORB-SLAM2.

config/models/orbslam2_mono.yaml ships with placeholder intrinsics
(fx=fy=700, cx=320, cy=240, zero distortion) — monocular ORB-SLAM2's
initialization is quite sensitive to these being right, and wrong values
are a common reason it keeps rejecting its own initial map ("Wrong
initialization, reseting..."). This runs a standard OpenCV chessboard
calibration and writes real fx/fy/cx/cy/k1/k2/p1/p2 back into that file.

Print any standard OpenCV chessboard pattern (e.g. the 9x6-inner-corner
one from https://github.com/opencv/opencv/blob/4.x/doc/pattern.png),
mount it on something flat and rigid, and have it in frame.

Live view is streamed over MJPEG so you can position the board without a
monitor attached to the Nano.

Controls
--------
  c / C       -> capture the current frame as a calibration sample
                 (only registers if a board was detected in it)
  q / Ctrl-C  -> stop capturing and run calibration

How many captures?
-------------------
--samples defaults to 15 — a good baseline for a typical CSI/webcam setup.
Calibration refuses to run below 5 (too unreliable to trust); 20+ is worth
it if the reported reprojection error comes back marginal (over ~0.5-1.0 px).
Count matters less than variety — spread captures across near/far, off-center
positions (not just dead-center), and tilted angles in both axes, so the
solver can actually disambiguate fx/fy/cx/cy from the distortion terms.

Procedure
---------
  1. Print a chessboard pattern and mount it flat and rigid (see above).
  2. Run this script. Pass --board-cols/--board-rows if your pattern isn't
     the default 9x6 inner corners, and --camera usb / --stream-port PORT
     if you're not on the default CSI camera / port 8080.
  3. Open the printed stream URL (http://<nano-ip>:PORT/stream) in a browser.
  4. Hold the board in view — a green "BOARD FOUND" overlay means it's
     detected. Press c to capture, then move the board to a new position,
     angle, or distance and repeat (see "How many captures?" above).
  5. Press q (or Ctrl-C) once you've captured enough samples — this stops
     capturing and runs the fit.
  6. Check the printed mean reprojection error, then either copy the
     written config/models/orbslam2_mono.calibrated.yaml over
     config/models/orbslam2_mono.yaml yourself, or re-run with
     --update-config to have it back up the current config and apply the
     new values automatically.
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

_ORBSLAM2_CONFIG = Path(__file__).resolve().parent.parent / "config/models/orbslam2_mono.yaml"
_MIN_SAMPLES = 5  # calibrateCamera will run below --samples, but not below this


def _find_corners(gray, board_size, criteria):
    """Return refined corners if a full board is visible in gray, else None."""
    import cv2

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, board_size, flags=flags)
    if not found:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def _annotate(frame, board_size, corners, num_samples, target_samples):
    """Draw detected corners (if any) and a status line onto a copy of frame."""
    import cv2

    out = frame.copy()
    if corners is not None:
        cv2.drawChessboardCorners(out, board_size, corners, True)
        status, color = "BOARD FOUND — press c to capture", (0, 255, 0)
    else:
        status, color = "no board detected", (0, 0, 255)
    cv2.putText(out, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(
        out,
        f"samples: {num_samples}/{target_samples}",
        (8, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _capture_loop(fd, cam, server, board_size, target_samples):
    """Keyboard loop: c captures a sample, q finishes. Returns (objpoints, imgpoints, image_size)."""
    import select

    import cv2
    import numpy as np

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cols, rows = board_size
    # Square size doesn't matter here — it only rescales tvecs, not the fitted
    # camera matrix or distortion coefficients we actually care about.
    board_objp = np.zeros((rows * cols, 3), np.float32)
    board_objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)  # pylint: disable=no-member

    objpoints, imgpoints = [], []
    image_size = None

    try:
        while True:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.02)

            frame = cam.read()
            image_size = frame.shape[1::-1]  # (w, h)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = _find_corners(gray, board_size, criteria)

            if server is not None:
                server.frame_buffer.put(
                    _annotate(frame, board_size, corners, len(objpoints), target_samples)
                )

            if not rlist:
                continue

            chunk = os.read(fd, 3)
            if chunk in (b"q", b"Q", b"\x03"):
                break
            if chunk in (b"c", b"C"):
                if corners is None:
                    sys.stdout.write("\r[calibrate] no board detected — try again          ")
                else:
                    objpoints.append(board_objp)
                    imgpoints.append(corners)
                    sys.stdout.write(
                        f"\r[calibrate] captured {len(objpoints)}/{target_samples}          "
                    )
                sys.stdout.flush()

    except KeyboardInterrupt:
        pass

    return objpoints, imgpoints, image_size


def _run_calibration(objpoints, imgpoints, image_size):
    """Fit intrinsics + distortion and report mean reprojection error (px)."""
    import cv2

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    if not ret:
        raise RuntimeError("cv2.calibrateCamera failed to converge")

    total_error = 0.0
    for i, objp in enumerate(objpoints):
        proj, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], mtx, dist)
        total_error += cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
    mean_error = total_error / len(objpoints)

    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]
    k1, k2, p1, p2 = dist.ravel()[:4]
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
    }, mean_error


def _patch_camera_block(text: str, values: dict) -> str:
    """Replace the Camera.fx/fy/cx/cy/k1/k2/p1/p2 lines in an orbslam2_mono.yaml body."""
    for key in ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"):
        pattern = rf"^(Camera\.{key}:\s*)[-\d.eE]+"
        replacement = rf"\g<1>{values[key]:.6f}"
        text, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if n == 0:
            raise ValueError(f"Camera.{key} line not found — file format may have changed")
    return text


def calibrate(
    board_cols: int,
    board_rows: int,
    target_samples: int,
    camera_source: str,
    stream_port: int,
    out_path: Path,
    update_config: bool,
):
    import termios
    import tty

    from lib.camera import Camera, MjpegServer
    from lib.network import get_wifi_ip

    board_size = (board_cols, board_rows)
    cam = Camera(source=camera_source)
    cam.open()

    server = MjpegServer(port=stream_port)
    server.start()
    ip = get_wifi_ip() or "<nano-ip>"
    print(f"[calibrate] Stream -> http://{ip}:{stream_port}/stream")
    print(
        f"\n[calibrate] Show a {board_cols}x{board_rows}-inner-corner checkerboard to the camera.\n"
        f"[calibrate] Move it to different positions, angles and distances between captures.\n"
        f"[calibrate]   c = capture sample (needs a detected board)\n"
        f"[calibrate]   q or Ctrl-C = finish and calibrate\n"
    )

    fd = sys.stdin.fileno()
    old_attr = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        objpoints, imgpoints, image_size = _capture_loop(
            fd, cam, server, board_size, target_samples
        )
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
        server.stop()
        cam.release()

    print(f"\n[calibrate] {len(objpoints)} sample(s) captured.")
    if len(objpoints) < _MIN_SAMPLES:
        print(
            f"[calibrate] Need at least {_MIN_SAMPLES} samples (ideally {target_samples}+) "
            "for a reliable calibration — aborting without writing anything."
        )
        return

    values, mean_error = _run_calibration(objpoints, imgpoints, image_size)
    print(
        "[calibrate] Result "
        f"(mean reprojection error: {mean_error:.3f} px — under ~0.5 is good, over ~1.0 is suspect):\n"
        f"  fx={values['fx']:.2f}  fy={values['fy']:.2f}  cx={values['cx']:.2f}  cy={values['cy']:.2f}\n"
        f"  k1={values['k1']:.5f}  k2={values['k2']:.5f}  p1={values['p1']:.5f}  p2={values['p2']:.5f}"
    )

    original = _ORBSLAM2_CONFIG.read_text(encoding="utf-8")
    patched = _patch_camera_block(original, values)
    out_path.write_text(patched, encoding="utf-8")
    print(f"[calibrate] Wrote {out_path}")

    if update_config:
        backup = _ORBSLAM2_CONFIG.with_suffix(
            f".yaml.bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        backup.write_text(original, encoding="utf-8")
        _ORBSLAM2_CONFIG.write_text(patched, encoding="utf-8")
        print(f"[calibrate] Backed up previous config to {backup}")
        print(f"[calibrate] Updated {_ORBSLAM2_CONFIG} in place")
    else:
        print(
            f"[calibrate] Review {out_path}, then copy it over {_ORBSLAM2_CONFIG}\n"
            "[calibrate] (or re-run with --update-config to do that automatically)."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate camera intrinsics for ORB-SLAM2 with a checkerboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--board-cols", type=int, default=9, help="Inner corners along the board's long axis"
    )
    parser.add_argument(
        "--board-rows", type=int, default=6, help="Inner corners along the board's short axis"
    )
    parser.add_argument(
        "--samples", type=int, default=15, dest="target_samples", help="Target number of captures"
    )
    parser.add_argument(
        "--camera",
        default="csi",
        dest="camera_source",
        choices=["csi", "usb"],
        help="Camera source",
    )
    parser.add_argument(
        "--stream-port", type=int, default=8080, dest="stream_port", help="MJPEG server port"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_ORBSLAM2_CONFIG.with_name("orbslam2_mono.calibrated.yaml"),
        help="Where to write the calibrated config",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        default=False,
        dest="update_config",
        help=f"Also back up and patch {_ORBSLAM2_CONFIG} in place",
    )
    args = parser.parse_args()

    calibrate(
        board_cols=args.board_cols,
        board_rows=args.board_rows,
        target_samples=args.target_samples,
        camera_source=args.camera_source,
        stream_port=args.stream_port,
        out_path=args.out,
        update_config=args.update_config,
    )


if __name__ == "__main__":
    main()
