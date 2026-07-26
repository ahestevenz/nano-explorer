#!/usr/bin/env python3
"""
Collect road-following training data by driving the robot with arrow keys.

Each frame captured while a key is held is saved with its steering label:
  LEFT  -> -1.0   (full left)
  RIGHT -> +1.0   (full right)
  UP    ->  0.0   (straight ahead)
  DOWN  -> ignored (no label for reverse)

Output layout
-------------
  <out_dir>/
    labels.csv      # filename,angle
    img_000001.jpg
    img_000002.jpg
    ...

Usage
-----
  python tools/collect_road_data.py --out datasets/road_001
  python tools/collect_road_data.py --out datasets/road_001 --speed 0.25 --fps 10
  python tools/collect_road_data.py --out datasets/road_001 --stream
  python tools/collect_road_data.py --out datasets/road_001 --stream --stream-port 8080
"""

import argparse
import csv
import os
import sys
import threading
from pathlib import Path

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

_KEY_TIMEOUT = 0.15  # seconds after last keypress before treating as "released"
_ARROW_MAP = {
    b"\x1b[A": ("forward", 0.0),
    b"\x1b[D": ("left", -1.0),
    b"\x1b[C": ("right", 1.0),
}


def _fire_stop_timer(motors, state, stop_timer):
    """Cancel any pending stop timer and schedule a new one."""
    if stop_timer is not None:
        stop_timer.cancel()

    def _do_stop():
        state["action"] = "stop"
        motors.stop()

    t = threading.Timer(_KEY_TIMEOUT, _do_stop)
    t.daemon = True
    t.start()
    return t


def _apply_key(motors, state, speed, turn_gain, action, angle):
    """Update drive state and send motor command for one keypress."""
    state["action"] = action
    state["angle"] = angle
    turn = speed * turn_gain
    if action == "forward":
        motors.forward(speed)
    elif action == "left":
        motors.turn_left(turn)
    elif action == "right":
        motors.turn_right(turn)


def _annotate(frame, count, state):
    """Overlay collection stats onto a copy of frame."""
    import cv2

    out = frame.copy()
    cv2.putText(
        out,
        f"frames={count[0]}  angle={state['angle']:+.1f}  {state['action']}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _write_frame(cam, out_dir, count, state, writer, csv_file, server):
    """Capture one frame, save it, append its label to the CSV, and push to stream."""
    import cv2

    frame = cam.read()
    fname = f"img_{count[0]:06d}.jpg"
    cv2.imwrite(str(out_dir / fname), frame)
    writer.writerow([fname, f"{state['angle']:.4f}"])
    csv_file.flush()
    count[0] += 1
    sys.stdout.write(
        f"\r[collect] frames={count[0]}  action={state['action']:<7}  "
        f"angle={state['angle']:+.1f}   "
    )
    sys.stdout.flush()
    if server is not None:
        server.frame_buffer.put(_annotate(frame, count, state))


def _run_loop(
    fd,
    cam,
    motors,
    out_dir,
    state,
    count,
    writer,
    csv_file,
    frame_interval,
    speed,
    turn_gain,
    server,
):
    """Keyboard + frame-capture loop. Returns when the user presses q."""
    import select
    import time

    stop_timer = None
    next_frame = time.monotonic()
    try:
        while True:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
            if rlist:
                chunk = os.read(fd, 3)
                if chunk in (b"q", b"Q", b"\x03"):
                    break
                mapping = _ARROW_MAP.get(chunk)
                if mapping:
                    _apply_key(motors, state, speed, turn_gain, *mapping)
                    stop_timer = _fire_stop_timer(motors, state, stop_timer)

            now = time.monotonic()
            if now >= next_frame and state["action"] != "stop":
                _write_frame(cam, out_dir, count, state, writer, csv_file, server)
                next_frame = now + frame_interval
            else:
                if server is not None:
                    server.frame_buffer.put(_annotate(cam.read(), count, state))
                time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        if stop_timer is not None:
            stop_timer.cancel()


def collect(
    out_dir: Path,
    speed: float,
    turn_gain: float,
    fps: int,
    camera_source: str,
    stream: bool,
    stream_port: int,
):
    import termios
    import tty

    from lib.camera import Camera, MjpegServer
    from lib.motor import MotorController
    from lib.network import get_wifi_ip

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "labels.csv"
    existing = csv_path.exists()

    motors = MotorController()
    cam = Camera(source=camera_source, fps=fps)
    motors.open()
    cam.open()

    server = None
    if stream:
        server = MjpegServer(port=stream_port)
        server.start()
        ip = get_wifi_ip() or "<nano-ip>"
        print(f"[collect] Stream -> http://{ip}:{stream_port}/stream")

    state = {"action": "stop", "angle": 0.0}
    count = [0]

    print(
        f"\n[collect] Saving to {out_dir}\n"
        f"[collect] Arrow keys: UP=straight  LEFT=left  RIGHT=right\n"
        f"[collect] 'q' or Ctrl-C to stop\n"
    )

    fd = sys.stdin.fileno()
    old_attr = termios.tcgetattr(fd)

    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not existing:
            writer.writerow(["filename", "angle"])
        try:
            tty.setraw(fd)
            _run_loop(
                fd,
                cam,
                motors,
                out_dir,
                state,
                count,
                writer,
                csv_file,
                1.0 / fps,
                speed,
                turn_gain,
                server,
            )
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
            motors.stop()
            motors.close()
            if server is not None:
                server.stop()
            cam.release()
            print(f"\n[collect] Saved {count[0]} frames to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect road-following training data via keyboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", required=True, type=Path, help="Output dataset directory")
    parser.add_argument("--speed", type=float, default=0.25, help="Motor speed [0,1]")
    parser.add_argument(
        "--turn-gain", type=float, default=0.6, dest="turn_gain", help="Turn gain [0,1]"
    )
    parser.add_argument("--fps", type=int, default=10, help="Frame capture rate")
    parser.add_argument(
        "--camera",
        default="csi",
        dest="camera_source",
        choices=["csi", "usb"],
        help="Camera source",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Enable MJPEG stream (view in browser)",
    )
    parser.add_argument(
        "--stream-port",
        type=int,
        default=8080,
        dest="stream_port",
        help="MJPEG server port",
    )
    args = parser.parse_args()

    collect(
        out_dir=args.out,
        speed=args.speed,
        turn_gain=args.turn_gain,
        fps=args.fps,
        camera_source=args.camera_source,
        stream=args.stream,
        stream_port=args.stream_port,
    )


if __name__ == "__main__":
    main()
