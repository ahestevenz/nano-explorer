#!/usr/bin/env python3
"""
Collect collision-avoidance training data by driving the robot with arrow keys.

Drive the robot to a position, then press:
  f / F  ->  save current frame as "free"   (safe to proceed)
  b / B  ->  save current frame as "blocked" (obstacle ahead)
  Arrow keys -> drive the robot (held = moving, released = stop)
  q / Ctrl-C -> quit

Output layout
-------------
  <out_dir>/
    free/
      img_000001.jpg
      img_000002.jpg
      ...
    blocked/
      img_000001.jpg
      ...

Usage
-----
  python tools/collect_collision_data.py --out datasets/collision_001
  python tools/collect_collision_data.py --out datasets/collision_001 --speed 0.25
  python tools/collect_collision_data.py --out datasets/collision_001 --stream
  python tools/collect_collision_data.py --out datasets/collision_001 --stream --stream-port 8080
"""

import argparse
import os
import sys
import threading
from pathlib import Path

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

_KEY_TIMEOUT = 0.15  # seconds after last keypress before treating as "released"
_ARROW_MAP = {
    b"\x1b[A": "forward",
    b"\x1b[B": "backward",
    b"\x1b[D": "left",
    b"\x1b[C": "right",
}


def _fire_stop_timer(motors, stop_timer):
    """Cancel any pending stop timer and schedule a new one."""
    if stop_timer is not None:
        stop_timer.cancel()
    t = threading.Timer(_KEY_TIMEOUT, motors.stop)
    t.daemon = True
    t.start()
    return t


def _drive(motors, speed, turn_gain, action):
    """Send a motor command for one keypress."""
    turn = speed * turn_gain
    if action == "forward":
        motors.forward(speed)
    elif action == "backward":
        motors.backward(speed * 0.5)
    elif action == "left":
        motors.turn_left(turn)
    elif action == "right":
        motors.turn_right(turn)


def _annotate(frame, counts, label=None):
    """Overlay collection stats onto a copy of frame."""
    import cv2

    out = frame.copy()
    stats = f"free={counts['free']}  blocked={counts['blocked']}"
    cv2.putText(out, stats, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if label is not None:
        color = (0, 255, 0) if label == "free" else (0, 0, 255)
        cv2.putText(
            out, label.upper(), (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA
        )
    return out


def _save_frame(cam, counts, dirs, label, server):
    """Capture one frame, save it, and push an annotated version to the stream."""
    import cv2

    idx = counts[label]
    fname = f"img_{idx:06d}.jpg"
    frame = cam.read()
    cv2.imwrite(str(dirs[label] / fname), frame)
    counts[label] += 1
    sys.stdout.write(
        f"\r[collect] free={counts['free']}  blocked={counts['blocked']}  " f"saved={label:<7}  "
    )
    sys.stdout.flush()
    if server is not None:
        server.frame_buffer.put(_annotate(frame, counts, label))


def _run_loop(fd, cam, motors, counts, dirs, speed, turn_gain, server):
    """Keyboard loop: arrow keys drive, f/b capture, q quits."""
    import select

    stop_timer = None
    try:
        while True:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.02)
            if not rlist:
                if server is not None:
                    server.frame_buffer.put(_annotate(cam.read(), counts))
                continue

            chunk = os.read(fd, 3)

            if chunk in (b"q", b"Q", b"\x03"):
                break
            if chunk in (b"f", b"F"):
                _save_frame(cam, counts, dirs, "free", server)
                continue
            if chunk in (b"b", b"B"):
                _save_frame(cam, counts, dirs, "blocked", server)
                continue

            action = _ARROW_MAP.get(chunk)
            if action:
                _drive(motors, speed, turn_gain, action)
                stop_timer = _fire_stop_timer(motors, stop_timer)

    except KeyboardInterrupt:
        pass
    finally:
        if stop_timer is not None:
            stop_timer.cancel()


def collect(
    out_dir: Path,
    speed: float,
    turn_gain: float,
    camera_source: str,
    stream: bool,
    stream_port: int,
):
    import termios
    import tty

    from lib.camera import Camera, MjpegServer
    from lib.motor import MotorController
    from lib.network import get_wifi_ip

    free_dir = out_dir / "free"
    blocked_dir = out_dir / "blocked"
    free_dir.mkdir(parents=True, exist_ok=True)
    blocked_dir.mkdir(parents=True, exist_ok=True)

    dirs = {"free": free_dir, "blocked": blocked_dir}
    counts = {
        "free": len(list(free_dir.glob("*.jpg"))),
        "blocked": len(list(blocked_dir.glob("*.jpg"))),
    }

    motors = MotorController()
    cam = Camera(source=camera_source)
    motors.open()
    cam.open()

    server = None
    if stream:
        server = MjpegServer(port=stream_port)
        server.start()
        ip = get_wifi_ip() or "<nano-ip>"
        print(f"[collect] Stream -> http://{ip}:{stream_port}/stream")

    print(
        f"\n[collect] Saving to {out_dir}\n"
        f"[collect] Arrow keys: drive robot to position\n"
        f"[collect]   f = capture FREE frame\n"
        f"[collect]   b = capture BLOCKED frame\n"
        f"[collect]   q or Ctrl-C = quit\n"
        f"[collect] Resuming: free={counts['free']}  blocked={counts['blocked']}\n"
    )

    fd = sys.stdin.fileno()
    old_attr = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        _run_loop(fd, cam, motors, counts, dirs, speed, turn_gain, server)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
        motors.stop()
        motors.close()
        if server is not None:
            server.stop()
        cam.release()
        print(f"\n[collect] Done — free={counts['free']}  blocked={counts['blocked']}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect collision-avoidance training data via keyboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", required=True, type=Path, help="Output dataset directory")
    parser.add_argument("--speed", type=float, default=0.25, help="Motor speed [0,1]")
    parser.add_argument(
        "--turn-gain", type=float, default=0.6, dest="turn_gain", help="Turn gain [0,1]"
    )
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
        camera_source=args.camera_source,
        stream=args.stream,
        stream_port=args.stream_port,
    )


if __name__ == "__main__":
    main()
