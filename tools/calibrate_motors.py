#!/usr/bin/env python3
"""
Calibrate per-wheel motor power trim so the JetBot drives straight.

Cheap DC gear motors commonly differ 5-15% in actual speed at the same
commanded value, even between two units of the exact same model, due to
manufacturing tolerance, gearbox friction, and wheel/tire differences.
lib/motor.py sends both wheels the same commanded speed with no
compensation, so this shows up as the robot arcing to one side when told
to drive straight. This tool finds a left/right trim multiplier
(NANO_MOTOR_LEFT_TRIM / NANO_MOTOR_RIGHT_TRIM) that corrects it.

Trim only ever reduces the stronger wheel's power (each trim stays in
(0.0, 1.0]) — it can't push a wheel past what was actually commanded,
only bring the stronger one down to match the weaker one.

Controls
--------
  f / space   -> drive forward for --duration seconds at --speed, then stop
  right arrow -> "it drifted RIGHT last run" (nudges trim to compensate)
  left arrow  -> "it drifted LEFT last run"  (nudges trim to compensate)
  +           -> increase the nudge step size
  -           -> decrease the nudge step size
  s           -> save the current trim to ~/.nano-explorer.env and exit
  q / Ctrl-C  -> quit without saving

Procedure
---------
  1. Put the robot on the floor with room to roll forward a meter or two.
  2. Run this script (add --speed/--duration to match how you actually
     drive — the drift can vary with speed).
  3. Press f to run a forward test. Watch which way the nose drifts.
  4. Press the arrow key matching the drift direction (drifts left ->
     left arrow, drifts right -> right arrow). Repeat from step 3.
  5. Once it runs close to straight, press s to save. This writes
     NANO_MOTOR_LEFT_TRIM / NANO_MOTOR_RIGHT_TRIM to ~/.nano-explorer.env
     (backing up any existing file first), so every command that builds
     a MotorController picks up the correction automatically.

Note: this always starts from a clean 1.0/1.0 (no trim), not whatever is
currently saved — if you're refining an existing calibration rather than
starting over, the current values are printed at startup so you can nudge
from there with the same key presses.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

_ENV_FILE = Path.home() / ".nano-explorer.env"
_MAX_BIAS = 0.5  # keeps both trims within settings.py's (0.0, 1.0] bound


def _clamp_bias(bias: float) -> float:
    return max(-_MAX_BIAS, min(_MAX_BIAS, bias))


def _trims_from_bias(bias: float):
    """bias > 0 corrects a rightward drift (reduces left); bias < 0 corrects leftward."""
    left = 1.0 - max(bias, 0.0)
    right = 1.0 - max(-bias, 0.0)
    return left, right


def _read_key(fd) -> str:
    """Blocking single-keypress read; resolves arrow-key escape sequences to left/right."""
    import select

    chunk = os.read(fd, 1)
    if chunk != b"\x1b":
        return chunk.decode(errors="ignore")

    # Arrow keys are ESC [ A/B/C/D — the rest should follow within a few ms;
    # a short select() avoids blocking forever on a bare Esc keypress.
    rest = b""
    while len(rest) < 2:
        ready, _, _ = select.select([fd], [], [], 0.05)
        if not ready:
            break
        rest += os.read(fd, 2 - len(rest))
    return {b"[C": "right", b"[D": "left"}.get(rest, "")


def _run_forward(controller, speed: float, duration: float) -> None:
    print(f"[calibrate-motors] driving forward at speed={speed:.2f} for {duration:.1f}s...")
    controller.forward(speed)
    time.sleep(duration)
    controller.stop()


def _write_env_trim(left_trim: float, right_trim: float) -> None:
    """Set NANO_MOTOR_LEFT_TRIM/RIGHT_TRIM in _ENV_FILE, preserving any other lines."""
    lines = []
    if _ENV_FILE.exists():
        original = _ENV_FILE.read_text(encoding="utf-8")
        backup = _ENV_FILE.with_name(
            f"{_ENV_FILE.name}.bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        backup.write_text(original, encoding="utf-8")
        print(f"[calibrate-motors] Backed up existing env file to {backup}")
        lines = original.splitlines()

    updates = {
        "NANO_MOTOR_LEFT_TRIM": f"{left_trim:.4f}",
        "NANO_MOTOR_RIGHT_TRIM": f"{right_trim:.4f}",
    }
    seen = set()
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip() if "=" in line else None
        if key in updates:
            lines[i] = f"{key}={updates[key]}"
            seen.add(key)
    for key, value in updates.items():
        if key not in seen:
            lines.append(f"{key}={value}")

    _ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[calibrate-motors] Wrote {_ENV_FILE}")


def calibrate(speed: float, duration: float, step: float) -> None:
    import termios
    import tty

    from lib.motor import MotorController

    controller = MotorController(left_trim=1.0, right_trim=1.0)
    controller.open()

    bias = 0.0
    left_trim, right_trim = _trims_from_bias(bias)
    controller.set_trim(left_trim, right_trim)

    print(
        "\n[calibrate-motors] f/space = forward test  |  "
        "left/right arrow = it drifted that way  |  +/- = step size  |  "
        "s = save & quit  |  q = quit without saving\n"
        f"[calibrate-motors] step size: {step:.3f}\n"
        f"[calibrate-motors] trim: left={left_trim:.3f}  right={right_trim:.3f}"
    )

    fd = sys.stdin.fileno()
    old_attr = termios.tcgetattr(fd)
    key = "q"
    try:
        tty.setcbreak(fd)
        while True:
            key = _read_key(fd)
            if key in ("f", " "):
                _run_forward(controller, speed, duration)
            elif key == "right":
                bias = _clamp_bias(bias + step)
            elif key == "left":
                bias = _clamp_bias(bias - step)
            elif key == "+":
                step = min(0.2, step * 1.5)
                print(f"[calibrate-motors] step size: {step:.3f}")
                continue
            elif key == "-":
                step = max(0.005, step / 1.5)
                print(f"[calibrate-motors] step size: {step:.3f}")
                continue
            elif key in ("s", "q", "\x03"):
                break
            else:
                continue

            if key in ("left", "right"):
                left_trim, right_trim = _trims_from_bias(bias)
                controller.set_trim(left_trim, right_trim)
                print(f"[calibrate-motors] trim: left={left_trim:.3f}  right={right_trim:.3f}")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
        controller.stop()
        controller.close()

    if key == "s":
        print(f"[calibrate-motors] Final trim: left={left_trim:.3f}  right={right_trim:.3f}")
        _write_env_trim(left_trim, right_trim)
    else:
        print("[calibrate-motors] Quit without saving.")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate per-wheel motor trim so the JetBot drives straight",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--speed", type=float, default=0.3, help="Forward test speed [0.0, 1.0]")
    parser.add_argument(
        "--duration", type=float, default=1.5, help="Forward test duration in seconds"
    )
    parser.add_argument(
        "--step", type=float, default=0.02, help="Initial trim nudge step per arrow-key press"
    )
    args = parser.parse_args()
    calibrate(speed=args.speed, duration=args.duration, step=args.step)


if __name__ == "__main__":
    main()
