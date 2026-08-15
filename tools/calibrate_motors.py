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
  f / space / up-arrow -> hold to drive forward at --speed; release to stop
  right arrow -> "it drifted RIGHT last run" (nudges trim to compensate)
  left arrow  -> "it drifted LEFT last run"  (nudges trim to compensate)
  +           -> increase the nudge step size
  -           -> decrease the nudge step size
  s           -> save the current trim to ~/.nano-explorer.env and exit
  q / Ctrl-C  -> quit without saving

Procedure
---------
  1. Put the robot on the floor with room to roll forward a meter or two.
  2. Run this script (add --speed to match how you actually drive — the
     drift can vary with speed).
  3. Hold f (space or the up arrow also work) to drive forward and watch
     which way the nose drifts; release the key to stop.
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

# Terminals have no real key-up event, so "release" is inferred the same way
# lib.motion.teleoperation does: a held key auto-repeats every keystroke while
# down, so if _STOP_DELAY passes with no drive keystroke, the key must be up.
_STOP_DELAY = 0.15
_POLL_TIMEOUT = 0.05
# How long to hold onto an incomplete arrow-key escape sequence (only \x1b
# has arrived so far) before discarding it as stale.
_PARTIAL_SEQ_TIMEOUT = 0.1
_ARROW_SEQS = {b"\x1b[A": "drive", b"\x1b[C": "right", b"\x1b[D": "left"}
_SINGLE_BYTE_ACTIONS = {
    b"f": "drive",
    b" ": "drive",
    b"s": "save",
    b"q": "quit",
    b"\x03": "quit",
    b"+": "step_up",
    b"-": "step_down",
}


def _clamp_bias(bias: float) -> float:
    return max(-_MAX_BIAS, min(_MAX_BIAS, bias))


def _trims_from_bias(bias: float):
    """bias > 0 corrects a rightward drift (reduces left); bias < 0 corrects leftward."""
    left = 1.0 - max(bias, 0.0)
    right = 1.0 - max(-bias, 0.0)
    return left, right


class _KeyReader:
    """
    Assembles raw terminal bytes into one resolved action per non-blocking poll:
    "drive" (f, space, or up arrow — hold to drive forward), "left"/"right"
    (drift feedback), "step_up"/"step_down", "save", "quit", or "" for nothing
    yet.

    Arrow keys arrive as a 3-byte escape sequence (\\x1b[A etc.) that a single
    os.read() isn't guaranteed to return in one piece — under load it can wake
    up after only 1 or 2 bytes — so partial sequences are buffered across
    calls instead of being compared directly and silently dropped, the same
    fix already applied to lib.motion.teleoperation's _ArrowKeyReader.
    """

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._pending = b""
        self._pending_since: float = 0.0

    def read_action(self) -> str:
        import select

        ready, _, _ = select.select([self._fd], [], [], _POLL_TIMEOUT)
        if not ready:
            if self._pending and time.time() - self._pending_since > _PARTIAL_SEQ_TIMEOUT:
                self._pending = b""
            return ""

        self._pending += os.read(self._fd, 3 - len(self._pending))
        if not self._pending_since:
            self._pending_since = time.time()

        if self._pending[:1] != b"\x1b":
            chunk, self._pending, self._pending_since = self._pending, b"", 0.0
            return _SINGLE_BYTE_ACTIONS.get(chunk, "")

        if len(self._pending) < 3:
            return ""  # escape sequence still incomplete — wait for the rest

        chunk, self._pending, self._pending_since = self._pending, b"", 0.0
        return _ARROW_SEQS.get(chunk, "")


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


# pylint: disable=too-many-statements
def calibrate(speed: float, step: float) -> None:
    import termios
    import threading
    import tty

    from lib.motor import MotorController

    controller = MotorController(left_trim=1.0, right_trim=1.0)
    controller.open()

    bias = 0.0
    left_trim, right_trim = _trims_from_bias(bias)
    controller.set_trim(left_trim, right_trim)

    print(
        "\n[calibrate-motors] hold f / space / up-arrow = drive forward, release = stop  |  "
        "left/right arrow = it drifted that way  |  +/- = step size  |  "
        "s = save & quit  |  q = quit without saving\n"
        f"[calibrate-motors] step size: {step:.3f}\n"
        f"[calibrate-motors] trim: left={left_trim:.3f}  right={right_trim:.3f}"
    )

    fd = sys.stdin.fileno()
    old_attr = termios.tcgetattr(fd)
    reader = _KeyReader(fd)
    driving = False
    stop_timer = None
    final_action = "quit"

    def _stop_if_released() -> None:
        nonlocal driving
        controller.stop()
        driving = False
        print("[calibrate-motors] stopped.")

    def _schedule_stop() -> None:
        nonlocal stop_timer
        if stop_timer is not None:
            stop_timer.cancel()
        stop_timer = threading.Timer(_STOP_DELAY, _stop_if_released)
        stop_timer.daemon = True
        stop_timer.start()

    try:
        tty.setcbreak(fd)
        while True:
            action = reader.read_action()
            if action == "drive":
                if not driving:
                    driving = True
                    print(f"[calibrate-motors] driving forward at speed={speed:.2f}...")
                    controller.forward(speed)
                _schedule_stop()
                continue
            if action == "right":
                bias = _clamp_bias(bias + step)
            elif action == "left":
                bias = _clamp_bias(bias - step)
            elif action == "step_up":
                step = min(0.2, step * 1.5)
                print(f"[calibrate-motors] step size: {step:.3f}")
                continue
            elif action == "step_down":
                step = max(0.005, step / 1.5)
                print(f"[calibrate-motors] step size: {step:.3f}")
                continue
            elif action in ("save", "quit"):
                final_action = action
                break
            else:
                continue

            left_trim, right_trim = _trims_from_bias(bias)
            controller.set_trim(left_trim, right_trim)
            print(f"[calibrate-motors] trim: left={left_trim:.3f}  right={right_trim:.3f}")
    finally:
        if stop_timer is not None:
            stop_timer.cancel()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
        controller.stop()
        controller.close()

    if final_action == "save":
        print(f"[calibrate-motors] Final trim: left={left_trim:.3f}  right={right_trim:.3f}")
        _write_env_trim(left_trim, right_trim)
    else:
        print("[calibrate-motors] Quit without saving.")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate per-wheel motor trim so the JetBot drives straight",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--speed", type=float, default=0.3, help="Forward drive speed [0.0, 1.0]")
    parser.add_argument(
        "--step", type=float, default=0.02, help="Initial trim nudge step per arrow-key press"
    )
    args = parser.parse_args()
    calibrate(speed=args.speed, step=args.step)


if __name__ == "__main__":
    main()
