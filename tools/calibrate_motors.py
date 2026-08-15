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

Method: measure once, compute directly
---------------------------------------
An earlier version of this tool worked by live trial-and-error: hold a key
to drive, watch which way the robot drifted, nudge a trim value, repeat.
In practice that's hard to do well — it demands watching the robot on the
floor and reacting on a keyboard in the same instant, and each correction
is a subjective guess rather than a measurement, so it takes many cycles
to converge.

This version instead drives a fixed, repeatable test (same speed, same
duration, every time) and asks you to physically measure the result with a
tape measure: how far forward the robot travelled, and how far off to one
side it ended up. From those two numbers plus the robot's wheelbase, the
exact wheel-speed ratio causing the drift is computed directly from
differential-drive kinematics — no guessing, and it converges in 1-2 test
drives instead of many.

The math (for the curious)
---------------------------
For a differential-drive robot with wheel speeds v_l, v_r and wheelbase W,
driving for a short arc produces a robot-frame displacement of
approximately:
    forward distance   D ~= v*T
    lateral offset      L ~= D^2 / (2*R)      (R = signed turning radius)
where R = W*(v_r+v_l) / (2*(v_r-v_l)). Solving for the wheel-speed ratio
rho = v_r/v_l that produced a measured (D, L) — with L > 0 meaning the
robot drifted right — gives:
    rho = (D^2 - W*L) / (D^2 + W*L)
Trim is then set to bring the stronger wheel down until both effective
speeds match (rho == 1): whichever of left/right trim needs to shrink is
computed from rho, the other stays at 1.0. This is a small-angle
approximation, valid as long as the drift stays much smaller than the
forward distance — exactly the "arcing slightly instead of driving
straight" case this tool exists to fix. If a test drive drifts too far
for that to hold, shorten --duration and try again.

Wheelbase
---------
--wheelbase is the distance between the two wheels' ground-contact points
(their centers), in centimetres. Measure it once with a ruler or calipers
— it doesn't change unless you change the chassis. There's no default
baked in here, since a wrong wheelbase silently skews every calibration;
you have to pass it explicitly.

Procedure
---------
  1. Put the robot on the floor with ~2m of clear straight-line space ahead.
  2. Mark a straight reference line on the floor (tape, a rug seam, a row
     of tiles) and place the robot's center at one end, facing along it.
  3. Run: python tools/calibrate_motors.py --wheelbase <cm you measured>
  4. Press ENTER to run a test drive (forward at --speed for --duration).
  5. Once it stops, measure with a tape measure and enter when prompted:
       - forward distance travelled along the reference line, in cm
       - how far off the line it ended up, and to which side
  6. The tool computes and applies the corrected trim from those numbers.
  7. Press ENTER again to run a verification drive with the new trim — it
     should track much closer to the line now. Repeat steps 5-7 once more
     if you want to refine it further (each round corrects the residual
     drift from whatever trim was active during that drive, so it's safe
     to keep going from wherever you left off).
  8. Press 's' + ENTER to save. This writes NANO_MOTOR_LEFT_TRIM /
     NANO_MOTOR_RIGHT_TRIM to ~/.nano-explorer.env (backing up any existing
     file first), so every command that builds a MotorController picks up
     the correction automatically. Press 'q' + ENTER at any point to quit
     without saving.

Note: this always starts from a clean 1.0/1.0 (no trim), not whatever is
currently saved — if you're refining an existing calibration rather than
starting over, the current values are printed at startup so a fresh
measurement round builds on top of them correctly.
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

_ENV_FILE = Path.home() / ".nano-explorer.env"

# A real motor pair rarely differs by more than 2x — this catches measurement
# typos (wrong units, swapped digits) rather than genuine motor mismatch.
_MIN_RATIO = 0.5
_MAX_RATIO = 1.0 / _MIN_RATIO

# Below this, a measured lateral offset is treated as "straight" noise
# rather than a real drift worth correcting.
_STRAIGHT_TOLERANCE_CM = 0.1


def _ratio_from_measurement(forward_cm: float, lateral_cm: float, wheelbase_cm: float) -> float:
    """
    v_r/v_l implied by a measured forward distance and signed lateral offset
    (positive = drifted right), from the small-angle diff-drive kinematics
    derived in the module docstring.
    """
    denom = forward_cm**2 + wheelbase_cm * lateral_cm
    if denom <= 0:
        raise ValueError(
            "Measured drift is too large relative to the forward distance for "
            "the small-angle approximation this tool relies on — re-run the "
            "test with a shorter --duration (less time to arc) and try again."
        )
    return (forward_cm**2 - wheelbase_cm * lateral_cm) / denom


def _trims_from_ratio(ratio: float) -> Tuple[float, float]:
    """Normalize a v_r/v_l ratio into (left_trim, right_trim), weaker wheel at 1.0."""
    clamped = max(_MIN_RATIO, min(_MAX_RATIO, ratio))
    if clamped != ratio:
        print(
            f"[calibrate-motors] computed ratio {ratio:.3f} is outside the expected "
            f"[{_MIN_RATIO}, {_MAX_RATIO}] range for a real motor mismatch — clamping. "
            "Double-check your measurements."
        )
    if clamped >= 1.0:
        return 1.0, 1.0 / clamped
    return clamped, 1.0


def _prompt_float(prompt: str) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            return float(raw)
        except ValueError:
            print("  please enter a number.")


def _prompt_lateral_cm() -> float:
    while True:
        side = input("  Which side did it drift toward? [l]eft / [r]ight / [s]traight: ")
        side = side.strip().lower()
        if side in ("s", "straight", ""):
            return 0.0
        if side in ("l", "left", "r", "right"):
            magnitude = _prompt_float("  How far off the line, in cm: ")
            return magnitude if side.startswith("r") else -magnitude
        print("  please enter l, r, or s.")


def _run_test_drive(controller, speed: float, duration: float) -> None:
    print(f"[calibrate-motors] driving forward at speed={speed:.2f} for {duration:.1f}s...")
    controller.forward(speed)
    time.sleep(duration)
    controller.stop()
    print("[calibrate-motors] stopped — go measure.")


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


def calibrate(speed: float, duration: float, wheelbase: float) -> None:
    from lib.motor import MotorController

    controller = MotorController(left_trim=1.0, right_trim=1.0)
    controller.open()

    left_trim, right_trim = 1.0, 1.0
    controller.set_trim(left_trim, right_trim)

    print(
        f"\n[calibrate-motors] wheelbase={wheelbase:.2f}cm  speed={speed:.2f}  "
        f"test duration={duration:.1f}s\n"
        f"[calibrate-motors] starting trim: left={left_trim:.3f}  right={right_trim:.3f}\n"
        "[calibrate-motors] Mark a straight reference line on the floor and place the "
        "robot's center at the start, facing along it."
    )

    save = False
    try:
        while True:
            choice = (
                input(
                    "\nPress ENTER to run a test drive, 's' + ENTER to save & quit, "
                    "or 'q' + ENTER to quit without saving: "
                )
                .strip()
                .lower()
            )
            if choice in ("q", "quit"):
                break
            if choice in ("s", "save"):
                save = True
                break

            _run_test_drive(controller, speed, duration)

            forward_cm = _prompt_float("  Forward distance travelled along the line (cm): ")
            if forward_cm <= 0:
                print("  that doesn't look right (must be > 0) — skipping this measurement.")
                continue

            lateral_cm = _prompt_lateral_cm()
            if abs(lateral_cm) < _STRAIGHT_TOLERANCE_CM:
                print("[calibrate-motors] Straight within measurement noise — nothing to change.")
                continue

            try:
                ratio_eff = _ratio_from_measurement(forward_cm, lateral_cm, wheelbase)
            except ValueError as exc:
                print(f"[calibrate-motors] {exc}")
                continue

            # ratio_eff is v_r/v_l under the trim that was active during this
            # drive; rescale by that trim to recover the true underlying
            # motor ratio so refinement rounds compose correctly.
            ratio_true = ratio_eff * (left_trim / right_trim)
            left_trim, right_trim = _trims_from_ratio(ratio_true)
            controller.set_trim(left_trim, right_trim)
            print(f"[calibrate-motors] trim updated: left={left_trim:.3f}  right={right_trim:.3f}")
    finally:
        controller.stop()
        controller.close()

    if save:
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
        "--duration", type=float, default=2.0, help="Forward test drive duration in seconds"
    )
    parser.add_argument(
        "--wheelbase",
        type=float,
        required=True,
        help=(
            "Distance between the two wheels' ground-contact points, in cm "
            "(measure with a ruler/calipers — see the module docstring)"
        ),
    )
    args = parser.parse_args()
    calibrate(speed=args.speed, duration=args.duration, wheelbase=args.wheelbase)


if __name__ == "__main__":
    main()
