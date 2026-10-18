"""
Motor controller abstraction using the official JetBot Robot class.

    from jetbot import Robot
    robot = Robot()
    robot.forward(speed=0.3)
    robot.stop()

The Robot class handles all PCA9685 / I2C wiring internally.
This thin wrapper keeps the rest of the codebase (teleop, collision,
line follower, etc.) decoupled from the JetBot API directly.

Speed values are floats in [0.0, 1.0].
"""

import functools
import threading
from typing import Tuple

import yaml
from loguru import logger
from pydantic import BaseModel, Field  # pylint: disable=no-name-in-module

from lib.settings import NanoSettings, ensure_user_config

# One lock for every motor command in the process. A single command writes both
# motors, and each motor update in jetbot is several separate I2C writes (speed,
# direction, two PWM pins), so two threads sending commands at once (teleop,
# the key-release stop timer, auto-recovery) could interleave and leave one
# wheel on the old command: the robot turns or circles when told to go straight.
# Module-level rather than per instance because jetbot's Robot is a singleton —
# every MotorController drives the same board. Re-entrant because close() calls
# stop() and steer() calls set_speeds().
MOTOR_LOCK = threading.RLock()


def _locked(method):
    """Run a MotorController method under MOTOR_LOCK, so its motor writes are never interleaved."""

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        with MOTOR_LOCK:
            return method(*args, **kwargs)

    return wrapper


try:
    from jetbot import Robot

    _HW_AVAILABLE = True
except ImportError:
    _HW_AVAILABLE = False
    logger.warning(
        "jetbot library not found — running in DRY-RUN mode. "
        "Install with: pip3 install jetbot  "
        "(or follow the Waveshare JetBot setup guide)"
    )


def _load_trim() -> Tuple[float, float]:
    """
    Read (left_trim, right_trim) from NanoSettings().motor_trim_config_path
    (~/.nano-explorer/config/motors/trim.yaml — seeded from the bundled
    default the first time this runs), defaulting to (1.0, 1.0) if even the
    bundled default is somehow missing. Deliberately a plain YAML read, not
    a pydantic env-settings field — a real (exported) shell env var silently
    overrides a same-named .env file entry, which made a stale export
    indistinguishable from a freshly-calibrated value. See
    tools/calibrate_motors.py and config/motors/trim.yaml.
    """
    path = ensure_user_config(NanoSettings().motor_trim_config_path)
    if not path.exists():
        return 1.0, 1.0
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return float(cfg.get("left_trim", 1.0)), float(cfg.get("right_trim", 1.0))


class WheelSpeeds(BaseModel):
    left: float = Field(..., ge=-1.0, le=1.0)
    right: float = Field(..., ge=-1.0, le=1.0)


class MotorController:
    """
    Thin wrapper around jetbot.Robot.

    All speed values are floats in [0.0, 1.0].
    In DRY-RUN mode (jetbot not installed) every call is logged but
    no hardware is touched.

    Left/right here always mean the robot's physical wheels as seen driving
    camera-forward, and match jetbot's own left_motor/right_motor: the motors
    are wired in the stock JetBot orientation, so jetbot "forward" is toward
    what the camera sees.

    Args:
        left_trim:  Per-wheel power trim [0.0, 1.0] for the physical left
                    wheel (jetbot's left_motor). Defaults to the value in
                    config/motors/trim.yaml (1.0 = no correction) if not
                    given. See tools/calibrate_motors.py.
        right_trim: Same, for the physical right wheel (jetbot's right_motor).
    """

    def __init__(self, left_trim: float = None, right_trim: float = None):
        self._robot = None
        self._dry_run = not _HW_AVAILABLE
        default_left, default_right = _load_trim()
        self._left_trim = default_left if left_trim is None else left_trim
        self._right_trim = default_right if right_trim is None else right_trim

    @_locked
    def open(self) -> None:
        """Initialise the JetBot Robot instance."""
        if self._dry_run:
            logger.warning("DRY-RUN: MotorController.open() skipped.")
            return
        self._robot = Robot(left_motor_alpha=self._left_trim, right_motor_alpha=self._right_trim)
        logger.success(
            f"MotorController ready (jetbot.Robot)  "
            f"trim: left={self._left_trim:.3f}  right={self._right_trim:.3f}"
        )

    @_locked
    def set_trim(self, left_trim: float, right_trim: float) -> None:
        """Update per-wheel power trim live — takes effect on the next motor command."""
        self._left_trim = left_trim
        self._right_trim = right_trim
        if self._dry_run:
            logger.debug(f"DRY-RUN set_trim: left={left_trim:.3f}  right={right_trim:.3f}")
            return
        if self._robot is not None:
            self._robot.left_motor.alpha = left_trim
            self._robot.right_motor.alpha = right_trim

    @_locked
    def close(self) -> None:
        """Stop motors and release the Robot instance."""
        self.stop()
        self._robot = None
        logger.info("MotorController closed.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    @_locked
    def set_speeds(self, left: float, right: float) -> None:
        """
        Set individual wheel speeds.

        Args:
            left:  Physical left wheel speed  [-1.0, 1.0] (positive = toward the camera view)
            right: Physical right wheel speed [-1.0, 1.0] (positive = toward the camera view)
        """
        speeds = WheelSpeeds(left=left, right=right)
        if self._dry_run:
            logger.debug(f"DRY-RUN set_speeds: L={speeds.left:.2f}  R={speeds.right:.2f}")
            return
        self._robot.set_motors(speeds.left, speeds.right)

    @_locked
    def stop(self) -> None:
        if self._dry_run:
            logger.debug("DRY-RUN: stop()")
            return
        if self._robot:
            self._robot.stop()

    @_locked
    def forward(self, speed: float = 0.3) -> None:
        if self._dry_run:
            logger.debug(f"DRY-RUN: forward({speed:.2f})")
            return
        self._robot.forward(speed=speed)

    @_locked
    def backward(self, speed: float = 0.3) -> None:
        if self._dry_run:
            logger.debug(f"DRY-RUN: backward({speed:.2f})")
            return
        self._robot.backward(speed=speed)

    @_locked
    def turn_left(self, speed: float = 0.3) -> None:
        if self._dry_run:
            logger.debug(f"DRY-RUN: turn_left({speed:.2f})")
            return
        self._robot.left(speed=speed)

    @_locked
    def turn_right(self, speed: float = 0.3) -> None:
        if self._dry_run:
            logger.debug(f"DRY-RUN: turn_right({speed:.2f})")
            return
        self._robot.right(speed=speed)

    @_locked
    def steer(self, speed: float, steering: float) -> None:
        """
        Differential steering helper.

        Args:
            speed:    Base forward speed [0.0, 1.0]
            steering: Steering value [-1.0, 1.0]
                      -1.0 = hard left, +1.0 = hard right
        """
        left = speed + steering
        right = speed - steering
        max_val = max(abs(left), abs(right), 1.0)
        self.set_speeds(left / max_val, right / max_val)
