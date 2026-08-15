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

from loguru import logger

try:
    from jetbot import Robot

    class InvertedRobot(Robot):
        """
        A Robot subclass where motor directions are inverted to align with the camera orientation.

        On this physical configuration, the camera faces the direction of travel, but the
        robot's chassis is mounted such that the motors drive it in the opposite direction
        relative to the JetBot coordinate system. To correct for this, all motor commands
        are inverted so that callers can use intuitive directional semantics:

            - forward()  -> robot moves toward what the camera sees
            - backward() -> robot moves away from what the camera sees

        Internally, forward() delegates to the parent's backward() and vice versa.
        This inversion is intentional and should not be "fixed" — it reflects the
        physical mounting of the robot, not a bug in the logic.

        Usage:
            robot = InvertedRobot()
            robot.forward(0.3)   # moves in the camera-facing direction
            robot.backward(0.3)  # moves away from the camera
        """

        def forward(self, speed=0.4):
            super().backward(speed)

        def backward(self, speed=0.4):
            super().forward(speed)

        def set_motors(self, left_speed, right_speed):
            # Also invert raw motor commands
            super().set_motors(-left_speed, -right_speed)

    _HW_AVAILABLE = True
except ImportError:
    _HW_AVAILABLE = False
    logger.warning(
        "jetbot library not found — running in DRY-RUN mode. "
        "Install with: pip3 install jetbot  "
        "(or follow the Waveshare JetBot setup guide)"
    )


from typing import Tuple

import yaml
from pydantic import BaseModel, Field  # pylint: disable=no-name-in-module

from lib.settings import NanoSettings


def _load_trim() -> Tuple[float, float]:
    """
    Read (left_trim, right_trim) from NanoSettings().motor_trim_config_path,
    defaulting to (1.0, 1.0) if the file doesn't exist yet. Deliberately a
    plain YAML read, not a pydantic env-settings field — a real (exported)
    shell env var silently overrides a same-named .env file entry, which
    made a stale export indistinguishable from a freshly-calibrated value.
    See tools/calibrate_motors.py and config/motors/trim.yaml.
    """
    path = NanoSettings().motor_trim_config_path
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

    Args:
        left_trim:  Per-wheel power trim [0.0, 1.0], applied via jetbot's
                    left_motor_alpha. Defaults to the value in
                    config/motors/trim.yaml (1.0 = no correction) if not
                    given. See tools/calibrate_motors.py.
        right_trim: Same, for the right wheel.
    """

    def __init__(self, left_trim: float = None, right_trim: float = None):
        self._robot = None
        self._dry_run = not _HW_AVAILABLE
        default_left, default_right = _load_trim()
        self._left_trim = default_left if left_trim is None else left_trim
        self._right_trim = default_right if right_trim is None else right_trim

    def open(self) -> None:
        """Initialise the JetBot Robot instance."""
        if self._dry_run:
            logger.warning("DRY-RUN: MotorController.open() skipped.")
            return
        self._robot = InvertedRobot(
            left_motor_alpha=self._left_trim, right_motor_alpha=self._right_trim
        )
        logger.success(
            f"MotorController ready (jetbot.Robot)  "
            f"trim: left={self._left_trim:.3f}  right={self._right_trim:.3f}"
        )

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

    def set_speeds(self, left: float, right: float) -> None:
        """
        Set individual wheel speeds.

        Args:
            left:  Left wheel speed  [0.0, 1.0]  (negative = reverse)
            right: Right wheel speed [0.0, 1.0]  (negative = reverse)
        """
        speeds = WheelSpeeds(left=left, right=right)
        if self._dry_run:
            logger.debug(f"DRY-RUN set_speeds: L={speeds.left:.2f}  R={speeds.right:.2f}")
            return
        self._robot.left_motor.value = speeds.left
        self._robot.right_motor.value = speeds.right

    def stop(self) -> None:
        if self._dry_run:
            logger.debug("DRY-RUN: stop()")
            return
        if self._robot:
            self._robot.stop()

    def forward(self, speed: float = 0.3) -> None:
        if self._dry_run:
            logger.debug(f"DRY-RUN: forward({speed:.2f})")
            return
        self._robot.forward(speed=speed)

    def backward(self, speed: float = 0.3) -> None:
        if self._dry_run:
            logger.debug(f"DRY-RUN: backward({speed:.2f})")
            return
        self._robot.backward(speed=speed)

    def turn_left(self, speed: float = 0.3) -> None:
        if self._dry_run:
            logger.debug(f"DRY-RUN: turn_left({speed:.2f})")
            return
        self._robot.left(speed=speed)

    def turn_right(self, speed: float = 0.3) -> None:
        if self._dry_run:
            logger.debug(f"DRY-RUN: turn_right({speed:.2f})")
            return
        self._robot.right(speed=speed)

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
