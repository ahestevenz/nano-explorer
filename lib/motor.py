"""
lib/motor.py

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

    _HW_AVAILABLE = True
except ImportError:
    _HW_AVAILABLE = False
    logger.warning(
        "jetbot library not found — running in DRY-RUN mode. "
        "Install with: pip3 install jetbot  "
        "(or follow the Waveshare JetBot setup guide)"
    )


class MotorController:
    """
    Thin wrapper around jetbot.Robot.

    All speed values are floats in [0.0, 1.0].
    In DRY-RUN mode (jetbot not installed) every call is logged but
    no hardware is touched.
    """

    def __init__(self):
        self._robot = None
        self._dry_run = not _HW_AVAILABLE

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Initialise the JetBot Robot instance."""
        if self._dry_run:
            logger.warning("DRY-RUN: MotorController.open() skipped.")
            return
        self._robot = Robot()
        logger.success("MotorController ready (jetbot.Robot)")

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

    # ── Primitive control ─────────────────────────────────────────────────────

    def set_speeds(self, left: float, right: float) -> None:
        """
        Set individual wheel speeds.

        Args:
            left:  Left wheel speed  [0.0, 1.0]  (negative = reverse)
            right: Right wheel speed [0.0, 1.0]  (negative = reverse)
        """
        left = max(-1.0, min(1.0, left))
        right = max(-1.0, min(1.0, right))
        if self._dry_run:
            logger.debug("DRY-RUN set_speeds: L={:.2f}  R={:.2f}".format(left, right))
            return
        self._robot.left_motor.value = left
        self._robot.right_motor.value = right

    def stop(self) -> None:
        if self._dry_run:
            logger.debug("DRY-RUN: stop()")
            return
        if self._robot:
            self._robot.stop()

    # ── Convenience helpers ───────────────────────────────────────────────────

    def forward(self, speed: float = 0.3) -> None:
        if self._dry_run:
            logger.debug("DRY-RUN: forward({:.2f})".format(speed))
            return
        self._robot.forward(speed=speed)

    def backward(self, speed: float = 0.3) -> None:
        if self._dry_run:
            logger.debug("DRY-RUN: backward({:.2f})".format(speed))
            return
        self._robot.backward(speed=speed)

    def turn_left(self, speed: float = 0.3) -> None:
        if self._dry_run:
            logger.debug("DRY-RUN: turn_left({:.2f})".format(speed))
            return
        self._robot.left(speed=speed)

    def turn_right(self, speed: float = 0.3) -> None:
        if self._dry_run:
            logger.debug("DRY-RUN: turn_right({:.2f})".format(speed))
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
