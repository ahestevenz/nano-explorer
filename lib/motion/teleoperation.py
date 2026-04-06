"""
Keyboard teleoperation for the JetBot.

Three input modes (auto-selected or forced with --mode):

  arrows  (default)
          Raw terminal arrow-key reading via tty/termios — works over plain
          SSH with no X server and no root.  Press and hold an arrow key to
          move; releasing the key stops the robot after a short timeout.

  pynput  Classic pynput Listener — needs DISPLAY set (local/SSH -X) or
          /dev/uinput writable (sudo modprobe uinput && sudo chmod a+rw /dev/uinput).

  stdin   Type a letter + ENTER.  Works everywhere, no real-time feel.

Auto-selection order:  arrows -> pynput (if DISPLAY set) -> stdin

Controls (arrows / pynput modes):
    UP    -> forward
    DOWN  -> backward
    LEFT  -> turn left
    RIGHT -> turn right
    q/ESC -> quit

Controls (stdin mode):
    f / forward   -> forward
    b / backward  -> backward
    l / left      -> turn left
    r / right     -> turn right
    s / stop      -> stop
    q / quit      -> quit
"""

import os
import sys
import threading
import time

from loguru import logger

from lib.motor import MotorController

_ARROW_MAP = {
    b"\x1b[A": "forward",  # Up
    b"\x1b[B": "backward",  # Down
    b"\x1b[C": "right",  # Right
    b"\x1b[D": "left",  # Left
}

# How long (seconds) after the last keypress before we treat it as "released"
_KEY_TIMEOUT: float = 0.15


def _import_keyboard():
    """
    Try every available pynput backend in order.
    Returns kb module on success, raises RuntimeError if all fail.
    """
    errors = {}
    backends_to_try = []

    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        backends_to_try.append("xorg")
    backends_to_try.append("uinput")
    if "xorg" not in backends_to_try:
        backends_to_try.append("xorg")

    for backend in backends_to_try:
        try:
            os.environ["PYNPUT_BACKEND"] = backend
            import importlib

            import pynput
            import pynput.keyboard

            importlib.reload(pynput.keyboard)
            importlib.reload(pynput)
            from pynput import keyboard as kb

            _ = kb.Key.up  # sanity check
            logger.debug("pynput backend '{}' loaded OK.".format(backend))
            return kb
        except Exception as exc:
            errors[backend] = str(exc)
            for mod in list(sys.modules.keys()):
                if "pynput" in mod:
                    del sys.modules[mod]

    diag = "\n".join("  {}: {}".format(b, e) for b, e in errors.items())
    raise RuntimeError(
        "pynput could not find a working backend.\n\nBackends tried:\n{}\n\n"
        "Try: sudo modprobe uinput && sudo chmod a+rw /dev/uinput".format(diag)
    )


# stdin fallback

_STDIN_HELP = """
stdin teleop mode — type commands and press ENTER:
  f  or  forward   -> forward
  b  or  backward  -> backward
  l  or  left      -> turn left
  r  or  right     -> turn right
  s  or  stop      -> stop
  q  or  quit      -> quit
"""

_STDIN_MAP = {
    "f": "forward",
    "forward": "forward",
    "b": "backward",
    "backward": "backward",
    "l": "left",
    "left": "left",
    "r": "right",
    "right": "right",
    "s": "stop",
    "stop": "stop",
    "q": "quit",
    "quit": "quit",
}


# Main controller
class TeleopController:
    """
    Keyboard-driven teleoperation controller.

    Args:
        speed:     Linear motor speed [0.0, 1.0]
        turn_gain: Differential turn gain [0.0, 1.0]
        mode:      "arrows" | "pynput" | "stdin" | "auto"
    """

    def __init__(
        self,
        speed: float = 0.3,
        turn_gain: float = 0.5,
        mode: str = "auto",
    ):
        self.speed = speed
        self.turn_gain = turn_gain
        self.mode = mode
        self._motors = MotorController()
        self._running = False

    def run(self) -> None:
        if self.mode == "auto":
            self._run_auto()
        elif self.mode == "arrows":
            self._run_arrows()
        elif self.mode == "pynput":
            self._run_pynput()
        elif self.mode == "stdin":
            self._run_stdin()
        else:
            raise ValueError("Unknown teleop mode: {}".format(self.mode))

    def _run_auto(self) -> None:
        """Try modes in order: arrows -> pynput -> stdin."""
        # arrows works everywhere a TTY is available
        if sys.stdin.isatty():
            self._run_arrows()
            return
        # Try pynput
        try:
            _import_keyboard()
            self._run_pynput()
            return
        except RuntimeError as exc:
            logger.warning("pynput unavailable ({}), using stdin.".format(exc))
        self._run_stdin()

    # Mode: raw arrow keys
    def _run_arrows(self) -> None:
        """
        Read raw terminal bytes.  Arrow keys produce 3-byte escape sequences.
        A background timer stops the robot if no key arrives within _KEY_TIMEOUT.
        """
        import select
        import termios
        import tty

        _HELP = (
            "\n[teleop] Arrow keys to drive  |  q = quit\n"
            "         Hold key -> move  |  Release -> stop\n"
        )
        print(_HELP)

        self._motors.open()
        self._running = True
        stop_timer = None  # threading.Timer
        _READ_3_BYTES: int = 3

        def _schedule_stop():
            """Start/restart the release-detection timer."""
            nonlocal stop_timer
            if stop_timer is not None:
                stop_timer.cancel()
            stop_timer = threading.Timer(_KEY_TIMEOUT, self._motors.stop)
            stop_timer.daemon = True
            stop_timer.start()

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)

            while self._running:
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not ready:
                    continue

                chunk = os.read(fd, _READ_3_BYTES)

                if chunk in (b"q", b"Q", b"\x03"):
                    break

                action = _ARROW_MAP.get(chunk)
                if action is None:
                    continue

                self._apply_action(action)
                _schedule_stop()

        except Exception as exc:
            logger.error("Arrow teleop error: {}".format(exc))
        finally:
            if stop_timer is not None:
                stop_timer.cancel()
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            self._motors.stop()
            self._motors.close()
            print("\n[teleop] stopped.")

    # Mode: pynput
    def _run_pynput(self) -> None:
        kb = _import_keyboard()
        keymap = {
            kb.Key.up: "forward",
            kb.Key.down: "backward",
            kb.Key.left: "left",
            kb.Key.right: "right",
            kb.Key.space: "stop",
        }

        current_action = ["stop"]
        running = [True]

        def on_press(key):
            action = keymap.get(key)
            if action:
                current_action[0] = action
                return
            try:
                if key.char in ("q", "Q"):
                    running[0] = False
            except AttributeError:
                if key == kb.Key.esc:
                    running[0] = False

        def on_release(key):
            if key in keymap:
                current_action[0] = "stop"

        logger.info(
            "Teleop (pynput) — arrow keys to drive, SPACE to stop, Q/ESC to quit"
        )
        self._motors.open()

        with kb.Listener(on_press=on_press, on_release=on_release) as listener:
            try:
                while running[0]:
                    self._apply_action(current_action[0])
                    time.sleep(0.05)
            except KeyboardInterrupt:
                pass
            finally:
                listener.stop()
                self._motors.stop()
                self._motors.close()
                logger.info("Teleop stopped.")

    # Mode: stdin
    def _run_stdin(self) -> None:
        print(_STDIN_HELP)
        self._motors.open()
        try:
            while True:
                try:
                    raw = input("teleop> ").strip().lower()
                except EOFError:
                    break
                action = _STDIN_MAP.get(raw)
                if action is None:
                    print("  Unknown: '{}'. Try: f b l r s q".format(raw))
                    continue
                if action == "quit":
                    break
                self._apply_action(action)
                print("  -> {}".format(action))
        except KeyboardInterrupt:
            pass
        finally:
            self._motors.stop()
            self._motors.close()
            logger.info("Teleop stopped.")

    def _apply_action(self, action: str) -> None:
        turn_speed = self.speed * self.turn_gain
        dispatch = {
            "forward": lambda: self._motors.forward(self.speed),
            "backward": lambda: self._motors.backward(self.speed),
            "left": lambda: self._motors.turn_left(turn_speed),
            "right": lambda: self._motors.turn_right(turn_speed),
            "stop": self._motors.stop,
        }
        dispatch.get(action, self._motors.stop)()
        sys.stdout.write("\r[teleop] {:<10}  speed={:.2f}  ".format(action, self.speed))
        sys.stdout.flush()
