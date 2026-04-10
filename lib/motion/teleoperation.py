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
from pydantic import BaseModel, Field, validator

from lib.camera import Camera, MjpegServer
from lib.motor import MotorController
from lib.network import get_wifi_ip

_ARROW_MAP = {
    b"\x1b[A": "forward",  # Up
    b"\x1b[B": "backward",  # Down
    b"\x1b[C": "right",  # Right
    b"\x1b[D": "left",  # Left
}

# How long (seconds) after the last keypress before we treat it as "released"
_KEY_TIMEOUT: float = 0.15
_SLEEP_TIME: float = 0.05
_TIME_OUT: float = 0.05


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
            logger.debug(f"pynput backend '{backend}' loaded OK.")
            return kb
        except Exception as exc:
            errors[backend] = str(exc)
            for mod in list(sys.modules.keys()):
                if "pynput" in mod:
                    del sys.modules[mod]

    diag = "\n".join("  {}: {}".format(b, e) for b, e in errors.items())
    raise RuntimeError(
        f"pynput could not find a working backend.\n\nBackends tried:\n{diag}\n\n"
        "Try: sudo modprobe uinput && sudo chmod a+rw /dev/uinput"
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


class TeleopConfig(BaseModel):
    """
    Robot teleoperation controller config

    Args:
        speed:     Linear motor speed [0.0, 1.0]
        turn_gain: Differential turn gain [0.0, 1.0]
        mode:      "arrows" | "pynput" | "stdin" | "auto"
        stream:      Start a background MJPEG camera stream while driving
        stream_port: Port for the MJPEG server (default 8080)

    """

    speed: float = Field(0.3, ge=0.0, le=1.0)
    turn_gain: float = Field(0.5, ge=0.0, le=1.0)
    mode: str = "arrows"
    stream: bool = False
    stream_port: int = Field(8080, gt=1024, lt=65535)

    @validator("mode")
    def mode_must_be_valid(cls, v):
        allowed = {"auto", "arrows", "pynput", "stdin"}
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}")
        return v


class TeleopController:
    """
    Keyboard-driven teleoperation controller.

    All parameters are validated by TeleopConfig before reaching here.
    Construct via TeleopController(**config.dict()) — do not pass raw
    argparse values directly.
    """

    def __init__(self, **kwargs):
        self._config = TeleopConfig(**kwargs)
        self._motors = MotorController()
        self._running = False
        self._server = None
        ip = get_wifi_ip()
        self._nano_ip: str = ip if ip is not None else "<nano-ip>"

    def run(self) -> None:
        if self._config.mode == "auto":
            self._run_auto()
        elif self._config.mode == "arrows":
            self._run_arrows()
        elif self._config.mode == "pynput":
            self._run_pynput()
        elif self._config.mode == "stdin":
            self._run_stdin()
        else:
            raise ValueError(f"Unknown teleop mode: {self._config.mode}")

    def _open_camera(self) -> Camera:
        """Open the single shared camera instance."""
        cam = Camera()
        cam.open()
        return cam

    def _close_camera(self, cam: Camera) -> None:
        """Stop the MJPEG server first, then release the camera."""
        if self._server is not None:
            self._server.stop()
            self._server = None
        cam.release()

    def _start_stream(self) -> None:
        """Start the MJPEG server pointed at the shared frame buffer."""
        self._server = MjpegServer(port=self._config.stream_port)
        self._server.start()

    # Mode: auto
    def _run_auto(self) -> None:
        """Try modes in order: arrows -> pynput -> stdin."""
        # arrows works everywhere a TTY is available
        if sys.stdin.isatty():
            self._run_arrows()
            return
        # Try pynput
        if not (os.environ.get("SSH_CLIENT") or os.environ.get("SSH_TTY")):
            try:
                _import_keyboard()
                self._run_pynput()
                return
            except RuntimeError as exc:
                logger.warning(f"pynput unavailable ({exc}), using stdin.")
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
        if self._config.stream:
            _HELP += f"         Camera stream -> http://{(self._nano_ip)}:{self._config.stream_port}/stream\n"
        print(_HELP)

        self._motors.open()
        cam = None

        # Start MJPEG server only after camera is confirmed open
        _stop_capture = threading.Event()
        if self._config.stream:
            cam = self._open_camera()
            self._start_stream()
            self._start_capture_thread(cam, _stop_capture)

        self._running = True
        stop_timer = None  # threading.Timer
        _NUMBER_BYTES_TO_READ: int = 3

        def _schedule_stop() -> None:
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
                ready, _, _ = select.select([sys.stdin], [], [], _TIME_OUT)
                if not ready:
                    continue

                chunk = os.read(fd, _NUMBER_BYTES_TO_READ)

                if chunk in (b"q", b"Q", b"\x03"):
                    break

                action = _ARROW_MAP.get(chunk)
                if action is None:
                    continue

                self._apply_action(action)
                _schedule_stop()

        except Exception as exc:
            logger.error(f"Arrow teleop error: {exc}")
        finally:
            if stop_timer is not None:
                stop_timer.cancel()
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            self._motors.stop()
            self._motors.close()
            _stop_capture.set()
            if cam:
                self._close_camera(cam)
            print("\n[teleop] stopped.")

    # Mode: pynput
    def _run_pynput(self) -> None:
        if os.environ.get("SSH_CLIENT") or os.environ.get("SSH_TTY"):
            raise RuntimeError(
                "pynput mode is not supported over SSH.\n\n"
                "pynput requires either an X display (DISPLAY) or write access to\n"
                "/dev/uinput — neither is available in a plain SSH session.\n\n"
                "Use --mode arrows instead:\n"
                "  nano-explorer motion teleop --mode arrows --stream\n\n"
                "arrows mode uses raw terminal input over your existing SSH TTY\n"
                "and works without a display or root."
            )
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

        def on_press(key)->None:
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

        def on_release(key)->None:
            if key in keymap:
                current_action[0] = "stop"

        logger.info(
            "Teleop (pynput) — arrow keys to drive, SPACE to stop, Q/ESC to quit"
        )
        self._motors.open()
        cam = None
        _stop_capture = threading.Event()
        if self._config.stream:
            cam = self._open_camera()
            self._start_stream()
            self._start_capture_thread(cam, _stop_capture)
            logger.info(
                f"Camera stream -> http://{self._nano_ip}:{self._config.stream_port}/stream"
            )

        with kb.Listener(on_press=on_press, on_release=on_release) as listener:
            try:
                while running[0]:
                    self._apply_action(current_action[0])
                    time.sleep(_SLEEP_TIME)
            except KeyboardInterrupt:
                pass
            finally:
                listener.stop()
                self._motors.stop()
                self._motors.close()
                _stop_capture.set()
                if cam:
                    self._close_camera(cam)
                logger.info("Teleop stopped.")

    # Mode: stdin
    def _run_stdin(self) -> None:
        print(_STDIN_HELP)
        self._motors.open()
        cam = None
        _stop_capture = threading.Event()
        if self._config.stream:
            cam = self._open_camera()
            self._start_stream()
            self._start_capture_thread(cam, _stop_capture)
            print(
                f"Camera stream -> http://{self._nano_ip}:{self._config.stream_port}/stream\n"
            )
        try:
            while True:
                try:
                    raw = input("teleop> ").strip().lower()
                except EOFError:
                    break
                action = _STDIN_MAP.get(raw)
                if action is None:
                    print(f"  Unknown: '{raw}'. Try: f b l r s q")
                    continue
                if action == "quit":
                    break
                self._apply_action(action)
                print(f"  -> {action}")
        except KeyboardInterrupt:
            pass
        finally:
            self._motors.stop()
            self._motors.close()
            _stop_capture.set()
            if cam:
                self._close_camera(cam)
            elif self._server is not None:
                self._server.stop()
            logger.info("Teleop stopped.")

    def _apply_action(self, action: str) -> None:
        turn_speed = self._config.speed * self._config.turn_gain
        dispatch = {
            "forward": lambda: self._motors.forward(self._config.speed),
            "backward": lambda: self._motors.backward(self._config.speed),
            "left": lambda: self._motors.turn_left(turn_speed),
            "right": lambda: self._motors.turn_right(turn_speed),
            "stop": self._motors.stop,
        }
        dispatch.get(action, self._motors.stop)()
        sys.stdout.write(f"\r[teleop] {action:<10}  speed={self._config.speed:.2f}  ")
        sys.stdout.flush()

    def _start_capture_thread(
        self, cam: Camera, stop_event: threading.Event
    ) -> threading.Thread:
        """
        Push frames from cam into the stream buffer in a dedicated thread.
        Decouples capture rate from the input/control loop so the stream
        never freezes while waiting for a keypress or blocking read().
        """

        def _loop():
            while not stop_event.is_set():
                try:
                    frame = cam.read()
                    self._server.frame_buffer.put(frame)
                except Exception:
                    break

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        return t
