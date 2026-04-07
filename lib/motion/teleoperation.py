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

from lib.camera import Camera
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

# ── Shared frame buffer ───────────────────────────────────────────────────────


class _FrameBuffer:
    """Thread-safe single-frame buffer. Written by the main loop, read by MJPEG."""

    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()

    def put(self, frame) -> None:
        with self._lock:
            self._frame = frame.copy()

    def get(self):
        with self._lock:
            return self._frame


# ── Background MJPEG server ───────────────────────────────────────────────────


class _MjpegServer:
    """
    Serves frames from an external _FrameBuffer over MJPEG.

    The camera is owned and operated by the main teleop loop.
    This class only encodes and streams whatever is in the buffer —
    it never opens the camera itself, preventing NVARGUS session conflicts.
    """

    def __init__(self, buf: _FrameBuffer, port: int = 8080, fps: int = 30):
        self._buf = buf
        self._port = port
        self._fps = fps
        self._stop_event = threading.Event()
        self._thread = None

    def start(self) -> None:
        try:
            from aiohttp import web  # noqa: F401
        except ImportError:
            logger.warning(
                "aiohttp not installed — camera stream unavailable. "
                "Run: pip3 install 'aiohttp==3.7.4'"
            )
            return

        import asyncio

        import cv2

        async def _handler(request):
            from aiohttp import web as _web

            response = _web.StreamResponse()
            response.content_type = "multipart/x-mixed-replace; boundary=frame"
            await response.prepare(request)

            while not self._stop_event.is_set():
                frame = self._buf.get()

                # Skip if buffer not yet filled or frame is empty
                if frame is None or frame.size == 0:
                    await asyncio.sleep(0.02)
                    continue

                success, jpeg = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
                )
                if not success or jpeg is None:
                    await asyncio.sleep(0.02)
                    continue

                data = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                )
                try:
                    await response.write(data)
                except Exception:
                    break
                await asyncio.sleep(1.0 / self._fps)

        def _serve():
            async def _main():
                from aiohttp import web as _web

                app = _web.Application()
                app.router.add_get("/stream", _handler)
                runner = _web.AppRunner(app)
                await runner.setup()
                await _web.TCPSite(runner, port=self._port).start()
                while not self._stop_event.is_set():
                    await asyncio.sleep(0.5)
                await runner.cleanup()

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_main())
            finally:
                loop.close()

        self._thread = threading.Thread(target=_serve, daemon=True)
        self._thread.start()
        logger.success(
            "Camera stream started — open "
            "http://<nano-ip>:{}/stream in your browser".format(self._port)
        )

    def stop(self) -> None:
        self._stop_event.set()
        # Thread is a daemon — exits when main thread does.
        # Camera is managed by the caller, not here.


class TeleopController:
    """
    Keyboard-driven teleoperation controller.

    Args:
        speed:     Linear motor speed [0.0, 1.0]
        turn_gain: Differential turn gain [0.0, 1.0]
        mode:      "arrows" | "pynput" | "stdin" | "auto"
        stream:      Start a background MJPEG camera stream while driving
        stream_port: Port for the MJPEG server (default 8080)
    """

    def __init__(
        self,
        speed: float = 0.3,
        turn_gain: float = 0.5,
        mode: str = "arrows",
        stream: bool = False,
        stream_port: int = 8080,
    ):
        self.speed = speed
        self.turn_gain = turn_gain
        self.mode = mode
        self.stream = stream
        self.stream_port = stream_port
        self._motors = MotorController()
        self._running = False
        self._server = None
        self._buf = _FrameBuffer()

    # ── Camera helpers ────────────────────────────────────────────────────────

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
        self._server = _MjpegServer(buf=self._buf, port=self.stream_port)
        self._server.start()

    def _feed_stream(self, cam: Camera) -> None:
        """Read one frame and push it to the stream buffer."""
        if self.stream:
            frame = cam.read()
            self._buf.put(frame)

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
        if self.stream:
            _HELP += "         Camera stream -> http://<nano-ip>:{}/stream\n".format(
                self.stream_port
            )
        print(_HELP)

        self._motors.open()
        cam = self._open_camera()

        # Start MJPEG server only after camera is confirmed open
        if self.stream:
            self._start_stream()

        self._running = True
        stop_timer = None  # threading.Timer
        _NUMBER_BYTES_TO_READ: int = 3

        def _schedule_stop():
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
                self._feed_stream(cam)
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
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
            logger.error("Arrow teleop error: {}".format(exc))
        finally:
            if stop_timer is not None:
                stop_timer.cancel()
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            self._motors.stop()
            self._motors.close()
            self._close_camera(cam)
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
        cam = self._open_camera()

        if self.stream:
            self._start_stream()
            logger.info(
                "Camera stream -> http://<nano-ip>:{}/stream".format(self.stream_port)
            )

        with kb.Listener(on_press=on_press, on_release=on_release) as listener:
            try:
                while running[0]:
                    self._feed_stream(cam)
                    self._apply_action(current_action[0])
                    time.sleep(0.05)
            except KeyboardInterrupt:
                pass
            finally:
                listener.stop()
                self._motors.stop()
                self._motors.close()
                self._close_camera(cam)
                logger.info("Teleop stopped.")

    # Mode: stdin
    def _run_stdin(self) -> None:
        print(_STDIN_HELP)
        self._motors.open()
        cam = self._open_camera() if self.stream else None

        if self.stream:
            self._start_stream()
            print(
                "Camera stream -> http://<nano-ip>:{}/stream\n".format(self.stream_port)
            )
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
                if cam is not None:
                    self._feed_stream(cam)
                self._apply_action(action)
                print("  -> {}".format(action))
        except KeyboardInterrupt:
            pass
        finally:
            self._motors.stop()
            self._motors.close()
            if cam is not None:
                self._close_camera(cam)
            elif self._server is not None:
                self._server.stop()
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
