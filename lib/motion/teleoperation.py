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

import collections
import os
import select
import sys
import threading
import time
from typing import Any, Deque, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.camera_motion_mixin import CameraMotionMixIn
from lib.motor import MOTOR_LOCK, MotorController

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
_NUMBER_BYTES_TO_READ: int = 3

# How long (seconds) to hold onto an incomplete escape sequence (e.g. only the
# leading \x1b has arrived so far) before giving up and discarding it as stale —
# covers a lone ESC keypress, or a genuinely dropped byte.
_PARTIAL_SEQ_TIMEOUT: float = 0.1

# Auto-recovery (used by SLAM when tracking is lost — see TeleopController.request_recovery):
# replay the operator's most recent moves in reverse so the camera swings back
# toward the view the map was built from.
_INVERSE_ACTION = {"forward": "backward", "backward": "forward", "left": "right", "right": "left"}
# Keypress history kept for recovery; only the last few seconds can ever be replayed.
_HISTORY_KEEP_S: float = 5.0
# Default upper bound on how much motion one recovery replays. ORB-SLAM3
# monocular gives up relocalizing 3 s after tracking is lost by default, so the
# replay has to fit before that; SLAM passes a longer limit when the backend's
# window is longer (System.RecentlyLostSeconds).
_RECOVERY_MAX_S: float = 2.0
# How long without a keypress before the operator counts as "hands off" and the
# replay may start — replaying while a key is still held would fight the operator.
_RECOVERY_IDLE_S: float = 0.3
# Motor commands are re-sent at this interval during a replay step, the same way
# a held key re-sends them, so a stray stop from the key-release timer only
# interrupts a step for one tick.
_RECOVERY_TICK_S: float = 0.05


def _segments_from_events(
    events: List[Tuple[float, str]], since: float, until: float, key_timeout: float
) -> List[Tuple[str, float]]:
    """
    Turn timestamped keypresses into (action, duration) motion segments, oldest first.

    Each keypress drives the motors until the next keypress or until the
    key-release timeout fires, whichever comes first — the same rule
    _run_arrows uses. Consecutive presses of the same key merge into one segment.
    Only presses inside [since, until] are counted.
    """
    inside = [(t, a) for t, a in events if since <= t <= until]
    segments: List[Tuple[str, float]] = []
    for i, (t, action) in enumerate(inside):
        nxt = inside[i + 1][0] if i + 1 < len(inside) else until
        duration = max(0.0, min(nxt - t, key_timeout, until - t))
        if segments and segments[-1][0] == action:
            segments[-1] = (action, segments[-1][1] + duration)
        else:
            segments.append((action, duration))
    return [(a, d) for a, d in segments if d > 0]


def _reverse_plan(segments: List[Tuple[str, float]], max_total: float) -> List[Tuple[str, float]]:
    """
    Undo plan for motion segments: newest first, each action inverted, and cut
    off once max_total seconds are planned (keeping the most recent motion,
    which is what moved the camera off the mapped view).
    """
    plan: List[Tuple[str, float]] = []
    remaining = max_total
    for action, duration in reversed(segments):
        inverse = _INVERSE_ACTION.get(action)
        if inverse is None or remaining <= 0:
            continue
        step = min(duration, remaining)
        plan.append((inverse, step))
        remaining -= step
    return plan


def _import_keyboard() -> Any:
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

    diag = "\n".join(f"  {b}: {e}" for b, e in errors.items())
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
    def mode_must_be_valid(cls, v: str) -> str:  # pylint: disable=no-self-argument
        allowed = {"auto", "arrows", "pynput", "stdin"}
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}")
        return v


class _ArrowKeyReader:
    """
    Assembles raw terminal bytes into one resolved teleop action per call.

    Arrow keys arrive as a 3-byte escape sequence (\x1b[A etc.), but a single
    os.read() call isn't guaranteed to return all 3 bytes at once — under load
    (e.g. a CPU-heavy vision/SLAM loop sharing this process) it can wake up
    after only 1 or 2 bytes have arrived. This buffers across reads until a
    full sequence (or a single-byte q/Q/Ctrl-C) is assembled, instead of
    comparing a possibly-partial chunk directly and silently dropping the key.
    """

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._pending = b""
        self._pending_since: Optional[float] = None

    def read_action(self) -> Optional[str]:
        """One polling cycle: returns a resolved action, "__quit__", or None."""
        ready, _, _ = select.select([sys.stdin], [], [], _TIME_OUT)
        if not ready:
            if self._pending and time.time() - self._pending_since > _PARTIAL_SEQ_TIMEOUT:
                self._pending, self._pending_since = b"", None
            return None

        self._pending += os.read(self._fd, _NUMBER_BYTES_TO_READ - len(self._pending))
        if self._pending_since is None:
            self._pending_since = time.time()

        if self._pending[:1] != b"\x1b":
            chunk, self._pending, self._pending_since = self._pending, b"", None
            return "__quit__" if chunk in (b"q", b"Q", b"\x03") else None

        if len(self._pending) < _NUMBER_BYTES_TO_READ:
            return None  # escape sequence still incomplete — wait for the rest

        chunk, self._pending, self._pending_since = self._pending, b"", None
        return _ARROW_MAP.get(chunk)


class TeleopController(CameraMotionMixIn):
    """
    Keyboard-driven teleoperation controller.

    All parameters are validated by TeleopConfig before reaching here.
    Construct via TeleopController(**config.dict()) — do not pass raw
    argparse values directly.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._config = TeleopConfig(**kwargs)
        self._motors = MotorController()
        # Operator keypresses (timestamp, action) for auto-recovery. Recovery's own
        # replayed moves are never recorded, so a replay can't be replayed back.
        self._history: Deque[Tuple[float, str]] = collections.deque()
        self._history_lock = threading.Lock()
        self._last_key_ts = 0.0
        self._recover_cancel = threading.Event()
        self._recover_thread: Optional[threading.Thread] = None
        # Whether a cancelled replay should stop the motors on its way out: yes
        # when SLAM relocalized, no when the operator took over (their key is
        # already driving, and a late stop would cancel it).
        self._recover_stop_on_exit = True

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
    def _run_arrows(self, stop_event: Optional[threading.Event] = None) -> None:
        """
        Read raw terminal bytes.  Arrow keys produce 3-byte escape sequences.
        A background timer stops the robot if no key arrives within _KEY_TIMEOUT.

        Args:
            stop_event: Optional shared threading.Event. When provided (vision+teleop
                        mode) both sides signal each other through this event. When
                        omitted (standalone teleop) an internal event is used.
        """
        print(
            "\n[teleop] Arrow keys to drive  |  q = quit\n"
            "         Hold key -> move  |  Release -> stop\n"
        )

        import termios
        import tty

        self._motors.open()
        cam = None
        _stop = stop_event if stop_event is not None else threading.Event()

        if self._config.stream:
            cam = self._open_camera()
            self._start_stream(cam=cam, stop_event=_stop, stream_port=self._config.stream_port)

        stop_timer = None

        def _schedule_stop() -> None:
            nonlocal stop_timer
            if stop_timer is not None:
                stop_timer.cancel()
            stop_timer = threading.Timer(
                _KEY_TIMEOUT, self._release_stop, args=(self._last_key_ts,)
            )
            stop_timer.daemon = True
            stop_timer.start()

        fd = sys.stdin.fileno()
        key_reader = _ArrowKeyReader(fd)

        # Cosmetic: fix log line alignment in raw terminal mode
        # tty.setraw() strips carriage returns from stderr, causing loguru output
        # to start mid-line when the robot is idle. We temporarily replace the
        # default sink with one that prepends \r to reset the cursor to column 0.
        def _raw_sink(message: str) -> None:
            sys.stderr.write("\r" + str(message))
            sys.stderr.flush()

        logger.remove()
        raw_id = logger.add(_raw_sink, colorize=True)
        old = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)
            while not _stop.is_set():
                action = key_reader.read_action()
                if action is None:
                    continue
                if action == "__quit__":
                    _stop.set()
                    break
                if self.is_recovering():
                    # Any arrow key hands control straight back to the operator. Wait
                    # for the replay thread to finish its current step, so its last
                    # motor command can't land after the operator's.
                    self.cancel_recovery(stop_motors=False)
                    self._recover_thread.join(timeout=4 * _RECOVERY_TICK_S)
                    logger.info("[recover] cancelled — manual control")
                self._record_action(action)
                self._apply_action(action)
                _schedule_stop()
        except Exception as exc:
            logger.error(f"Arrow teleop error: {exc}")
        finally:
            if stop_timer is not None:
                stop_timer.cancel()
            logger.remove(raw_id)
            logger.add(sys.stderr, colorize=True)  # restore default sink
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            self._motors.stop()
            self._motors.close()
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

        def on_press(key: Any) -> None:
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

        def on_release(key: Any) -> None:
            if key in keymap:
                current_action[0] = "stop"

        logger.info("Teleop (pynput) — arrow keys to drive, SPACE to stop, Q/ESC to quit")
        self._motors.open()
        cam = None
        _stop_capture = threading.Event()
        if self._config.stream:
            cam = self._open_camera()
            self._start_stream(
                cam=cam, stop_event=_stop_capture, stream_port=self._config.stream_port
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
            self._start_stream(
                cam=cam, stop_event=_stop_capture, stream_port=self._config.stream_port
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

    def _drive(self, action: str) -> None:
        """Send the motor command for one action, without logging it."""
        turn_speed = self._config.speed * self._config.turn_gain
        dispatch = {
            "forward": lambda: self._motors.forward(self._config.speed),
            "backward": lambda: self._motors.backward(self._config.speed),
            "left": lambda: self._motors.turn_left(turn_speed),
            "right": lambda: self._motors.turn_right(turn_speed),
            "stop": self._motors.stop,
        }
        dispatch.get(action, self._motors.stop)()

    def _release_stop(self, key_ts: float) -> None:
        """
        Key-release stop, run by the timer _run_arrows schedules after each key.

        Timer.cancel() can't stop a timer that is already firing, so a stop for
        an earlier key could otherwise land right after a newer key's command.
        Skip it if any key arrived after the one it was scheduled for. Checked
        under MOTOR_LOCK, so a key recorded before its command is applied is
        always seen.
        """
        with MOTOR_LOCK:
            if self._last_key_ts > key_ts:
                return
            self._motors.stop()

    def _record_action(self, action: str, now: Optional[float] = None) -> None:
        """Remember an operator keypress for auto-recovery, dropping old entries."""
        now = time.time() if now is None else now
        with self._history_lock:
            self._history.append((now, action))
            while self._history and now - self._history[0][0] > _HISTORY_KEEP_S:
                self._history.popleft()
            self._last_key_ts = now

    # Auto-recovery
    def is_recovering(self) -> bool:
        return self._recover_thread is not None and self._recover_thread.is_alive()

    def cancel_recovery(self, stop_motors: bool = True) -> None:
        """
        Stop a pending or running recovery. stop_motors=False leaves the motors
        alone on the way out — for when the operator takes over and is already
        driving.
        """
        self._recover_stop_on_exit = stop_motors
        self._recover_cancel.set()

    def request_recovery(self, since: float, max_total: float = _RECOVERY_MAX_S) -> bool:
        """
        Start auto-recovery in the background: wait until the operator releases
        the keys, then replay their moves since `since` in reverse (newest first,
        each inverted, at most `max_total` seconds in total), and stop.

        Any arrow key, or cancel_recovery() (e.g. SLAM relocalized), ends it early.
        Returns False if a recovery is already running.
        """
        if self.is_recovering():
            return False
        self._recover_stop_on_exit = True
        self._recover_cancel.clear()
        self._recover_thread = threading.Thread(
            target=self._recovery_worker,
            args=(since, max_total),
            daemon=True,
            name="teleop-recover",
        )
        self._recover_thread.start()
        return True

    def _recovery_worker(self, since: float, max_total: float) -> None:
        while not self._recover_cancel.is_set():
            if time.time() - self._last_key_ts >= _RECOVERY_IDLE_S:
                break
            time.sleep(_RECOVERY_TICK_S)
        if self._recover_cancel.is_set():
            return

        with self._history_lock:
            events = list(self._history)
            until = self._last_key_ts + _KEY_TIMEOUT
        plan = _reverse_plan(_segments_from_events(events, since, until, _KEY_TIMEOUT), max_total)
        if not plan:
            logger.info("[recover] no recent moves to undo — drive back toward the mapped view")
            return
        self._execute_plan(plan)

    def _execute_plan(self, plan: List[Tuple[str, float]]) -> None:
        try:
            for action, duration in plan:
                logger.info(f"[recover] {action:<10} {duration:.2f}s")
                for _ in range(max(1, round(duration / _RECOVERY_TICK_S))):
                    if self._recover_cancel.is_set():
                        return
                    self._drive(action)
                    time.sleep(_RECOVERY_TICK_S)
        finally:
            if self._recover_stop_on_exit:
                self._motors.stop()

    def _apply_action(self, action: str) -> None:
        self._drive(action)
        turn_speed = self._config.speed * self._config.turn_gain
        # left/right send turn_speed (speed * turn_gain) to the motors, not the raw
        # linear speed — log the value actually applied, not always self._config.speed,
        # so this line can be trusted when debugging speed/turn-gain settings.
        applied_speed = turn_speed if action in ("left", "right") else self._config.speed
        sys.stdout.write(f"\r[teleop] {action:<10}  speed={applied_speed:.2f}\r\n")
        sys.stdout.flush()
