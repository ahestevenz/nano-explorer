"""
Unified camera abstraction for Jetson Nano.

Supports:
  - CSI camera via GStreamer + nvarguscamerasrc (best performance)
  - USB camera via V4L2 / OpenCV VideoCapture

Usage:
    cam = Camera(width=640, height=480, fps=30, source="csi")
    cam.open()
    frame = cam.read()   # numpy BGR array
    frame, t_ns = cam.read_stamped()   # + CLOCK_MONOTONIC ns, the IMU's clock
    cam.release()

    # Or as a context manager:
    with Camera() as cam:
        frame = cam.read()
"""

import multiprocessing as mp
import threading
import time
from typing import Any, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from lib.imu import _burn, monotonic_ns
from lib.network import get_wifi_ip

# GStreamer pipeline for CSI camera (uses NVIDIA hardware decoder)
_GST_CSI_PIPELINE = (
    "nvarguscamerasrc sensor-mode=2{exposure} ! "
    "video/x-raw(memory:NVMM),width=1920,height=1080,framerate={fps}/1 ! "  # sensor native
    "nvvidconv flip-method=0 ! "
    "video/x-raw,width={w},height={h},format=BGRx ! "  # scale here
    "videoconvert ! "
    "video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
)


# Shortest exposure the IMX219 sensor modes accept, in ns (from the Argus mode list).
_MIN_EXPOSURE_NS = 13000


def _exposure_range(exposure_max_ms: Optional[float]) -> str:
    """nvarguscamerasrc exposuretimerange property (ns), or '' for Argus' default range."""
    if exposure_max_ms is None:
        return ""
    max_ns = max(_MIN_EXPOSURE_NS, int(exposure_max_ms * 1e6))
    return f' exposuretimerange="{_MIN_EXPOSURE_NS} {max_ns}"'


class FrameBuffer:
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


class MjpegServer:
    """
    Serves frames from an external FrameBuffer over MJPEG.

    The camera is owned and operated by the main teleop loop.
    This class only encodes and streams whatever is in the buffer —
    it never opens the camera itself, preventing NVARGUS session conflicts.
    """

    def __init__(self, port: int = 8080, fps: int = 30):
        self.frame_buffer = FrameBuffer()
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

        async def _handler(request):
            from aiohttp import web as _web

            response = _web.StreamResponse()
            response.content_type = "multipart/x-mixed-replace; boundary=frame"
            await response.prepare(request)

            while not self._stop_event.is_set():
                frame = self.frame_buffer.get()

                # Skip if buffer not yet filled or frame is empty
                if frame is None or frame.size == 0:
                    await asyncio.sleep(0.02)
                    continue

                success, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if not success or jpeg is None:
                    await asyncio.sleep(0.02)
                    continue

                data = b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
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
        ip = get_wifi_ip()
        nano_ip = ip if ip is not None else "<nano-ip>"
        logger.success(
            "Camera stream started — open " f"http://{nano_ip}:{self._port}/stream in your browser"
        )

    def stop(self) -> None:
        self._stop_event.set()
        # Thread is a daemon — exits when main thread does.
        # Camera is managed by the caller, not here.


class CameraConfig(BaseModel):
    """
    Args:
        width:     Capture width in pixels.
        height:    Capture height in pixels.
        fps:       Target frames per second.
        source:    "csi" for the CSI ribbon camera, "usb" for a USB webcam.
        device_id: V4L2 device index for USB cameras (ignored for CSI).
        exposure_max_ms: Longest exposure auto-exposure may pick, in ms (CSI only).
                   None keeps Argus' own range (up to ~680 ms). Capping it trades
                   blur for noise: auto-exposure raises gain instead, which ORB
                   copes with better than motion blur.
        timestamps: What read_stamped() stamps a frame with, on CLOCK_MONOTONIC
                   (the IMU's clock):
                   "read" — the time read() returned. Always works, but includes
                            the pipeline's latency and its jitter.
                   "pts"  — the buffer timestamp (CAP_PROP_POS_MSEC), shifted
                            onto CLOCK_MONOTONIC. Closer to the capture time;
                            only valid if `map camera-check` says so.
    """

    width: int = Field(640, gt=0)
    height: int = Field(480, gt=0)
    fps: int = Field(30, gt=0)
    source: str = "csi"
    device_id: int = Field(0, ge=0)
    exposure_max_ms: Optional[float] = Field(None, gt=0)
    timestamps: str = "read"

    @validator("source")
    def source_must_be_valid(cls, v):  # pylint: disable=no-self-argument
        if v not in ("csi", "usb"):
            raise ValueError("source must be 'csi' or 'usb'")
        return v

    @validator("timestamps")
    def timestamps_must_be_valid(cls, v):  # pylint: disable=no-self-argument
        if v not in ("read", "pts"):
            raise ValueError("timestamps must be 'read' or 'pts'")
        return v


class FrameTimes(NamedTuple):
    """
    Both clocks for one frame, in ns.

    read_ns: CLOCK_MONOTONIC right after read() returned.
    pts_ns:  The buffer timestamp OpenCV reports (CAP_PROP_POS_MSEC), on the
             pipeline's own time base; None when the backend reports none (< 0).
    """

    read_ns: int
    pts_ns: Optional[int]


class Camera:
    """
    Thin wrapper around OpenCV VideoCapture, with a GStreamer pipeline
    for the CSI camera and plain V4L2 for USB cameras.
    """

    def __init__(self, **kwargs):
        self._config = CameraConfig(**kwargs)
        self._cap = None
        # read_stamped()'s PTS state; reset by open().
        self._last_pts_ns: Optional[int] = None
        self._pts_offset_ns: Optional[int] = None

    def _reset_clock(self) -> None:
        self._last_pts_ns = None
        self._pts_offset_ns = None

    def open(self) -> None:
        """Open the camera. Raises RuntimeError if it cannot be opened."""
        self._reset_clock()

        if self._config.source == "csi":
            pipeline = _GST_CSI_PIPELINE.format(
                w=self._config.width,
                h=self._config.height,
                fps=self._config.fps,
                exposure=_exposure_range(self._config.exposure_max_ms),
            )
            logger.info("Opening CSI camera via GStreamer pipeline")
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            logger.info(f"Opening USB camera at /dev/video{self._config.device_id}")
            self._cap = cv2.VideoCapture(self._config.device_id)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
            self._cap.set(cv2.CAP_PROP_FPS, self._config.fps)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open {self._config.source} camera. "
                "Check connections and run: v4l2-ctl --list-devices"
            )
        logger.success(
            f"Camera ready: {self._config.width}x{self._config.height} @ {self._config.fps} fps"
        )

    def read(self) -> np.ndarray:
        """
        Read one frame.

        Returns:
            numpy.ndarray: BGR frame, shape (H, W, 3).

        Raises:
            RuntimeError: if the camera is not open or the read fails.
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera is not open. Call open() first.")
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to read frame from camera.")
        return frame

    def read_timed(self) -> Tuple[np.ndarray, FrameTimes]:
        """Read one frame, with its read time and raw buffer timestamp."""
        frame = self.read()
        read_ns = monotonic_ns()
        pts_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        # 0 is a valid first PTS; a backend without timestamps returns -1, or 0
        # on every frame (which read_stamped() and the check reject as not increasing).
        pts_ns = int(round(pts_ms * 1e6)) if np.isfinite(pts_ms) and pts_ms >= 0 else None
        return frame, FrameTimes(read_ns, pts_ns)

    def read_stamped(self) -> Tuple[np.ndarray, int]:
        """
        Read one frame and its timestamp on CLOCK_MONOTONIC, in ns — the clock
        the IMU samples use (lib/imu.py), so the two streams can be matched.

        With timestamps="pts", the buffer timestamp is shifted onto
        CLOCK_MONOTONIC by the smallest (read time - PTS) seen so far: that is
        the pipeline's clock offset plus its shortest latency, and settles
        within the first frames. Its constant part is what Kalibr's time-offset
        estimate takes out.

        Raises:
            RuntimeError: as read(); or with timestamps="pts", if a frame comes
                without a buffer timestamp or with one that doesn't increase
                (switch to timestamps="read").
        """
        frame, times = self.read_timed()
        if self._config.timestamps == "read":
            return frame, times.read_ns

        if times.pts_ns is None or (
            self._last_pts_ns is not None and times.pts_ns <= self._last_pts_ns
        ):
            raise RuntimeError(
                f"Camera buffer timestamp unusable (got {times.pts_ns}, previous "
                f"{self._last_pts_ns}); use timestamps='read'."
            )
        self._last_pts_ns = times.pts_ns
        lag = times.read_ns - times.pts_ns
        if self._pts_offset_ns is None or lag < self._pts_offset_ns:
            self._pts_offset_ns = lag
        return frame, times.pts_ns + self._pts_offset_ns

    def release(self) -> None:
        """Release the camera resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()


# Frame timing check (`nano-explorer map camera-check`)

# (read time - PTS) steadier than this means POS_MSEC is just the pipeline
# position at the time of the query, i.e. the read time again, not the capture.
_LAG_STD_FLOOR_MS = 0.05


class IntervalSummary(NamedTuple):
    """Frame-to-frame intervals, in ms. late: intervals over 1.5x the frame period."""

    mean_ms: float
    std_ms: float
    p99_ms: float
    max_ms: float
    late: int


class CameraTiming(NamedTuple):
    """
    Result of analyze_frame_times().

    read:      intervals between read times.
    pts:       intervals between buffer timestamps; None if they're unusable.
    lag_std_ms / lag_spread_ms: jitter of (read time - PTS): its std and its
               p99 above the minimum. None if the PTS is unusable.
    verdict:   the CameraConfig.timestamps value to use, "pts" or "read".
    reason:    why.
    """

    frames: int
    read: IntervalSummary
    pts: Optional[IntervalSummary]
    lag_std_ms: Optional[float]
    lag_spread_ms: Optional[float]
    verdict: str
    reason: str


def _intervals(ts_ns: List[int], period_ms: float) -> IntervalSummary:
    dt = np.diff(np.asarray(ts_ns, dtype=np.int64)) / 1e6
    return IntervalSummary(
        mean_ms=float(dt.mean()),
        std_ms=float(dt.std()),
        p99_ms=float(np.percentile(dt, 99)),
        max_ms=float(dt.max()),
        late=int((dt > 1.5 * period_ms).sum()),
    )


def analyze_frame_times(times: List[FrameTimes], fps: float) -> CameraTiming:
    """
    Decide which frame timestamp to trust, from the FrameTimes of a recording.

    The buffer timestamp is worth using only if it is a real capture time:
    present and increasing on every frame, steadier than the read time, and not
    moving in lock-step with it (which would mean it's the query time).
    """
    if len(times) < 3:
        raise ValueError(f"need at least 3 frames, got {len(times)}")
    period_ms = 1000.0 / fps
    read = _intervals([t.read_ns for t in times], period_ms)

    def _read_only(reason: str) -> CameraTiming:
        return CameraTiming(len(times), read, None, None, None, "read", reason)

    pts_ns = [t.pts_ns for t in times]
    if any(p is None for p in pts_ns):
        return _read_only("the backend reports no buffer timestamp (POS_MSEC < 0)")
    if not any(pts_ns):
        return _read_only("the backend reports no buffer timestamp (POS_MSEC is always 0)")
    if any(b <= a for a, b in zip(pts_ns, pts_ns[1:])):
        return _read_only("buffer timestamps don't increase on every frame")

    pts = _intervals(pts_ns, period_ms)
    lag = np.array([t.read_ns - t.pts_ns for t in times], dtype=np.int64) / 1e6
    lag_std = float(lag.std())
    lag_spread = float(np.percentile(lag, 99) - lag.min())
    result = CameraTiming(len(times), read, pts, lag_std, lag_spread, "read", "")

    if lag_std < _LAG_STD_FLOOR_MS:
        return result._replace(
            reason="POS_MSEC moves in lock-step with the read time: it's the pipeline "
            "position when queried, not the capture time"
        )
    if pts.std_ms >= 0.5 * read.std_ms:
        return result._replace(reason="buffer timestamps are no steadier than the read time")
    return result._replace(
        verdict="pts",
        reason=f"buffer timestamps are {read.std_ms / max(pts.std_ms, 1e-6):.0f}x steadier "
        "than the read time",
    )


class CameraCheckConfig(BaseModel):
    """
    Args:
        seconds:  How long to record frame times, after the warm-up.
        warmup_s: Frames to drop first, while Argus settles exposure.
        width / height / fps / exposure_max_ms: As CameraConfig; keep them as
                  the SLAM run will use them.
        load:     Keep every other core busy, as ORB-SLAM3 does.
    """

    seconds: float = Field(20.0, gt=0)
    warmup_s: float = Field(2.0, ge=0)
    width: int = Field(640, gt=0)
    height: int = Field(480, gt=0)
    fps: int = Field(30, gt=0)
    exposure_max_ms: Optional[float] = Field(None, gt=0)
    load: bool = False


class CameraCheck:
    """
    Measures the CSI camera's frame timing (`nano-explorer map camera-check`)
    and says which timestamp read_stamped() should use.

    What to look for:
      - verdict: "pts" means the buffer timestamp is a capture time and should
        be used (CameraConfig timestamps="pts"); "read" means stick to the read
        time.
      - read interval std / p99: the jitter left in read-time stamps. Kalibr
        removes a constant offset, not this.
      - lag spread: how far the read time strays from the capture time.
      - late intervals: dropped frames (appsink drops when the reader falls behind).
    """

    def __init__(self, **kwargs: Any):
        self._config = CameraCheckConfig(**kwargs)

    def _collect(self) -> List[FrameTimes]:
        cfg = self._config
        times: List[FrameTimes] = []
        with Camera(
            width=cfg.width, height=cfg.height, fps=cfg.fps, exposure_max_ms=cfg.exposure_max_ms
        ) as cam:
            t_start = time.monotonic()
            while time.monotonic() - t_start < cfg.warmup_s:
                cam.read()
            logger.info(f"Recording frame times for {cfg.seconds:g} s...")
            t_end = time.monotonic() + cfg.seconds
            while time.monotonic() < t_end:
                times.append(cam.read_timed()[1])
        return times

    def run(self) -> CameraTiming:
        cfg = self._config
        stop_load = mp.Event()
        burners = []
        if cfg.load:
            for _ in range(max(1, mp.cpu_count() - 1)):
                p = mp.Process(target=_burn, args=(stop_load,), daemon=True)
                p.start()
                burners.append(p)
            logger.info(f"CPU load: {len(burners)} busy processes")
        try:
            times = self._collect()
        finally:
            stop_load.set()
            for p in burners:
                p.join(timeout=1.0)

        res = analyze_frame_times(times, cfg.fps)
        r = res.read
        logger.info(f"frames:        {res.frames} ({res.frames / cfg.seconds:.1f} fps)")
        logger.info(
            f"read interval: mean {r.mean_ms:.2f} ms, std {r.std_ms:.2f}, "
            f"p99 {r.p99_ms:.2f}, max {r.max_ms:.2f}, late {r.late}"
        )
        if res.pts is not None:
            p = res.pts
            logger.info(
                f"PTS interval:  mean {p.mean_ms:.2f} ms, std {p.std_ms:.2f}, "
                f"p99 {p.p99_ms:.2f}, max {p.max_ms:.2f}, late {p.late}"
            )
            logger.info(
                f"read - PTS:    std {res.lag_std_ms:.2f} ms, "
                f"p99 above min {res.lag_spread_ms:.2f} ms"
            )
        logger.success(f"verdict: timestamps='{res.verdict}' — {res.reason}")
        return res
