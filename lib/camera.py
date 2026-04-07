"""
Unified camera abstraction for Jetson Nano.

Supports:
  - CSI camera via GStreamer + nvarguscamerasrc (best performance)
  - USB camera via V4L2 / OpenCV VideoCapture

Usage:
    cam = Camera(width=640, height=480, fps=30, source="csi")
    cam.open()
    frame = cam.read()   # numpy BGR array
    cam.release()

    # Or as a context manager:
    with Camera() as cam:
        frame = cam.read()
"""

import threading

from loguru import logger

# GStreamer pipeline for CSI camera (uses NVIDIA hardware decoder)
_GST_CSI_PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw,width={w},height={h},format=BGRx ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! appsink drop=1"
)


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
    Serves frames from an external _FrameBuffer over MJPEG.

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

        import cv2

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


class Camera:
    """
    Thin wrapper around OpenCV VideoCapture, with a GStreamer pipeline
    for the CSI camera and plain V4L2 for USB cameras.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        source: str = "csi",
        device_id: int = 0,
    ):
        """
        Args:
            width:     Capture width in pixels.
            height:    Capture height in pixels.
            fps:       Target frames per second.
            source:    "csi" for the CSI ribbon camera, "usb" for a USB webcam.
            device_id: V4L2 device index for USB cameras (ignored for CSI).
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.source = source
        self.device_id = device_id
        self._cap = None

    def open(self) -> None:
        """Open the camera. Raises RuntimeError if it cannot be opened."""
        import cv2

        if self.source == "csi":
            pipeline = _GST_CSI_PIPELINE.format(
                w=self.width, h=self.height, fps=self.fps
            )
            logger.info("Opening CSI camera via GStreamer pipeline")
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            logger.info(f"Opening USB camera at /dev/video{self.device_id}")
            self._cap = cv2.VideoCapture(self.device_id)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open {self.source} camera. "
                "Check connections and run: v4l2-ctl --list-devices"
            )
        logger.success(f"Camera ready: {self.width}x{self.height} @ {self.fps} fps")

    def read(self):
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
