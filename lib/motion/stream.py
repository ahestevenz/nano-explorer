"""
lib/basic/stream.py

Camera streaming over Wi-Fi in three modes:

  mjpeg  — aiohttp server pushing MJPEG frames; open in any browser at
            http://<nano-ip>:<port>/stream
  opencv — local OpenCV window (useful when connected via HDMI/VNC)

JetPack 4.6.1 compatible:
    aiohttp   3.7.4  (last version supporting Python 3.6)
    OpenCV    4.1.1  (bundled with JetPack)
"""

import asyncio
import threading

from loguru import logger

# cv2 imported lazily inside each streaming method
from lib.camera import Camera


class _FrameBuffer:
    """Thread-safe single-frame ring buffer."""

    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()

    def put(self, frame) -> None:
        with self._lock:
            self._frame = frame

    def get(self):
        with self._lock:
            return self._frame


class CameraStreamer:
    """
    Multi-mode camera streamer.

    Args:
        mode:   "mjpeg" | "opencv" | "webrtc"
        port:   Server port (mjpeg / webrtc only)
        width:  Capture / stream width
        height: Capture / stream height
        fps:    Target frame rate
    """

    def __init__(
        self,
        mode: str = "mjpeg",
        port: int = 8080,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.mode = mode
        self.port = port
        self.width = width
        self.height = height
        self.fps = fps
        self._buf = _FrameBuffer()
        self._cam = Camera(width=width, height=height, fps=fps)

    def run(self) -> None:
        logger.info(f"Starting camera stream — mode={self.mode}")
        dispatch = {
            "mjpeg": self._run_mjpeg,
            "opencv": self._run_opencv,
        }
        fn = dispatch.get(self.mode)
        if fn is None:
            raise ValueError(f"Unknown stream mode: {self.mode}")
        fn()

    def _capture_loop(self, stop_event: threading.Event) -> None:
        with self._cam:
            while not stop_event.is_set():
                frame = self._cam.read()
                self._buf.put(frame)

    def _run_mjpeg(self) -> None:
        try:
            from aiohttp import web
        except ImportError:
            raise RuntimeError(
                "aiohttp is required for MJPEG streaming. "
                "Run: pip3 install 'aiohttp==3.7.4'"
            )

        stop_event = threading.Event()
        cap_thread = threading.Thread(
            target=self._capture_loop, args=(stop_event,), daemon=True
        )
        cap_thread.start()

        async def _mjpeg_handler(request):
            response = web.StreamResponse()
            response.content_type = "multipart/x-mixed-replace; boundary=frame"
            await response.prepare(request)

            while True:
                frame = self._buf.get()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                import cv2

                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                data = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                )
                try:
                    await response.write(data)
                except Exception:
                    break
                await asyncio.sleep(1.0 / self.fps)

        app = web.Application()
        app.router.add_get("/stream", _mjpeg_handler)

        logger.success(
            f"MJPEG server running — open http://<nano-ip>:{self.port}/stream"
        )
        try:
            web.run_app(app, port=self.port)
        finally:
            stop_event.set()

    def _run_opencv(self) -> None:
        import cv2

        logger.info("OpenCV display mode — press 'q' to quit")
        with self._cam:
            while True:
                frame = self._cam.read()
                cv2.imshow("nano-explorer | camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cv2.destroyAllWindows()
